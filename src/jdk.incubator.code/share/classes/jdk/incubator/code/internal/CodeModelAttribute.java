/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

package jdk.incubator.code.internal;

import java.lang.classfile.AttributeMapper;
import java.lang.classfile.AttributedElement;
import java.lang.classfile.BufWriter;
import java.lang.classfile.ClassReader;
import java.lang.classfile.CustomAttribute;
import java.lang.classfile.constantpool.Utf8Entry;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.VarType;

import static java.lang.constant.ConstantDescs.CD_void;

public class CodeModelAttribute extends CustomAttribute<CodeModelAttribute>{

    public static final String NAME = "CodeModel";

    public static final AttributeMapper<CodeModelAttribute> MAPPER = new AttributeMapper<>() {

        @Override
        public String name() {
            return NAME;
        }

        @Override
        public CodeModelAttribute readAttribute(AttributedElement enclosing, ClassReader cf, int pos) {
            return new CodeModelAttribute(readOp(new BufReader(cf, pos), false, null, null, new ArrayList<>()));
        }

        @Override
        public void writeAttribute(BufWriter buf, CodeModelAttribute attr) {
            writeOp(buf, attr.op, new HashMap<>());
        }

        @Override
        public AttributeMapper.AttributeStability stability() {
            return AttributeMapper.AttributeStability.CP_REFS;
        }
    };

    public static CodeModelAttribute of(Op op) {
        return new CodeModelAttribute(op);
    }

    private final Op op;

    private CodeModelAttribute(Op op) {
        super(MAPPER);
        this.op = op;
    }

    public Op op() {
        return op;
    }

    private static Op readOp(BufReader buf, boolean terminal, Body.Builder ancestorBody, Block.Builder[] ancestorBodyBlocks, List<Value> allValues) {
        return ExtendedOp.FACTORY.constructOpOrFail(readExOp(buf, terminal, ancestorBody, ancestorBodyBlocks, allValues));
    }

    private static ExternalizableOp.ExternalizedOp readExOp(BufReader buf, boolean terminal, Body.Builder ancestorBody, Block.Builder[] ancestorBodyBlocks, List<Value> allValues) {
        String name = buf.readUtf8();
        List<Value> operands = readValues(buf, allValues);
        String rType = buf.readUtf8OrNull();
        Map<String, Object> attributes = Map.of(); // @@@ attributes
        List<Body.Builder> bodies = readNestedBodies(buf, ancestorBody, allValues);
        return new ExternalizableOp.ExternalizedOp(
                name,
                operands,
                terminal ? readSuccessors(buf, ancestorBodyBlocks, allValues) : List.of(), // successors follow terminal ops
                rType == null ? JavaType.VOID : JavaType.type(ClassDesc.ofDescriptor(rType)),
                attributes,
                bodies);
    }

    private static void writeOp(BufWriter buf, Op op, Map<Value, Integer> valueMap) {
        // name
        buf.writeIndex(buf.constantPool().utf8Entry(op.opName()));
        // operands
        writeValues(buf, op.operands(), valueMap);
        // result type
        ClassDesc rt = toCD(op.resultType());
        buf.writeIndexOrZero(rt == CD_void ? null : buf.constantPool().utf8Entry(rt));
        // @@@ attributes

        // nested bodies
        writeNestedBodies(buf, op.bodies(), valueMap);

        valueMap.put(op.result(), valueMap.size());
    }

    private static List<Value> readValues(BufReader buf, List<Value> allValues) {
        // number of values
        var values = new Value[buf.readU2()];
        for (int i = 0; i < values.length; i++) {
            // value by index
            values[i] = allValues.get(buf.readU2());
        }
        return List.of(values);
    }

    private static void writeValues(BufWriter buf, List<Value> values, Map<Value, Integer> valueMap) {
        // number of values
        buf.writeU2(values.size());
        for (Value v : values) {
            // value index
            buf.writeU2(valueMap.get(v));
        }
    }

    private static List<Body.Builder> readNestedBodies(BufReader buf, Body.Builder ancestorBody, List<Value> allValues) {
        // number of bodies
        var bodies = new Body.Builder[buf.readU2()];
        for (int i = 0; i < bodies.length; i++) {
            // body type
            bodies[i] = Body.Builder.of(ancestorBody, toFuncType(MethodTypeDesc.ofDescriptor(buf.readUtf8())));
            // blocks
            readBlocks(buf, bodies[i], allValues);
        }
        return List.of(bodies);
    }

    private static void writeNestedBodies(BufWriter buf, List<Body> bodies, Map<Value, Integer> valueMap) {
        // number of bodies
        buf.writeU2(bodies.size());
        for (Body body : bodies) {
            // body type
            buf.writeIndex(buf.constantPool().utf8Entry(toMTD(body.bodyType())));
            // blocks
            writeBlocks(buf, body.blocks(), valueMap);
        }
    }

    private static void readBlocks(BufReader buf, Body.Builder bob, List<Value> allValues) {
        // number of blocks
        var blocks = new Block.Builder[buf.readU2()];
        blocks[0] = bob.entryBlock();
        allValues.addAll(bob.entryBlock().parameters());
        for (int bi = 1; bi < blocks.length; bi++) {
            blocks[bi] = bob.entryBlock().block();
            readBlockParameters(buf, blocks[bi], allValues);
        }
        for (Block.Builder bb : blocks) {
            readOps(buf, bb, blocks, allValues);
        }
    }

    private static void writeBlocks(BufWriter buf, List<Block> blocks, Map<Value, Integer> valueMap) {
        // number of blocks
        buf.writeU2(blocks.size());
        for (Block block : blocks) {
            // parameters
            if (block.isEntryBlock()) {
                for (var bp : block.parameters()) {
                    valueMap.put(bp, valueMap.size());
                }
            } else {
                writeBlockParameters(buf, block.parameters(), valueMap);
            }
            // ops
            writeOps(buf, block.ops(), valueMap);
            // successors
            writeSuccessors(buf, block.successors(), valueMap);
        }
    }

    private static void readBlockParameters(BufReader buf, Block.Builder bb, List<Value> allValues) {
        // number of block parameters
        int bpnum = buf.readU2();
        for (int i = 0; i < bpnum; i++) {
            // block parameter type
            allValues.add(bb.parameter(JavaType.type(ClassDesc.ofDescriptor(buf.readUtf8()))));
        }
    }

    private static void writeBlockParameters(BufWriter buf, List<Block.Parameter> parameters, Map<Value, Integer> valueMap) {
        // number of block parameters
        buf.writeU2(parameters.size());
        for (Block.Parameter bp : parameters) {
            // block parameter type
            buf.writeIndex(buf.constantPool().utf8Entry(toCD(bp.type())));
            valueMap.put(bp, valueMap.size());
        }
    }

    private static void readOps(BufReader buf, Block.Builder bb, Block.Builder[] allBlocks, List<Value> allValues) {
        // number of ops
        int opnum = buf.readU2();
        for (int i = 0; i < opnum; i++) {
            // op
            bb.op(readOp(buf, i == opnum - 1, bb.parentBody(), allBlocks, allValues));
        }
    }

    private static void writeOps(BufWriter buf, List<Op> ops, Map<Value, Integer> valueMap) {
        // number of ops
        buf.writeU2(ops.size());
        for (Op op : ops) {
            // op
            writeOp(buf, op, valueMap);
        }
    }

    private static List<Block.Reference> readSuccessors(BufReader buf, Block.Builder[] ancestorBodyBlocks, List<Value> allValues) {
        // number of successors
        var refs = new Block.Reference[buf.readU2()];
        for (int i = 0; i < refs.length; i++) {
            // block from index + arguments
            refs[i] = ancestorBodyBlocks[buf.readU2()].successor(readValues(buf, allValues));
        }
        return List.of(refs);
    }

    private static void writeSuccessors(BufWriter buf, List<Block.Reference> successors, Map<Value, Integer> valueMap) {
        // number of successors
        buf.writeU2(successors.size());
        for (Block.Reference succ : successors) {
            // block index
            buf.writeU2(succ.targetBlock().index());
            // arguments
            writeValues(buf, succ.arguments(), valueMap);
        }
    }

    private static FunctionType toFuncType(MethodTypeDesc mtd) {
        return FunctionType.functionType(JavaType.type(mtd.returnType()), mtd.parameterList().stream().map(JavaType::type).toList());
    }

    private static MethodTypeDesc toMTD(FunctionType ftype) {
        return MethodTypeDesc.of(toCD(ftype.returnType()), ftype.parameterTypes().stream().map(CodeModelAttribute::toCD).toList());
    }

    private static ClassDesc toCD(TypeElement type) {
        return switch (type) {
            case JavaType jt -> jt.toNominalDescriptor();
            case VarType vt -> toCD(vt.valueType());
            default -> throw new IllegalArgumentException(type.toString());
        };
    }

    private static final class BufReader {
        private final ClassReader cr;
        private int offset;
        BufReader(ClassReader cr, int offset) {
            this.cr = cr;
            this.offset = offset;
        }

        int readU2() {
            int i = cr.readInt(offset);
            offset += 2;
            return i;
        }

        String readUtf8() {
            String s = cr.readEntry(offset, Utf8Entry.class).stringValue();
            offset += 2;
            return s;
        }

        String readUtf8OrNull() {
            Utf8Entry u = cr.readEntry(offset, Utf8Entry.class);
            offset += 2;
            return u == null ? null : u.stringValue();
        }
    }
}
