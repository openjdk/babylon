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
import java.lang.classfile.constantpool.ConstantPoolBuilder;
import java.lang.classfile.constantpool.PoolEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.constantpool.Utf8Entry;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.type.CoreTypeFactory;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.VarType;

/**
 * <pre>
 * CodeModel_attribute {
 *     u2 attribute_name_index;
 *     u4 attribute_length;
 *     op_info;
 * }
 *
 * op_info {
 *     u2 op_name_index;
 *     u2 op_operands_length;
 *     u2 op_operands[op_operands_length];
 *     u2 op_result_type_index;
 *     u2 op_attributes_length;
 *     op_attribute_info op_attributes_table[op_attributes_length];
 *     u2 nested_bodies_length;
 *     {   u2 body_func_type_index;
 *         block_content_info; // entry block
 *         u2 blocks_length;
 *         {   u2 block_parameters_length;
 *             u2 block_parameter_type_index[block_parameters_length];
 *             block_content_info;
 *         } blocks_table[blocks_length];
 *     } nested_bodies_table[nested_bodies_length];
 * }
 *
 * union op_attribute_info {
 *     value_attribute_info;
 *     location_attribute_info;
 * }
 *
 * value_attribute_info {
 *     u2 attribute_name_index;
 *     u2 attribute_value_index;
 * }
 *
 * location_attribute_info {
 *     u2 location_attribute_name_index;
 *     u2 source_index;
 *     u2 line_number;
 *     u2 column_number;
 * }
 *
 * block_content_info {
 *     u2 ops_length;
 *     op_info ops_table[ops_length];
 *     terminal_op_info;
 * } blocks_table[blocks_length];
 *
 * terminal_op_info {
 *     op_info;
 *     u2 successors_length;
 *     {   u2 successor_block_index;
 *         u2 block_arguments_length;
 *         u2 block_arguments[block_arguments_length];
 *     } successors_table[successors_length]
 * }
 */
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
            buf.writeIndex(buf.constantPool().utf8Entry(NAME));
            int lengthIndex = buf.size();
            buf.writeInt(0);
            writeOp(buf, attr.op, new HashMap<>());
            int written = buf.size() - lengthIndex - 4;
            buf.patchInt(lengthIndex, 4, written);
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
        var extOp = readExOp(buf, terminal, ancestorBody, ancestorBodyBlocks, allValues);
        return ExtendedOp.FACTORY.constructOpOrFail(extOp);
    }

    private static ExternalizableOp.ExternalizedOp readExOp(BufReader buf, boolean terminal, Body.Builder ancestorBody, Block.Builder[] ancestorBodyBlocks, List<Value> allValues) {
        String name = buf.readUtf8();
        List<Value> operands = readValues(buf, allValues);
        TypeElement rType = toType(buf.readEntryOrNull());
        if (name.equals(CoreOp.VarOp.NAME)) rType = VarType.varType(rType);
        Map<String, Object> attributes = readAttributes(buf);
        List<Body.Builder> bodies = readNestedBodies(buf, ancestorBody, allValues);
        return new ExternalizableOp.ExternalizedOp(
                name,
                operands,
                terminal ? readSuccessors(buf, ancestorBodyBlocks, allValues) : List.of(), // successors follow terminal ops
                rType,
                attributes,
                bodies);
    }

    private static void writeOp(BufWriter buf, Op op, Map<Value, Integer> valueMap) {
        // name
        buf.writeIndex(buf.constantPool().utf8Entry(op.opName()));
        // operands
        writeValues(buf, op.operands(), valueMap);
        // result type, saving CP space by unwrapping VarType
        buf.writeIndexOrZero(toEntry(buf.constantPool(), op.resultType() instanceof VarType vt ? vt.valueType() : op.resultType()));
        // attributes
        writeAttributes(buf, op instanceof ExternalizableOp extOp ? extOp.attributes() : Map.of());
        // nested bodies
        writeNestedBodies(buf, op.bodies(), valueMap);

        if (op.result() != null) {
            valueMap.put(op.result(), valueMap.size());
        }

        // @@@ assumption terminating op is only the last one in each block
        if (op instanceof Op.Terminating) {
            writeSuccessors(buf, op.successors(), valueMap);
        }
    }

    private static Map<String, Object> readAttributes(BufReader buf) {
        // number of attributes
        int size = buf.readU2();
        var attrs = new LinkedHashMap<String, Object>(size);
        for (int i = 0; i < size; i++) {
            // attribute name
            String name = buf.readUtf8OrNull();
            // attribute value
            if (ExternalizableOp.ATTRIBUTE_LOCATION.equals(name)) {
                attrs.put(name, new Location(buf.readUtf8OrNull(), buf.readU2(), buf.readU2()));
            } else {
                attrs.put(name, buf.readUtf8OrNull());
            }
        }
        return attrs;
    }

    private static void writeAttributes(BufWriter buf, Map<String, Object> attributes) {
        // number of attributes
        buf.writeU2(attributes.size());
        for (var attre : attributes.entrySet()) {
            // attribute name
            buf.writeIndexOrZero(attre.getKey() == null ? null : buf.constantPool().utf8Entry(attre.getKey()));
            // attribute value
            if (ExternalizableOp.ATTRIBUTE_LOCATION.equals(attre.getKey())) {
                Location loc = switch (attre.getValue()) {
                    case Location l -> l;
                    case String s -> Location.fromString(s);
                    default -> throw new IllegalArgumentException(attre.toString());
                };
                buf.writeIndexOrZero(loc.sourceRef() == null ? null : buf.constantPool().utf8Entry(loc.sourceRef()));
                buf.writeU2(loc.line());
                buf.writeU2(loc.column());
            } else {
                buf.writeIndexOrZero(attre.getValue() == null ? null : buf.constantPool().utf8Entry(attre.getValue().toString()));
            }
        }
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
            bodies[i] = Body.Builder.of(ancestorBody, toFuncType(buf.readEntryOrNull()));
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
            buf.writeIndex(toEntry(buf.constantPool(), body.bodyType()));
            // blocks
            writeBlocks(buf, body.blocks(), valueMap);
        }
    }

    private static void readBlocks(BufReader buf, Body.Builder bob, List<Value> allValues) {
        // number of blocks
        var blocks = new Block.Builder[buf.readU2() + 1]; // entry block is mandatory
        blocks[0] = bob.entryBlock();
        for (int bi = 1; bi < blocks.length; bi++) {
            blocks[bi] = bob.entryBlock().block();
        }
        for (Block.Builder bb : blocks) {
            if (bb.isEntryBlock()) {
                allValues.addAll(bob.entryBlock().parameters());
            } else {
                readBlockParameters(buf, bb, allValues);
            }
            readOps(buf, bb, blocks, allValues);
        }
    }

    private static void writeBlocks(BufWriter buf, List<Block> blocks, Map<Value, Integer> valueMap) {
        // number of blocks - entry block
        buf.writeU2(blocks.size() - 1);
        for (Block block : blocks) {
            // parameters
            if (block.isEntryBlock()) { // @@@ assumption entry block is the first one
                for (var bp : block.parameters()) {
                    valueMap.put(bp, valueMap.size());
                }
            } else {
                writeBlockParameters(buf, block.parameters(), valueMap);
            }
            // ops
            writeOps(buf, block.ops(), valueMap);
        }
    }

    private static void readBlockParameters(BufReader buf, Block.Builder bb, List<Value> allValues) {
        // number of block parameters
        int bpnum = buf.readU2();
        for (int i = 0; i < bpnum; i++) {
            // block parameter type
            allValues.add(bb.parameter(toType(buf.readEntryOrNull())));
        }
    }

    private static void writeBlockParameters(BufWriter buf, List<Block.Parameter> parameters, Map<Value, Integer> valueMap) {
        // number of block parameters
        buf.writeU2(parameters.size());
        for (Block.Parameter bp : parameters) {
            // block parameter type
            buf.writeIndexOrZero(toEntry(buf.constantPool(), bp.type()));
            valueMap.put(bp, valueMap.size());
        }
    }

    private static void readOps(BufReader buf, Block.Builder bb, Block.Builder[] allBlocks, List<Value> allValues) {
        // number of ops
        int opnum = buf.readU2();
        for (int i = 0; i <= opnum; i++) { // +1 terminal op
            // op
            Op op = readOp(buf, i == opnum, bb.parentBody(), allBlocks, allValues);
            bb.op(op);
            if (op.result() != null) {
                allValues.add(op.result());
            }
        }
    }

    private static void writeOps(BufWriter buf, List<Op> ops, Map<Value, Integer> valueMap) {
        // number of ops - mandatory terminal op
        buf.writeU2(ops.size() - 1);
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

    private static FunctionType toFuncType(PoolEntry entry) {
        return switch (entry) {
            case Utf8Entry ue -> {
                var mtd = MethodTypeDesc.ofDescriptor(ue.stringValue());
                yield FunctionType.functionType(JavaType.type(mtd.returnType()), mtd.parameterList().stream().map(JavaType::type).toList());
            }
            case StringEntry se ->
                (FunctionType)CoreTypeFactory.CORE_TYPE_FACTORY.constructType(TypeElement.ExternalizedTypeElement.ofString(se.stringValue()));
            default ->
                throw new IllegalArgumentException(entry.toString());
        };
    }

    private static PoolEntry toEntry(ConstantPoolBuilder cp, FunctionType ftype) {
        if (ftype.returnType() instanceof JavaType jret
                && jret.erasure().equals(jret)
                && ftype.parameterTypes().stream().allMatch(te ->
                        te instanceof JavaType jt && jt.erasure().equals(jt))) {
            // prefer to store as method type descriptor
            return cp.utf8Entry(MethodTypeDesc.of(jret.toNominalDescriptor(), ftype.parameterTypes().stream().map(te -> ((JavaType)te).toNominalDescriptor()).toList()));
        } else {
            // fallback
            return cp.stringEntry(ftype.externalize().toString());
        }
    }

    private static TypeElement toType(PoolEntry entry) {
        return switch (entry) {
            case Utf8Entry ue ->
                JavaType.type(ClassDesc.ofDescriptor(ue.stringValue()));
            case StringEntry se ->
                CoreTypeFactory.CORE_TYPE_FACTORY.constructType(TypeElement.ExternalizedTypeElement.ofString(se.stringValue()));
            case null ->
                JavaType.VOID;
            default ->
                throw new IllegalArgumentException(entry.toString());
        };
    }

    private static PoolEntry toEntry(ConstantPoolBuilder cp, TypeElement type) {
        if (type.equals(JavaType.VOID)) return null;
        return type instanceof JavaType jt && jt.erasure().equals(jt)
                ? cp.utf8Entry(jt.toNominalDescriptor())
                : cp.stringEntry(type.externalize().toString());
    }

    private static final class BufReader {
        private final ClassReader cr;
        private int offset;
        BufReader(ClassReader cr, int offset) {
            this.cr = cr;
            this.offset = offset;
        }

        int readU2() {
            int i = cr.readU2(offset);
            offset += 2;
            return i;
        }

        String readUtf8() {
            String s = cr.readEntry(offset, Utf8Entry.class).stringValue();
            offset += 2;
            return s;
        }

        String readUtf8OrNull() {
            Utf8Entry u = cr.readEntryOrNull(offset, Utf8Entry.class);
            offset += 2;
            return u == null ? null : u.stringValue();
        }

        PoolEntry readEntryOrNull() {
            PoolEntry e = cr.readEntryOrNull(offset);
            offset += 2;
            return e;
        }
    }
}
