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

import java.lang.classfile.ClassReader;
import java.lang.classfile.constantpool.PoolEntry;
import java.lang.classfile.constantpool.StringEntry;
import java.lang.classfile.constantpool.Utf8Entry;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.util.ArrayList;
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
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.type.CoreTypeFactory;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;

final class OpReader {

    private final ClassReader cr;
    private final List<Value> allValues;
    private int offset;

    OpReader(ClassReader cr, int offset) {
        this.cr = cr;
        this.allValues = new ArrayList<>();
        this.offset = offset;
    }

    Op readOp(Body.Builder ancestorBody, Block.Builder[] ancestorBodyBlocks) {
        switch (CodeModelAttribute.OpTag.values()[readU1()]) {
            case AddOp ->
                CoreOp.add(readValue(), readValue());
            case AndOp ->
                CoreOp.and(readValue(), readValue());
            case ArrayLoadOp ->
                CoreOp.arrayLoadOp(readValue(), readValue(), readType());
            case ArrayStoreOp ->
                CoreOp.arrayStoreOp(readValue(), readValue(), readValue());
            case ArrayLengthOp ->
                CoreOp.arrayLength(readValue());
            case AshrOp ->
                CoreOp.ashr(readValue(), readValue());
            case BranchOp ->
                CoreOp.branch(readBlockReference(ancestorBodyBlocks));
            case CastOp ->
                CoreOp.cast(readType(), (JavaType)readType(), readValue());
            case ClosureCallOp ->
                CoreOp.closureCall(readValues());
            case ClosureOp ->
                CoreOp.closure(readNestedBody(ancestorBody));
            case ComplOp ->
                CoreOp.compl(readValue());
            case ConcatOp ->
                CoreOp.concat(readValue(), readValue());
            case ConditionalBranchOp ->
                CoreOp.conditionalBranch(readValue(), readBlockReference(ancestorBodyBlocks), readBlockReference(ancestorBodyBlocks));
            case ConstantOp ->
                CoreOp.constant(readType(), readUtf8OrNull()); // @@@ constant value is serialized as Utf8Entry
            case ConvOp ->
                CoreOp.conv(readType(), readValue());
            case DivOp ->
                CoreOp.div(readValue(), readValue());
            case EqOp ->
                CoreOp.eq(readValue(), readValue());
            case ExceptionRegionEnter ->
                CoreOp.exceptionRegionEnter(readBlockReference(ancestorBodyBlocks), readCatchers(ancestorBodyBlocks));
            case ExceptionRegionExit ->
                CoreOp.exceptionRegionExit(readBlockReference(ancestorBodyBlocks), readCatchers(ancestorBodyBlocks));
            case FieldLoadOp -> {
            }
            case FieldStoreOp -> {
            }
            case FuncCallOp -> {
            }
            case FuncOp -> {
            }
            case GeOp -> {
            }
            case GtOp -> {
            }
            case InstanceOfOp -> {
            }
            case InvokeOp -> {
            }
            case LambdaOp -> {
            }
            case LeOp -> {
            }
            case LshlOp -> {
            }
            case LshrOp -> {
            }
            case LtOp -> {
            }
            case ModOp -> {
            }
            case ModuleOp -> {
            }
            case MonitorEnterOp -> {
            }
            case MonitorExitOp -> {
            }
            case MulOp -> {
            }
            case NegOp -> {
            }
            case NeqOp -> {
            }
            case NewOp -> {
            }
            case NotOp -> {
            }
            case OrOp -> {
            }
            case QuotedOp -> {
            }
            case ReturnOp -> {
            }
            case SubOp -> {
            }
            case ThrowOp -> {
            }
            case TupleLoadOp -> {
            }
            case TupleOp -> {
            }
            case TupleWithOp -> {
            }
            case UnreachableOp -> {
            }
            case VarLoadOp -> {
            }
            case VarStoreOp -> {
            }
            case VarOp -> {
            }
            case XorOp -> {
            }
            case YieldOp -> {
            }
            case JavaBlockOp -> {
            }
            case JavaBreakOp -> {
            }
            case JavaConditionalAndOp -> {
            }
            case JavaConditionalExpressionOp -> {
            }
            case JavaConditionalOrOp -> {
            }
            case JavaContinueOp -> {
            }
            case JavaDoWhileOp -> {
            }
            case JavaEnhancedForOp -> {
            }
            case JavaForOp -> {
            }
            case JavaIfOp -> {
            }
            case JavaLabeledOp -> {
            }
            case JavaSwitchExpressionOp -> {
            }
            case JavaSwitchFallthroughOp -> {
            }
            case JavaSwitchStatementOp -> {
            }
            case MatchAllPatternOp -> {
            }
            case MatchOp -> {
            }
            case RecordPatternOp -> {
            }
            case TypePatternOp -> {
            }
        }
        return null;
    }

    private Map<String, Object> readAttributes()  {
        // number of attributes
        int size = readU2();
        var attrs = new LinkedHashMap<String, Object>(size);
        for (int i = 0; i < size; i++) {
            // attribute name
            String name = readUtf8OrNull();
            // attribute value
            if (ExternalizableOp.ATTRIBUTE_LOCATION.equals(name)) {
                attrs.put(name, new Location(readUtf8OrNull(), readU2(), readU2()));
            } else {
                attrs.put(name, readUtf8OrNull());
            }
        }
        return attrs;
    }

    private List<Value> readValues() {
        // number of values
        var values = new Value[readU2()];
        for (int i = 0; i < values.length; i++) {
            // value by index
            values[i] = allValues.get(readU2());
        }
        return List.of(values);
    }

    private List<Body.Builder> readNestedBodies(Body.Builder ancestorBody) {
        // number of bodies
        var bodies = new Body.Builder[readU2()];
        for (int i = 0; i < bodies.length; i++) {
            bodies[i] = readNestedBody(ancestorBody);
        }
        return List.of(bodies);
    }

    private Body.Builder readNestedBody(Body.Builder ancestorBody) {
        var bb = Body.Builder.of(ancestorBody, toFuncType(readEntryOrNull()));
        readBlocks(bb);
        return bb;
    }

    private void readBlocks(Body.Builder bob) {
        // number of blocks
        var blocks = new Block.Builder[readU2() + 1]; // entry block is mandatory
        blocks[0] = bob.entryBlock();
        for (int bi = 1; bi < blocks.length; bi++) {
            blocks[bi] = bob.entryBlock().block();
        }
        for (Block.Builder bb : blocks) {
            if (bb.isEntryBlock()) {
                allValues.addAll(bob.entryBlock().parameters());
            } else {
                readBlockParameters(bb);
            }
            readOps(bb, blocks);
        }
    }

    private void readBlockParameters(Block.Builder bb) {
        // number of block parameters
        int bpnum = readU2();
        for (int i = 0; i < bpnum; i++) {
            // block parameter type
            allValues.add(bb.parameter(toType(readEntryOrNull())));
        }
    }

    private void readOps(Block.Builder bb, Block.Builder[] allBlocks) {
        // number of ops
        int opnum = readU2();
        for (int i = 0; i <= opnum; i++) { // +1 terminal op
            // op
            Op op = readOp(bb.parentBody(), allBlocks);
            bb.op(op);
            if (op.result() != null) {
                allValues.add(op.result());
            }
        }
    }

    private List<Block.Reference> readSuccessors(Block.Builder[] ancestorBodyBlocks) {
        // number of successors
        var refs = new Block.Reference[readU2()];
        for (int i = 0; i < refs.length; i++) {
            // block from index + arguments
            refs[i] = ancestorBodyBlocks[readU2()].successor(readValues());
        }
        return List.of(refs);
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

    private int readU1() {
        int i = cr.readU1(offset);
        offset++;
        return i;
    }

    private int readU2() {
        int i = cr.readU2(offset);
        offset += 2;
        return i;
    }

    private String readUtf8() {
        String s = cr.readEntry(offset, Utf8Entry.class).stringValue();
        offset += 2;
        return s;
    }

    private String readUtf8OrNull() {
        Utf8Entry u = cr.readEntryOrNull(offset, Utf8Entry.class);
        offset += 2;
        return u == null ? null : u.stringValue();
    }

    private PoolEntry readEntryOrNull() {
        PoolEntry e = cr.readEntryOrNull(offset);
        offset += 2;
        return e;
    }

    private Value readValue() {
        return allValues.get(readU2());
    }

    private TypeElement readType() {
        return toType(readEntryOrNull());
    }

    private Block.Reference readBlockReference(Block.Builder[] ancestorBodyBlocks) {
        return ancestorBodyBlocks[readU2()].successor(readValues());
    }

    private Block.Reference[] readCatchers(Block.Builder[] ancestorBodyBlocks) {
        var catchers = new Block.Reference[readU2()];
        for (int i = 0; i < catchers.length; i++) {
            catchers[i] = ancestorBodyBlocks[readU2()].successor();
        }
        return catchers;
    }
}
