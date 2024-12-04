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
import java.util.List;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.CoreOp.FuncOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.type.CoreTypeFactory;
import jdk.incubator.code.type.FieldRef;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.RecordTypeRef;

final class OpReader {

    private final ClassReader cr;
    private final List<Value> allValues;
    private int offset;

    OpReader(ClassReader cr, int offset) {
        this.cr = cr;
        this.allValues = new ArrayList<>();
        allValues.add(null); // 0-index null value
        this.offset = offset;
    }

    Op readOp(Body.Builder ancestorBody, Block.Builder[] ancestorBodyBlocks) {
        int tag = readU1();
        Location location = Location.NO_LOCATION;
        if (tag == CodeModelAttribute.Tag.LocationAttr.ordinal()) { // tag for location
            location = new Location(readUtf8OrNull(), readU2(), readU2());
            tag = readU1();
        }
        Op op = switch (CodeModelAttribute.Tag.values()[tag]) {
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
                Value receiver = readValue();
                if (receiver == null) {
                    yield CoreOp.fieldLoad(readType(), FieldRef.field(readType(), readUtf8(), readType()));
                } else {
                    yield CoreOp.fieldLoad(readType(), FieldRef.field(readType(), readUtf8(), readType()), receiver);
                }
            }
            case FieldStoreOp -> {
                Value receiver = readValue();
                FieldRef field = FieldRef.field(readType(), readUtf8(), readType());
                if (receiver == null) {
                    yield CoreOp.fieldStore(field, readValue());
                } else {
                    yield CoreOp.fieldStore(field, receiver, readValue());
                }
            }
            case FuncCallOp ->
                CoreOp.funcCall(readUtf8(), readType(), readValues());
            case FuncOp ->
                CoreOp.func(readUtf8(), readNestedBody(ancestorBody));
            case GeOp ->
                CoreOp.ge(readValue(), readValue());
            case GtOp ->
                CoreOp.gt(readValue(), readValue());
            case InstanceOfOp ->
                CoreOp.instanceOf(readType(), readValue());
            case InvokeOp ->
                CoreOp.invoke(CoreOp.InvokeOp.InvokeKind.values()[readU1()],
                              readU1() != 0,
                              readType(),
                              MethodRef.method(readType(), readUtf8(), readFunctionType()),
                              List.of(readValues()));
            case LambdaOp ->
                CoreOp.lambda(readType(), readNestedBody(ancestorBody));
            case LeOp ->
                CoreOp.le(readValue(), readValue());
            case LshlOp ->
                CoreOp.lshl(readValue(), readValue());
            case LshrOp ->
                CoreOp.lshr(readValue(), readValue());
            case LtOp ->
                CoreOp.lt(readValue(), readValue());
            case ModOp ->
                CoreOp.mod(readValue(), readValue());
            case ModuleOp -> {
                var functions = new FuncOp[readU2()];
                for (int i = 0; i < functions.length; i++) {
                    functions[i] = CoreOp.func(readUtf8(), readNestedBody(ancestorBody));
                }
                yield CoreOp.module(functions);
            }
            case MonitorEnterOp ->
                CoreOp.monitorEnter(readValue());
            case MonitorExitOp ->
                CoreOp.monitorExit(readValue());
            case MulOp ->
                CoreOp.mul(readValue(), readValue());
            case NegOp ->
                CoreOp.neg(readValue());
            case NeqOp ->
                CoreOp.neq(readValue(), readValue());
            case NewOp ->
                CoreOp._new(readType(), readFunctionType(), readValues());
            case NotOp ->
                CoreOp.not(readValue());
            case OrOp ->
                CoreOp.or(readValue(), readValue());
            case QuotedOp ->
                CoreOp.quoted(readNestedBody(ancestorBody));
            case ReturnOp -> {
                Value v = readValue();
                if (v == null) {
                    yield CoreOp._return();
                } else {
                    yield CoreOp._return(v);
                }
            }
            case SubOp ->
                CoreOp.sub(readValue(), readValue());
            case ThrowOp ->
                CoreOp._throw(readValue());
            case TupleLoadOp ->
                CoreOp.tupleLoad(readValue(), readU2());
            case TupleOp ->
                CoreOp.tuple(readValues());
            case TupleWithOp -> {
                Value t = readValue();
                Value v = readValue();
                yield CoreOp.tupleWith(t, readU2(), v);
            }
            case UnreachableOp ->
                CoreOp.unreachable();
            case VarLoadOp ->
                CoreOp.varLoad(readValue());
            case VarStoreOp ->
                CoreOp.varStore(readValue(), readValue());
            case VarOp -> {
                Value init = readValue();
                if (init == null) {
                    yield CoreOp.var(readUtf8OrNull(), readType());
                } else {
                    yield CoreOp.var(readUtf8OrNull(), readType(), init);
                }
            }
            case XorOp ->
                CoreOp.xor(readValue(), readValue());
            case YieldOp -> {
                Value v = readValue();
                if (v == null) {
                    yield CoreOp._yield();
                } else {
                    yield CoreOp._yield(v);
                }
            }
            case JavaBlockOp ->
                ExtendedOp.block(readNestedBody(ancestorBody));
            case JavaBreakOp ->
                ExtendedOp._break(readValue());
            case JavaConditionalAndOp ->
                ExtendedOp.conditionalAnd(readNestedBodies(ancestorBody));
            case JavaConditionalExpressionOp ->
                ExtendedOp.conditionalExpression(readType(), readNestedBodies(ancestorBody));
            case JavaConditionalOrOp ->
                ExtendedOp.conditionalOr(readNestedBodies(ancestorBody));
            case JavaContinueOp ->
                ExtendedOp._continue(readValue());
            case JavaDoWhileOp ->
                ExtendedOp.doWhile(readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case JavaEnhancedForOp ->
                ExtendedOp.enhancedFor(readNestedBody(ancestorBody), readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case JavaForOp ->
                ExtendedOp._for(readNestedBody(ancestorBody), readNestedBody(ancestorBody), readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case JavaIfOp ->
                ExtendedOp._if(readNestedBodies(ancestorBody));
            case JavaLabeledOp ->
                ExtendedOp.labeled(readNestedBody(ancestorBody));
            case JavaSwitchExpressionOp ->
                ExtendedOp.switchExpression(readType(), readValue(), readNestedBodies(ancestorBody));
            case JavaSwitchFallthroughOp ->
                ExtendedOp.switchFallthroughOp();
            case JavaSwitchStatementOp ->
                ExtendedOp.switchStatement(readValue(), readNestedBodies(ancestorBody));
            case JavaSynchronizedOp ->
                ExtendedOp.synchronized_(readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case JavaTryOp ->
                ExtendedOp._try(readNestedBody(ancestorBody), readNestedBody(ancestorBody), readNestedBodies(ancestorBody), readNestedBody(ancestorBody));
            case JavaYieldOp -> {
                Value v = readValue();
                if (v == null) {
                    yield ExtendedOp.java_yield();
                } else {
                    yield ExtendedOp.java_yield(v);
                }
            }
            case JavaWhileOp ->
                ExtendedOp._while(readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case MatchAllPatternOp ->
                ExtendedOp.matchAllPattern();
            case MatchOp ->
                ExtendedOp.match(readValue(), readNestedBody(ancestorBody), readNestedBody(ancestorBody));
            case RecordPatternOp ->
                ExtendedOp.recordPattern(RecordTypeRef.recordType(readType(), readRecordComponents()), readValues());
            case TypePatternOp ->
                ExtendedOp.typePattern(readType(), readUtf8());
            default -> throw new UnsupportedOperationException("tag: " + tag);
        };
        if (location != Location.NO_LOCATION) {
            op.setLocation(location);
        }
        return op;
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
        var type = readEntryOrNull();
        if (type == null) return null;
        var bb = Body.Builder.of(ancestorBody, toFuncType(type));
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

    private Value[] readValues() {
        var values = new Value[readU2()];
        for (int i = 0; i < values.length; i++) {
            values[i] = allValues.get(readU2());
        }
        return values;
    }

    private Value readValue() {
        return allValues.get(readU2());
    }

    private RecordTypeRef.ComponentRef[] readRecordComponents() {
        var types = new RecordTypeRef.ComponentRef[readU2()];
        for (int i = 0; i < types.length; i++) {
            types[i] = new RecordTypeRef.ComponentRef(readType(), readUtf8());
        }
        return types;
    }

    private TypeElement readType() {
        return toType(readEntryOrNull());
    }

    private FunctionType readFunctionType() {
        return toFuncType(readEntryOrNull());
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
