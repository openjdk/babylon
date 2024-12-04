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

import java.lang.classfile.BufWriter;
import java.lang.classfile.constantpool.ConstantPoolBuilder;
import java.lang.classfile.constantpool.PoolEntry;
import java.lang.constant.MethodTypeDesc;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.CoreOp.FuncOp;
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.type.FieldRef;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;
import jdk.incubator.code.type.RecordTypeRef;

import static jdk.incubator.code.internal.CodeModelAttribute.OpTag.*;

final class OpWriter {

    private final BufWriter buf;
    private final ConstantPoolBuilder cp;
    private final Map<Value, Integer> valueMap;

    OpWriter(BufWriter buf) {
        this.buf = buf;
        this.cp = buf.constantPool();
        this.valueMap = new HashMap<>();
        valueMap.put(null, 0); // 0-index null value
    }

    void writeOp(Op op) {
        Location location = op.location();
        if (location != Location.NO_LOCATION) {
            writeTag(LocationAttr);
            writeUtf8EntryOrZero(location.sourceRef());
            buf.writeU2(location.line());
            buf.writeU2(location.column());
        }
        var operands = op.operands();
        switch (op) {
            case CoreOp.AddOp _ ->
                writeOpWithFixedOperandValues(AddOp, operands);
            case CoreOp.AndOp _ ->
                writeOpWithFixedOperandValues(AndOp, operands);
            case CoreOp.ArrayAccessOp.ArrayLoadOp _ -> {
                writeOpWithFixedOperandValues(ArrayLoadOp, operands);
                writeType(op.resultType());
            }
            case CoreOp.ArrayAccessOp.ArrayStoreOp _ ->
                writeOpWithFixedOperandValues(ArrayStoreOp, operands);
            case CoreOp.ArrayLengthOp _ ->
                writeOpWithFixedOperandValues(ArrayLengthOp, operands);
            case CoreOp.AshrOp _ ->
                writeOpWithFixedOperandValues(AshrOp, operands);
            case CoreOp.BranchOp bo -> {
                 writeTag(BranchOp);
                 writeTarget(bo.branch());
            }
            case CoreOp.CastOp co -> {
                writeTag(CastOp);
                writeType(co.resultType());
                writeType(co.type());
                writeValue(operands.getFirst());
            }
            case CoreOp.ClosureCallOp _ -> {
                writeTag(ClosureCallOp);
                writeValuesList(operands);
            }
            case CoreOp.ClosureOp co -> {
                writeTag(ClosureOp);
                writeNestedBody(co.body());
            }
            case CoreOp.ComplOp _ ->
                writeOpWithFixedOperandValues(ComplOp, operands);
            case CoreOp.ConcatOp _ ->
                writeOpWithFixedOperandValues(ConcatOp, operands);
            case CoreOp.ConditionalBranchOp cbo -> {
                writeOpWithFixedOperandValues(ConditionalBranchOp, operands);
                writeTarget(cbo.trueBranch());
                writeTarget(cbo.falseBranch());
            }
            case CoreOp.ConstantOp co -> {
                writeTag(ConstantOp);
                writeType(co.resultType());
                writeUtf8EntryOrZero(co.value()); // @@@ constant value serialized as Utf8Entry
            }
            case CoreOp.ConvOp _ -> {
                writeTag(ConvOp);
                writeType(op.resultType());
                writeValue(operands.getFirst());
            }
            case CoreOp.DivOp _ ->
                writeOpWithFixedOperandValues(DivOp, operands);
            case CoreOp.EqOp _ ->
                writeOpWithFixedOperandValues(EqOp, operands);
            case CoreOp.ExceptionRegionEnter ere -> {
                writeTag(ExceptionRegionEnter);
                writeTarget(ere.start());
                writeCatchers(ere.catchBlocks());
            }
            case CoreOp.ExceptionRegionExit ere -> {
                writeTag(ExceptionRegionExit);
                writeTarget(ere.end());
                writeCatchers(ere.catchBlocks());
            }
            case CoreOp.FieldAccessOp.FieldLoadOp flo -> {
                writeTag(FieldLoadOp);
                writeValue(flo.receiver());
                writeType(flo.resultType());
                FieldRef fd = flo.fieldDescriptor();
                writeType(fd.refType());
                writeUtf8EntryOrZero(fd.name());
                writeType(fd.type());
            }
            case CoreOp.FieldAccessOp.FieldStoreOp fso -> {
                writeTag(FieldStoreOp);
                writeValue(fso.receiver());
                FieldRef fd = fso.fieldDescriptor();
                writeType(fd.refType());
                writeUtf8EntryOrZero(fd.name());
                writeType(fd.type());
                writeValue(fso.value());
            }
            case CoreOp.FuncCallOp fco -> {
                writeTag(FuncCallOp);
                writeUtf8EntryOrZero(fco.funcName());
                writeType(fco.resultType());
                writeValuesList(operands);
            }
            case CoreOp.FuncOp fo -> {
                writeTag(FuncOp);
                writeUtf8EntryOrZero(fo.funcName());
                writeNestedBody(fo.body());
            }
            case CoreOp.GeOp _ ->
                writeOpWithFixedOperandValues(GeOp, operands);
            case CoreOp.GtOp _ ->
                writeOpWithFixedOperandValues(GtOp, operands);
            case CoreOp.InstanceOfOp ioo -> {
                writeTag(InstanceOfOp);
                writeType(ioo.type());
                writeValue(operands.getFirst());
            }
            case CoreOp.InvokeOp io -> {
                writeTag(InvokeOp);
                buf.writeU1(io.invokeKind().ordinal());
                buf.writeU1(io.isVarArgs() ? 1 : 0);
                writeType(io.resultType());
                MethodRef mr = io.invokeDescriptor();
                writeType(mr.refType());
                writeUtf8EntryOrZero(mr.name());
                writeFunctionType(mr.type());
                writeValuesList(operands);
            }
            case CoreOp.LambdaOp lo -> {
                writeTag(LambdaOp);
                writeType(lo.functionalInterface());
                writeNestedBody(lo.body());
            }
            case CoreOp.LeOp _ ->
                writeOpWithFixedOperandValues(LeOp, operands);
            case CoreOp.LshlOp _ ->
                writeOpWithFixedOperandValues(LshlOp, operands);
            case CoreOp.LshrOp _ ->
                writeOpWithFixedOperandValues(LshrOp, operands);
            case CoreOp.LtOp _ ->
                writeOpWithFixedOperandValues(LtOp, operands);
            case CoreOp.ModOp _ ->
                writeOpWithFixedOperandValues(ModOp, operands);
            case CoreOp.ModuleOp mo -> {
                writeTag(ModuleOp);
                buf.writeU2(mo.functionTable().size());
                for (FuncOp fo : mo.functionTable().values()) {
                     writeUtf8EntryOrZero(fo.funcName());
                     writeNestedBody(fo.body());
                }
            }
            case CoreOp.MonitorOp.MonitorEnterOp _ ->
                writeOpWithFixedOperandValues(MonitorEnterOp, operands);
            case CoreOp.MonitorOp.MonitorExitOp _ ->
                writeOpWithFixedOperandValues(MonitorExitOp, operands);
            case CoreOp.MulOp _ ->
                writeOpWithFixedOperandValues(MulOp, operands);
            case CoreOp.NegOp _ ->
                writeOpWithFixedOperandValues(NegOp, operands);
            case CoreOp.NeqOp _ ->
                writeOpWithFixedOperandValues(NeqOp, operands);
            case CoreOp.NewOp no -> {
                writeTag(NewOp);
                writeType(no.resultType());
                writeFunctionType(no.constructorType());
                writeValuesList(no.operands());
            }
            case CoreOp.NotOp _ ->
                writeOpWithFixedOperandValues(NotOp, operands);
            case CoreOp.OrOp _ ->
                writeOpWithFixedOperandValues(OrOp, operands);
            case CoreOp.QuotedOp qo -> {
                writeTag(QuotedOp);
                writeNestedBody(qo.bodies().getFirst());
            }
            case CoreOp.ReturnOp _ -> {
                writeTag(ReturnOp);
                if (operands.isEmpty()) {
                    writeValue(null);
                } else {
                    writeValue(operands.getFirst());
                }
            }
            case CoreOp.SubOp _ ->
                writeOpWithFixedOperandValues(SubOp, operands);
            case CoreOp.ThrowOp _ ->
                writeOpWithFixedOperandValues(ThrowOp, operands);
            case CoreOp.TupleLoadOp tlo -> {
                writeOpWithFixedOperandValues(TupleLoadOp, operands);
                buf.writeU2(tlo.index());
            }
            case CoreOp.TupleOp _ ->
                writeOpWithFixedOperandValues(TupleOp, operands);
            case CoreOp.TupleWithOp two -> {
                writeOpWithFixedOperandValues(TupleWithOp, operands);
                buf.writeU2(two.index());
            }
            case CoreOp.UnreachableOp _ ->
                writeTag(UnreachableOp);
            case CoreOp.VarAccessOp.VarLoadOp _ ->
                writeOpWithFixedOperandValues(VarLoadOp, operands);
            case CoreOp.VarAccessOp.VarStoreOp _ ->
                writeOpWithFixedOperandValues(VarStoreOp, operands);
            case CoreOp.VarOp vo -> {
                writeTag(VarOp);
                if (vo.isUninitialized()) {
                    writeValue(null);
                } else {
                    writeValue(vo.initOperand());
                }
                writeUtf8EntryOrZero(vo.varName());
                writeType(vo.varValueType());
            }
            case CoreOp.XorOp _ ->
                writeOpWithFixedOperandValues(XorOp, operands);
            case CoreOp.YieldOp yo -> {
                writeTag(YieldOp);
                writeValue(yo.yieldValue());
            }
            case ExtendedOp.JavaBlockOp jbo -> {
                writeTag(JavaBlockOp);
                writeNestedBody(jbo.body());
            }
            case ExtendedOp.JavaBreakOp _ -> {
                writeTag(JavaBreakOp);
                writeValue(operands.isEmpty() ? null : operands.getFirst());
            }
            case ExtendedOp.JavaConditionalAndOp _ -> {
                writeTag(JavaConditionalAndOp);
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaConditionalExpressionOp _ -> {
                writeTag(JavaConditionalExpressionOp);
                writeType(op.resultType());
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaConditionalOrOp _ -> {
                writeTag(JavaConditionalOrOp);
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaContinueOp _ -> {
                writeTag(JavaContinueOp);
                writeValue(operands.isEmpty() ? null : operands.getFirst());
            }
            case ExtendedOp.JavaDoWhileOp _ ->
                writeOpWithFixedNestedBodies(JavaDoWhileOp, op);
            case ExtendedOp.JavaEnhancedForOp _ ->
                writeOpWithFixedNestedBodies(JavaEnhancedForOp, op);
            case ExtendedOp.JavaForOp _ ->
                writeOpWithFixedNestedBodies(JavaForOp, op);
            case ExtendedOp.JavaIfOp _ -> {
                writeTag(JavaIfOp);
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaLabeledOp _ ->
                writeOpWithFixedNestedBodies(JavaLabeledOp, op);
            case ExtendedOp.JavaSwitchExpressionOp _ -> {
                writeTag(JavaSwitchExpressionOp);
                writeType(op.resultType());
                writeValue(operands.getFirst());
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaSwitchFallthroughOp _ ->
                writeTag(JavaSwitchFallthroughOp);
            case ExtendedOp.JavaSwitchStatementOp _ -> {
                writeOpWithFixedOperandValues(JavaSwitchStatementOp, operands);
                writeNestedBodies(op.bodies());
            }
            case ExtendedOp.JavaSynchronizedOp _ ->
                writeOpWithFixedNestedBodies(JavaSynchronizedOp, op);
            case ExtendedOp.JavaTryOp jto -> {
                writeTag(JavaTryOp);
                writeNestedBody(jto.resources());
                writeNestedBody(jto.body());
                writeNestedBodies(jto.catchers());
                writeNestedBody(jto.finalizer());
            }
            case ExtendedOp.JavaYieldOp _ -> {
                writeTag(JavaYieldOp);
                writeValue(operands.isEmpty() ? null : operands.getFirst());
            }
            case ExtendedOp.JavaWhileOp _ ->
                writeOpWithFixedNestedBodies(JavaWhileOp, op);
            case ExtendedOp.PatternOps.MatchAllPatternOp _ ->
                writeTag(MatchAllPatternOp);
            case ExtendedOp.PatternOps.MatchOp mo -> {
                writeOpWithFixedOperandValues(MatchOp, operands);
                writeNestedBody(mo.pattern());
                writeNestedBody(mo.match());
            }
            case ExtendedOp.PatternOps.RecordPatternOp rpo -> {
                writeTag(RecordPatternOp);
                RecordTypeRef rd = rpo.recordDescriptor();
                writeType(rd.recordType());
                buf.writeU2(rd.components().size());
                for (RecordTypeRef.ComponentRef rc : rd.components()) {
                    writeType(rc.type());
                    writeUtf8EntryOrZero(rc.name());
                }
                 writeUtf8EntryOrZero(rpo.recordDescriptor().toString());
            }
            case ExtendedOp.PatternOps.TypePatternOp tpo -> {
                writeTag(TypePatternOp);
                writeType(tpo.targetType());
                writeUtf8EntryOrZero(tpo.bindingName());
            }
            default ->
                throw new IllegalArgumentException(op.toText());
        }
        if (op.result() != null) {
            valueMap.put(op.result(), valueMap.size());
        }
   }

    private void writeAttributes(Map<String, Object> attributes) {
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
                writeUtf8EntryOrZero(loc.sourceRef());
                buf.writeU2(loc.line());
                buf.writeU2(loc.column());
            } else {
                writeUtf8EntryOrZero(attre.getValue());
            }
        }
    }

    private void writeUtf8EntryOrZero(Object o) {
        buf.writeIndexOrZero(o == null ? null : cp.utf8Entry(o.toString()));
    }

    private void writeValuesList(List<Value> values) {
        // number of values
        buf.writeU2(values.size());
        for (Value v : values) {
            writeValue(v);
        }
    }

    private void writeNestedBodies(List<Body> bodies) {
        // number of bodies
        buf.writeU2(bodies.size());
        for (Body body : bodies) {
            writeNestedBody(body);
        }
    }

    private void writeNestedBody(Body body) {
        // body type
        buf.writeIndexOrZero(body == null ? null : toEntry(body.bodyType()));
        // blocks
        if (body != null) writeBlocks(body.blocks());
    }

    private void writeBlocks(List<Block> blocks) {
        // number of blocks - entry block
        buf.writeU2(blocks.size() - 1);
        for (Block block : blocks) {
            // parameters
            if (block.isEntryBlock()) { // @@@ assumption entry block is the first one
                for (var bp : block.parameters()) {
                    valueMap.put(bp, valueMap.size());
                }
            } else {
                writeBlockParameters(block.parameters());
            }
            // ops
            writeOps(block.ops());
        }
    }

    private void writeBlockParameters(List<Block.Parameter> parameters) {
        // number of block parameters
        buf.writeU2(parameters.size());
        for (Block.Parameter bp : parameters) {
            // block parameter type
            buf.writeIndexOrZero(toEntry(bp.type()));
            valueMap.put(bp, valueMap.size());
        }
    }

    private void writeOps(List<Op> ops) {
        // number of ops - mandatory terminal op
        buf.writeU2(ops.size() - 1);
        for (Op op : ops) {
            // op
            writeOp(op);
        }
    }

    private void writeSuccessors(List<Block.Reference> successors) {
        // number of successors
        buf.writeU2(successors.size());
        for (Block.Reference succ : successors) {
            // block index
            buf.writeU2(succ.targetBlock().index());
            // arguments
            writeValuesList(succ.arguments());
        }
    }

    private PoolEntry toEntry(FunctionType ftype) {
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

    private PoolEntry toEntry(TypeElement type) {
        if (type.equals(JavaType.VOID)) return null;
        return type instanceof JavaType jt && jt.erasure().equals(jt)
                ? cp.utf8Entry(jt.toNominalDescriptor())
                : cp.stringEntry(type.externalize().toString());
    }

    private void writeOpWithFixedOperandValues(CodeModelAttribute.OpTag tag, List<Value> operands) {
        writeTag(tag);
        for (Value v : operands) {
            buf.writeU2(valueMap.get(v));
        }
    }

    private void writeOpWithFixedNestedBodies(CodeModelAttribute.OpTag tag, Op op) {
        writeTag(tag);
        for (var body : op.bodies()) {
            writeNestedBody(body);
        }
    }

    private void writeTag(CodeModelAttribute.OpTag tag) {
        buf.writeU1(tag.ordinal());
    }

    private void writeValue(Value v) {
        buf.writeU2(valueMap.get(v));
    }

    private void writeType(TypeElement type) {
        buf.writeIndexOrZero(toEntry(type));
    }

    private void writeFunctionType(FunctionType ftype) {
        buf.writeIndex(toEntry(ftype));
    }

    private void writeTypesList(List<TypeElement> types) {
        buf.writeU2(types.size());
        for (TypeElement type : types) {
            writeType(type);
        }
    }

    private void writeTarget(Block.Reference target) {
        buf.writeU2(target.targetBlock().index());
        writeValuesList(target.arguments());
    }

    private void writeCatchers(List<Block.Reference> catchers) {
        buf.writeU2(catchers.size());
        for (var c : catchers) {
            buf.writeU2(c.targetBlock().index());
        }
    }
}
