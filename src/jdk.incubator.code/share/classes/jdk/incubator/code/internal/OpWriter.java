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
import jdk.incubator.code.op.ExtendedOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.VarType;

import static jdk.incubator.code.internal.CodeModelAttribute.OpTag.*;

final class OpWriter {

    private final BufWriter buf;
    private final ConstantPoolBuilder cp;
    private final Map<Value, Integer> valueMap;

    OpWriter(BufWriter buf) {
        this.buf = buf;
        this.cp = buf.constantPool();
        this.valueMap = new HashMap<>();
    }

    void writeOp(Op op) {
       var operands = op.operands();
       switch (op) {
           case CoreOp.AddOp _ ->
               writeOpWithFixedOperandValues(AddOp, op);
           case CoreOp.AndOp _ ->
               writeOpWithFixedOperandValues(AndOp, op);
           case CoreOp.ArrayAccessOp.ArrayLoadOp _ -> {
               writeOpWithFixedOperandValues(ArrayLoadOp, op);
               writeType(op.resultType());
           }
           case CoreOp.ArrayAccessOp.ArrayStoreOp _ ->
               writeOpWithFixedOperandValues(ArrayStoreOp, op);
           case CoreOp.ArrayLengthOp _ ->
               writeOpWithFixedOperandValues(ArrayLengthOp, op);
           case CoreOp.AshrOp _ ->
               writeOpWithFixedOperandValues(AshrOp, op);
           case CoreOp.BranchOp bo -> {
                writeTag(BranchOp);
                writeTarget(bo.branch());
           }
           case CoreOp.CastOp co -> {
               writeTag(CastOp);
               writeType(co.resultType());
               writeType(co.type());
               writeValue(co.operands().getFirst());
           }
           case CoreOp.ClosureCallOp cco -> {
               writeTag(ClosureCallOp);
               writeValuesList(cco.operands());
           }
           case CoreOp.ClosureOp co -> {
               writeTag(ClosureOp);
               writeNestedBody(co.body());
           }
           case CoreOp.ComplOp _ ->
               writeOpWithFixedOperandValues(ComplOp, op);
           case CoreOp.ConcatOp _ ->
               writeOpWithFixedOperandValues(ConcatOp, op);
           case CoreOp.ConditionalBranchOp cbo -> {
               writeTag(ConditionalBranchOp);
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
               writeValue(op.operands().getFirst());
           }
           case CoreOp.DivOp _ ->
               writeOpWithFixedOperandValues(DivOp, op);
           case CoreOp.EqOp _ ->
               writeOpWithFixedOperandValues(EqOp, op);
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
           case CoreOp.FieldAccessOp.FieldLoadOp _ -> {
               writeTag(FieldLoadOp);
           }
           case CoreOp.FieldAccessOp.FieldStoreOp _ -> {
               writeTag(FieldStoreOp);
           }
           case CoreOp.FuncCallOp _ -> {
               writeTag(FuncCallOp);
           }
           case CoreOp.FuncOp _ -> {
               writeTag(FuncOp);
           }
           case CoreOp.GeOp _ -> {
               writeTag(GeOp);
           }
           case CoreOp.GtOp _ -> {
               writeTag(GtOp);
           }
           case CoreOp.InstanceOfOp _ -> {
               writeTag(InstanceOfOp);
           }
           case CoreOp.InvokeOp _ -> {
               writeTag(InvokeOp);
           }
           case CoreOp.LambdaOp _ -> {
               writeTag(LambdaOp);
           }
           case CoreOp.LeOp _ -> {
               writeTag(LeOp);
           }
           case CoreOp.LshlOp _ -> {
               writeTag(LshlOp);
           }
           case CoreOp.LshrOp _ -> {
               writeTag(LshrOp);
           }
           case CoreOp.LtOp _ -> {
               writeTag(LtOp);
           }
           case CoreOp.ModOp _ -> {
               writeTag(ModOp);
           }
           case CoreOp.ModuleOp _ -> {
               writeTag(ModuleOp);
           }
           case CoreOp.MonitorOp.MonitorEnterOp _ -> {
               writeTag(MonitorEnterOp);
           }
           case CoreOp.MonitorOp.MonitorExitOp _ -> {
               writeTag(MonitorExitOp);
           }
           case CoreOp.MulOp _ -> {
               writeTag(MulOp);
           }
           case CoreOp.NegOp _ -> {
               writeTag(NegOp);
           }
           case CoreOp.NeqOp _ -> {
               writeTag(NeqOp);
           }
           case CoreOp.NewOp _ -> {
               writeTag(NewOp);
           }
           case CoreOp.NotOp _ -> {
               writeTag(NotOp);
           }
           case CoreOp.OrOp _ -> {
               writeTag(OrOp);
           }
           case CoreOp.QuotedOp _ -> {
               writeTag(QuotedOp);
           }
           case CoreOp.ReturnOp _ -> {
               writeTag(ReturnOp);
           }
           case CoreOp.SubOp _ -> {
               writeTag(SubOp);
           }
           case CoreOp.ThrowOp _ -> {
               writeTag(ThrowOp);
           }
           case CoreOp.TupleLoadOp _ -> {
               writeTag(TupleLoadOp);
           }
           case CoreOp.TupleOp _ -> {
               writeTag(TupleOp);
           }
           case CoreOp.TupleWithOp _ -> {
               writeTag(TupleWithOp);
           }
           case CoreOp.UnreachableOp _ -> {
               writeTag(UnreachableOp);
           }
           case CoreOp.VarAccessOp.VarLoadOp _ -> {
               writeTag(VarLoadOp);
           }
           case CoreOp.VarAccessOp.VarStoreOp _ -> {
               writeTag(VarStoreOp);
           }
           case CoreOp.VarOp _ -> {
               writeTag(VarOp);
           }
           case CoreOp.XorOp _ -> {
               writeTag(XorOp);
           }
           case CoreOp.YieldOp _ -> {
               writeTag(YieldOp);
           }
           case ExtendedOp.JavaBlockOp _ -> {
               writeTag(JavaBlockOp);
           }
           case ExtendedOp.JavaBreakOp _ -> {
               writeTag(JavaBreakOp);
           }
           case ExtendedOp.JavaConditionalAndOp _ -> {
               writeTag(JavaConditionalAndOp);
           }
           case ExtendedOp.JavaConditionalExpressionOp _ -> {
               writeTag(JavaConditionalExpressionOp);
           }
           case ExtendedOp.JavaConditionalOrOp _ -> {
               writeTag(JavaConditionalOrOp);
           }
           case ExtendedOp.JavaContinueOp _ -> {
               writeTag(JavaContinueOp);
           }
           case ExtendedOp.JavaDoWhileOp _ -> {
               writeTag(JavaDoWhileOp);
           }
           case ExtendedOp.JavaEnhancedForOp _ -> {
               writeTag(JavaEnhancedForOp);
           }
           case ExtendedOp.JavaForOp _ -> {
               writeTag(JavaForOp);
           }
           case ExtendedOp.JavaIfOp _ -> {
               writeTag(JavaIfOp);
           }
           case ExtendedOp.JavaLabeledOp _ -> {
               writeTag(JavaLabeledOp);
           }
           case ExtendedOp.JavaSwitchExpressionOp _ -> {
               writeTag(JavaSwitchExpressionOp);
           }
           case ExtendedOp.JavaSwitchFallthroughOp _ -> {
               writeTag(JavaSwitchFallthroughOp);
           }
           case ExtendedOp.JavaSwitchStatementOp _ -> {
               writeTag(JavaSwitchStatementOp);
           }
           case ExtendedOp.PatternOps.MatchAllPatternOp _ -> {
               writeTag(MatchAllPatternOp);
           }
           case ExtendedOp.PatternOps.MatchOp _ -> {
               writeTag(MatchOp);
           }
           case ExtendedOp.PatternOps.RecordPatternOp _ -> {
               writeTag(RecordPatternOp);
           }
           case ExtendedOp.PatternOps.TypePatternOp _ -> {
               writeTag(TypePatternOp);
           }
           default -> {}
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
        buf.writeIndex(toEntry(body.bodyType()));
        // blocks
        writeBlocks(body.blocks());
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

    private void writeOpWithFixedOperandValues(CodeModelAttribute.OpTag tag, Op op) {
        writeTag(tag);
        for (Value v : op.operands()) {
            buf.writeU2(valueMap.get(v));
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
