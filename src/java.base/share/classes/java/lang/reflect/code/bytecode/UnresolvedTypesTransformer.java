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
package java.lang.reflect.code.bytecode;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.ArrayList;
import java.util.List;

/**
 * Resolves unresolved types.
 */
final class UnresolvedTypesTransformer {

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {

        var unresolved = func.traverse(new ArrayList<Value>(), (q, e) -> {
            switch (e) {
                case Block b -> b.parameters().forEach(v -> {
                   if (toResolve(v)) q.add(v);
                });
                case Op op when toResolve(op.result()) ->
                   q.add(op.result());
                default -> {}
           }
           return q;
        });

        boolean changed = true;
        while (changed) {
            changed = false;
            for (Value v : unresolved) {
                changed |= resolve(v);
            }
        }

        // Remaining types are resolved to defaults
        for (Value v : unresolved) {
            switch (v.type()) {
                case UnresolvedType.Int ui when ui.resolved == null ->
                    ui.resolved = JavaType.INT;
                case UnresolvedType.Ref ur when ur.resolved == null ->
                    ur.resolved = JavaType.J_L_OBJECT;
                default -> {}
            }
        }

        return func.transform(blockParamTypesTransformer())
                   .transform(opTypesTransformer())
                   .transform(unifyOperandsTransformer());
    }

    private static boolean toResolve(Value v) {
        return v != null && v.type() instanceof UnresolvedType;
    }

    private static boolean resolve(Value v) {
        if (v == null || !(v.type() instanceof UnresolvedType ut)) {
            return false;
        }
        boolean changed = false;
        for (Op.Result useRes : v.uses()) {
            Op op = useRes.op();
            int i = op.operands().indexOf(v);
            if (i >= 0) {
                changed |= (boolean)switch (op) {
                    case CoreOp.BinaryTestOp bto ->
                        ut.resolveTo(bto.operands().get(1 - i).type());
                    case CoreOp.LshlOp _, CoreOp.LshrOp _, CoreOp.AshrOp _ -> // Second operands are asymetric
                        i == 0 && ut.resolveTo(op.resultType());
                    case CoreOp.BinaryOp bo ->
                        ut.resolveTo(bo.resultType())
                        || ut.resolveTo(bo.operands().get(1 - i).type());
                    case CoreOp.InvokeOp io -> {
                        MethodRef id = io.invokeDescriptor();
                        if (io.hasReceiver()) {
                            if (i == 0) yield id.refType();
                            i--;
                        }
                        yield ut.resolveTo(id.type().parameterTypes().get(i));
                    }
                    case CoreOp.FieldAccessOp fao ->
                        ut.resolveTo(fao.fieldDescriptor().refType());
                    case CoreOp.ReturnOp ro ->
                        ut.resolveTo(ro.ancestorBody().bodyType().returnType());
                    case CoreOp.VarOp vo ->
                        ut.resolveTo(vo.varValueType());
                    case CoreOp.VarAccessOp.VarStoreOp vso ->
                        ut.resolveTo(vso.varType().valueType());
                    case CoreOp.NewOp no ->
                        ut.resolveTo(no.constructorType().parameterTypes().get(i));
                    default -> false;
                };
            }
            // Pull block parameter type when used as block argument
            for (Block.Reference sucRef : useRes.op().successors()) {
                i = sucRef.arguments().indexOf(v);
                if (i >= 0) {
                    changed |= ut.resolveTo(sucRef.targetBlock().parameters().get(i).type());
                }
            }
        }
        return changed;
    }

    private UnresolvedTypesTransformer() {
    }

    private static OpTransformer blockParamTypesTransformer() {
        return new OpTransformer() {
            @Override
            public void apply(Block.Builder block, Block b) {
                if (block.isEntryBlock()) {
                    CopyContext cc = block.context();
                    List<Block> sourceBlocks = b.parentBody().blocks();

                    // Override blocks with changed parameter types
                    for (int i = 1; i < sourceBlocks.size(); i++) {
                        Block sourceBlock = sourceBlocks.get(i);
                        List<TypeElement> paramTypes = sourceBlock.parameterTypes();
                        if (paramTypes.stream().anyMatch(UnresolvedType.class::isInstance)) {
                            Block.Builder newBlock = block.block(paramTypes.stream()
                                    .map(pt -> pt instanceof UnresolvedType ut && ut.resolved != null ? ut.resolved : pt)
                                    .toList());
                            cc.mapBlock(sourceBlock, newBlock);
                            cc.mapValues(sourceBlock.parameters(), newBlock.parameters());
                        }
                    }

                }
                OpTransformer.super.apply(block, b);
            }

            @Override
            public Block.Builder apply(Block.Builder block, Op op) {
                block.op(op);
                return block;
            }
        };
    }

    private static OpTransformer opTypesTransformer() {
        return (block, op) -> {
            CopyContext cc = block.context();
            switch (op) {
                case CoreOp.ConstantOp cop when op.resultType() instanceof UnresolvedType ut && ut.resolved != null ->
                    cc.mapValue(op.result(), block.op(CoreOp.constant(ut.resolved, ut.convertValue(cop.value()))));
                case CoreOp.VarOp vop when vop.varValueType() instanceof UnresolvedType ut && ut.resolved != null ->
                    cc.mapValue(op.result(), block.op(CoreOp.var(vop.varName(), ut.resolved, cc.getValueOrDefault(vop.initOperand(), vop.initOperand()))));
                default ->
                    block.op(op);
            }
            return block;
        };
    }

    private static OpTransformer unifyOperandsTransformer() {
        return (block, op) -> {
            switch (op) {
                case CoreOp.BinaryTestOp _ ->
                    unify(block, op, JavaType.INT, JavaType.INT);
                case CoreOp.LshlOp _, CoreOp.LshrOp _, CoreOp.AshrOp _ ->
                    unify(block, op, op.resultType(), JavaType.INT);
                case CoreOp.BinaryOp _ ->
                    unify(block, op, op.resultType(), op.resultType());
                default ->
                    block.op(op);
            }
            return block;
        };
    }

    private static void unify(Block.Builder block, Op op, TypeElement firstType, TypeElement secondType) {
        List<Value> operands = op.operands();
        CopyContext cc = CopyContext.create(block.context());
        Value first = operands.getFirst();
        boolean changed = false;
        if (first.type() instanceof PrimitiveType && !first.type().equals(firstType)) {
            cc.mapValue(first, block.op(CoreOp.conv(firstType, cc.getValueOrDefault(first, first))));
            changed = true;
        }
        Value second = operands.get(1);
        if (second.type() instanceof PrimitiveType && !second.type().equals(secondType)) {
            cc.mapValue(second, block.op(CoreOp.conv(secondType, cc.getValueOrDefault(second, second))));
            changed = true;
        }
        if (changed) {
            block.context().mapValue(op.result(), block.op(op.copy(cc)));
        } else {
            block.op(op);
        }
    }
}
