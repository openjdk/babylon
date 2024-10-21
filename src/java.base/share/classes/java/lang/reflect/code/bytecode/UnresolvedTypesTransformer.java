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
import java.lang.reflect.code.op.CoreOp;
import java.util.List;

/**
 * Resolves unresolved types.
 */
final class UnresolvedTypesTransformer {

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {

        // @@@ analyze and resolve unresolved types

        return func.transform(blockParamTypesTransformer()).transform(opTypesTransformer());
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
                                    .map(pt -> pt instanceof UnresolvedType ut ? ut.resolved() : pt)
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
                case CoreOp.ConstantOp cop when op.resultType() instanceof UnresolvedType ut ->
                    cc.mapValue(op.result(), block.op(CoreOp.constant(ut.resolved(), ut.convertValue(cop.value()))));
                case CoreOp.VarOp vop when vop.varValueType() instanceof UnresolvedType ut ->
                    cc.mapValue(op.result(), block.op(CoreOp.var(vop.varName(), ut.resolved(), vop.initOperand())));
                default ->
                    block.op(op);
            }
            return block;
        };
    }
}

//            private static boolean isBoolean(Set<Block.Parameter> booleans, Block.Parameter bp) {
//                for (Value v : bp.dependsOn()) {
//                    if (v.type().equals(JavaType.BOOLEAN) || v instanceof Block.Parameter dbp && booleans.contains(dbp)) {
//                        return true;
//                    }
//                }
//                for (Op.Result useRes : bp.uses()) {
//                    // Pull type from operands
//                    int i = useRes.op().operands().indexOf(bp);
//                    if (i >= 0) {
//                        TypeElement type = switch (useRes.op()) {
//                            case CoreOp.InvokeOp io -> {
//                                MethodRef id = io.invokeDescriptor();
//                                if (io.hasReceiver()) {
//                                    if (i == 0) yield id.refType();
//                                    i--;
//                                }
//                                yield id.type().parameterTypes().get(i);
//                            }
//                            case CoreOp.FieldAccessOp fao ->
//                                fao.fieldDescriptor().refType();
//                            case CoreOp.ReturnOp ro ->
//                                ro.ancestorBody().bodyType().returnType();
//                            case CoreOp.VarOp vo ->
//                                vo.varValueType();
//                            case CoreOp.VarAccessOp.VarStoreOp vso ->
//                                vso.varType().valueType();
//                            default -> null;
//                        };
//                        if (JavaType.BOOLEAN.equals(type)) {
//                            return true;
//                        }
//                    } else {
//                        // Pull block parameter type when used as block argument
//                        for (Block.Reference sucRef : useRes.op().successors()) {
//                            i = sucRef.arguments().indexOf(bp);
//                            if (i >= 0) {
//                                Block.Parameter sbp = sucRef.targetBlock().parameters().get(i);
//                                if (JavaType.BOOLEAN.equals(sbp.type()) || booleans.contains(sbp)) {
//                                    return true;
//                                }
//                            }
//                        }
//                    }
//                }
//                return false;
//            }
//
//    private static TypeElement inferTypeFromDirectUse(TypeElement fallback, Value v) {
//        for (Op.Result useRes : v.uses()) {
//            // Pull type from operands
//            int i = useRes.op().operands().indexOf(v);
//            if (i >= 0) {
//                TypeElement type = switch (useRes.op()) {
//                    case CoreOp.BinaryTestOp bto ->
//                        bto.operands().get(1 - i).type();
//                    case CoreOp.InvokeOp io -> {
//                        MethodRef id = io.invokeDescriptor();
//                        if (io.hasReceiver()) {
//                            if (i == 0) yield id.refType();
//                            i--;
//                        }
//                        yield id.type().parameterTypes().get(i);
//                    }
//                    case CoreOp.FieldAccessOp fao ->
//                        fao.fieldDescriptor().refType();
//                    case CoreOp.ReturnOp ro ->
//                        ro.ancestorBody().bodyType().returnType();
//                    case CoreOp.VarOp vo ->
//                        vo.varValueType();
//                    case CoreOp.VarAccessOp.VarStoreOp vso ->
//                        vso.varType().valueType();
//                    default -> null;
//                };
//                if (type != null && !fallback.equals(type)) {
//                    return type;
//                }
//            } else {
//                // Pull block parameter type when used as block argument
//                for (Block.Reference sucRef : useRes.op().successors()) {
//                    i = sucRef.arguments().indexOf(v);
//                    if (i >= 0) {
//                        TypeElement type = sucRef.targetBlock().parameters().get(i).type();
//                        if (!fallback.equals(type)) {
//                            return type;
//                        }
//                    }
//                }
//            }
//        }
//        return fallback;
//    }
//
//    private static Block.Reference convert(Block.Builder block, Block.Reference ref) {
//        CopyContext cc = block.context();
//        Value[] args = cc.getValues(ref.arguments()).toArray(Value[]::new);
//        Block target = ref.targetBlock();
//        List<TypeElement> paramTypes = target.parameterTypes();
//        assert args.length == paramTypes.size();
//        for (int i = 0; i < args.length; i++) {
//            TypeElement pt = paramTypes.get(i);
//            Value arg = args[i];
//            if (requiresExplicitConversion(arg.type(), pt)) {
//                args[i] = block.op(CoreOp.conv(pt, arg));
//            }
//        }
//        return cc.getBlock(target).successor(args);
//    }
//
//    private static boolean requiresExplicitConversion(TypeElement from, TypeElement to) {
//        return from.equals(JavaType.BOOLEAN) && to.equals(JavaType.INT)
//            || from.equals(JavaType.INT) && to.equals(JavaType.BOOLEAN);
//    }
