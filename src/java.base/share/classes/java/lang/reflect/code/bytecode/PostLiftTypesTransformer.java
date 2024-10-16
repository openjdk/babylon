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
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;

import java.util.List;

/**
 * Fixes null constant types and injects mandatory explicit conversions.
 */
final class PostLiftTypesTransformer {

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {
        return func.transform((block, op) -> {
            CopyContext cc = block.context();
            switch (op) {
                case CoreOp.ConstantOp cop when JavaType.J_L_OBJECT.equals(op.resultType()) && cop.value() == null ->
                    cc.mapValue(op.result(), block.op(CoreOp.constant(inferNullTypeFromDirectUse(op.result()), null)));
                case CoreOp.BranchOp bo -> {
                    Value[] args = cc.getValues(bo.branch().arguments()).toArray(Value[]::new);
                    Block target = bo.branch().targetBlock();
                    if (convertArgs(block, args, target.parameterTypes())) {
                        block.op(CoreOp.branch(cc.getBlock(target).successor(args)));
                    } else {
                        block.op(op);
                    }
                }
                default ->
                    block.op(op);
            }
            return block;
        });
    }

    private static TypeElement inferNullTypeFromDirectUse(Value v) {
        for (Op.Result useRes : v.uses()) {
            // Pull type from operands
            int i = useRes.op().operands().indexOf(v);
            if (i >= 0) {
                TypeElement type = switch (useRes.op()) {
                    case CoreOp.InvokeOp io -> {
                        MethodRef id = io.invokeDescriptor();
                        if (io.hasReceiver()) {
                            if (i == 0) yield id.refType();
                            i--;
                        }
                        yield id.type().parameterTypes().get(i);
                    }
                    case CoreOp.FieldAccessOp fao ->
                        fao.fieldDescriptor().refType();
                    case CoreOp.ReturnOp ro ->
                        ro.resultType();
                    case CoreOp.VarOp vo ->
                        vo.varValueType();
                    case CoreOp.VarAccessOp.VarStoreOp vso ->
                        vso.varType().valueType();
                    default -> null;
                };
                if (type != null && !JavaType.J_L_OBJECT.equals(type)) {
                    return type;
                }
            } else {
                // Pull block parameter type when used as block argument
                for (Block.Reference sucRef : useRes.op().successors()) {
                    i = sucRef.arguments().indexOf(v);
                    if (i >= 0) {
                        TypeElement type = sucRef.targetBlock().parameters().get(i).type();
                        if (!JavaType.J_L_OBJECT.equals(type)) {
                            return type;
                        }
                    }
                }
            }
        }
        return JavaType.J_L_OBJECT;
    }


    private static boolean convertArgs(Block.Builder block, Value[] args, List<TypeElement> paramTypes) {
        assert args.length == paramTypes.size();
        boolean convert = false;
        for (int i = 0; i < args.length; i++) {
            TypeElement pt = paramTypes.get(i);
            Value arg = args[i];
            if (requiresExplicitConversion(arg.type(), pt)) {
                args[i] = block.op(CoreOp.conv(pt, arg));
                convert = true;
            }
        }
        return convert;
    }

    private static boolean requiresExplicitConversion(TypeElement from, TypeElement to) {
        return from.equals(JavaType.BOOLEAN) && to.equals(JavaType.INT)
            || from.equals(JavaType.INT) && to.equals(JavaType.BOOLEAN);
    }
}
