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
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;

import static java.lang.Thread.yield;

/**
 * Fixes null constant types and injects mandatory explicit casts and conversions.
 */
final class PostLiftTypesTransformer {

    static CoreOp.FuncOp transform(CoreOp.FuncOp func) {
        return func.transform((block, op) -> {
            if (op instanceof CoreOp.ConstantOp cop && JavaType.J_L_OBJECT.equals(op.resultType()) && cop.value() == null) {
                block.context().mapValue(op.result(), block.op(CoreOp.constant(inferNullTypeFromDirectUse(op.result()), null)));
            } else {
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
}
