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

package java.lang.reflect.code.analysis;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.type.PrimitiveType;
import java.util.*;

/**
 * StringConcatTransformer is an {@link java.lang.reflect.code.OpTransformer} that removes concatenation operations
 * from blocks and replaces them with equivalent {@link java.lang.StringBuilder} operations. This provides a pathway
 * to remove {@link java.lang.reflect.code.op.CoreOp.ConcatOp} for easier lowering to Bytecode.
 */
public class StringConcatTransformer implements OpTransformer {

    private static final JavaType J_L_BYTESTRING = JavaType.type(StringBuilder.class);
    private static final JavaType J_L_CHARSEQUENCE = JavaType.type(CharSequence.class);
    private static final JavaType J_L_OBJECT = JavaType.type(Object.class);
    private static final MethodRef SB_TO_STRING = MethodRef.method(J_L_BYTESTRING, "toString", JavaType.J_L_STRING);

    final private Set<Value> StringBuilders = new HashSet<>();
    final private Map<Integer, Value> IntConstants = new HashMap<>();

    public StringConcatTransformer() {
    }

    private static boolean reusableResult(Op.Result r) {
        if (r.uses().size() == 1) {
            return r.uses().stream().allMatch((use) -> use.op() instanceof CoreOp.ConcatOp);
        }
        return false;
    }

    @Override public Block.Builder apply(Block.Builder builder, Op op) {
        if (op instanceof CoreOp.ConcatOp cz) {

            Value left = builder.context().getValue(cz.operands().get(0));
            Value right = builder.context().getValue(cz.operands().get(1));
            Op.Result result = cz.result();
            if (reusableResult(result)) {
                Value sb = stringBuilder(builder, left, right);
                builder.context().mapValue(result, sb);
            } else {
                Value sb = stringBuilder(builder, left, right);
                Value str = buildString(builder, sb);
                builder.context().mapValue(result, str);
            }
        } else {
            builder.op(op);
        }
        return builder;
    }

    //Uses StringBuilder and Immediate String Value
    private Value buildString(Block.Builder builder, Value sb){
        var toStringInvoke = CoreOp.invoke(SB_TO_STRING, sb);
        return builder.apply(toStringInvoke);
    }

    private Value stringBuilder(Block.Builder builder, Value left, Value right) {
        if (left.type().equals(J_L_BYTESTRING) && StringBuilders.contains(left)
                || right.type().equals(J_L_BYTESTRING) && StringBuilders.contains(right)) {
            var res = builder.op(append(builder, left, right));
            StringBuilders.add(res);
            return res;
        } else {
            CoreOp.NewOp newBuilder = CoreOp._new(FunctionType.functionType(J_L_BYTESTRING));
            Value sb = builder.op(newBuilder);
            StringBuilders.add(sb);
            var res = builder.op(append(builder, sb, left));
            StringBuilders.add(res);
            var res2 = builder.op(append(builder, res, right));
            StringBuilders.add(res2);
            return res2;
        }
    }

    private Op append(Block.Builder builder, Value left, Value right) {
        //Left argument is a blessed stringbuilder
        if (left.type().equals(J_L_BYTESTRING) && StringBuilders.contains(left)) {
            var rightType = right.type();
            if (rightType.equals(J_L_BYTESTRING)) {
                rightType = J_L_CHARSEQUENCE;
                MethodRef leftMethodDesc = MethodRef.method(J_L_BYTESTRING, "append", J_L_BYTESTRING, rightType);
                return CoreOp.invoke(leftMethodDesc, left, right);
            }

            if (rightType instanceof PrimitiveType) {
                if (List.of(JavaType.BYTE, JavaType.SHORT).contains(rightType)) {
                    Value widened = builder.op(CoreOp.conv(JavaType.INT, right));
                    MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "append", J_L_BYTESTRING, JavaType.INT);
                    return CoreOp.invoke(methodDesc, left, widened);
                } else {
                    MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "append", J_L_BYTESTRING, rightType);
                    return CoreOp.invoke(methodDesc, left, right);
                }
            } else if (rightType.equals(JavaType.J_L_STRING)) {
                MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "append", J_L_BYTESTRING, rightType);
                return CoreOp.invoke(methodDesc, left, right);
            } else {
                MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "append", J_L_BYTESTRING, J_L_OBJECT);
                return CoreOp.invoke(methodDesc, left, right);
            }
        } else if (right.type().equals(J_L_BYTESTRING) && StringBuilders.contains(right)) {
            var leftType = left.type();
            var zero = getConstant(builder, 0);
            if (leftType.equals(J_L_BYTESTRING)) {
                leftType = J_L_CHARSEQUENCE;
                MethodRef leftMethodDesc = MethodRef.method(J_L_BYTESTRING, "insert", J_L_BYTESTRING, JavaType.INT, leftType);
                return CoreOp.invoke(leftMethodDesc, right, zero, left);
            }
            if (leftType instanceof PrimitiveType) {
                if (List.of(JavaType.BYTE, JavaType.SHORT).contains(leftType)) {
                    Value widened = builder.op(CoreOp.conv(JavaType.INT, left));
                    MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "insert", J_L_BYTESTRING, JavaType.INT, JavaType.INT);
                    return CoreOp.invoke(methodDesc, right, zero, widened);
                } else {
                    MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "insert", J_L_BYTESTRING, JavaType.INT, leftType);
                    return CoreOp.invoke(methodDesc, right, zero, left);
                }
            } else if (leftType.equals(JavaType.J_L_STRING)) {
                MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "insert", J_L_BYTESTRING, JavaType.INT, leftType);
                return CoreOp.invoke(methodDesc, right, zero, left);
            } else {
                MethodRef methodDesc = MethodRef.method(J_L_BYTESTRING, "insert", J_L_BYTESTRING, JavaType.INT, J_L_OBJECT);
                return CoreOp.invoke(methodDesc, right, zero, left);
            }
        }
        throw new RuntimeException("append requires a blessed StringBuilder as one Value argument");
    }

    private Value getConstant(Block.Builder builder, int con) {
        var val = IntConstants.get(con);
        if (val == null) {
            var constOp = CoreOp.constant(JavaType.INT, con);
            val = builder.op(constOp);
            IntConstants.put(0, val);
        }
        return val;
    }

}