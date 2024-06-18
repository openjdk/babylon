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

/**
 * StringConcatTransformer is an {@link java.lang.reflect.code.OpTransformer} that removes concatenation operations
 * from blocks and replaces them with equivalent {@link java.lang.StringBuilder} operations. This provides a pathway
 * to remove {@link java.lang.reflect.code.op.CoreOp.ConcatOp} for easier lowering to Bytecode.
 */
public class StringConcatTransformer implements OpTransformer {

    private static final JavaType SBC_TYPE = JavaType.type(StringBuilder.class);
    private static final JavaType CHAR_SEQ_TYPE = JavaType.type(CharSequence.class);
    private static final MethodRef SB_TO_STRING = MethodRef.method(SBC_TYPE, "toString", JavaType.J_L_STRING);

    public StringConcatTransformer() {}

    private static boolean reusableResult(Op.Result r) {
        if (r.uses().size() == 1) {
            return r.uses().stream().noneMatch((use) -> use.op() instanceof CoreOp.ReturnOp ||
                    use.op() instanceof CoreOp.VarOp);
        }
        return false;
    }

    @Override
    public Block.Builder apply(Block.Builder builder, Op op) {
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
    private static Value buildString(Block.Builder builder, Value sb){
        var toStringInvoke = CoreOp.invoke(SB_TO_STRING, sb);
        return builder.apply(toStringInvoke);
    }

    private static Value stringBuilder(Block.Builder builder, Value left, Value right) {
        if (left.type().equals(SBC_TYPE)) {
            return builder.op(append(left, right));
        } else {
            CoreOp.NewOp newBuilder = CoreOp._new(FunctionType.functionType(SBC_TYPE));
            Value sb = builder.apply(newBuilder);
            var res = builder.op(append(sb, left));
            return builder.op(append(res, right));
        }
    }

    private static Op append(Value stringBuilder, Value arg) {
        var argType = arg.type();
        if (argType.equals(SBC_TYPE)) {
            argType = CHAR_SEQ_TYPE;
        }
        MethodRef leftMethodDesc = MethodRef.method(SBC_TYPE, "append", SBC_TYPE, argType);
        return CoreOp.invoke(leftMethodDesc, stringBuilder, arg);
    }

}