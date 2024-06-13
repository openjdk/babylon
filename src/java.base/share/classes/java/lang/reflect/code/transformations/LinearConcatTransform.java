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

package java.lang.reflect.code.transformations;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.MethodRef;

public class LinearConcatTransform {

    private static final JavaType SBC_TYPE = JavaType.type(StringBuilder.class);

    record StringAndBuilder(Value string, Value stringBuilder) { }

    public LinearConcatTransform() {}

    public static final class ConcatTransform implements OpTransformer {

        public ConcatTransform () {

        }

        @Override
        public Block.Builder apply(Block.Builder builder, Op op) {
            if (op instanceof CoreOp.ConcatOp cz) {

                Value left = builder.context().getValue(cz.operands().get(0));
                Value right = builder.context().getValue(cz.operands().get(1));

                Value result = cz.result();

                StringAndBuilder newRes = stringBuild(builder, left, right);
                builder.context().mapValue(result, newRes.string);
            } else {
                builder.op(op);
            }
            return builder;
        }

        //Uses StringBuilder and Immediate String Value
        private static StringAndBuilder stringBuild(Block.Builder builder, Value left, Value right) {
            var newB = stringBuilder(builder, left, right);
            var toStringMR = MethodRef.method(SBC_TYPE, "toString", JavaType.J_L_STRING);
            var toStringInvoke = CoreOp.invoke(toStringMR, newB);
            Value newString = builder.apply(toStringInvoke);
            return new StringAndBuilder(newString, newB);
        }

        private static Value stringBuilder(Block.Builder builder, Value left, Value right) {
            CoreOp.NewOp newBuilder = CoreOp._new(FunctionType.functionType(SBC_TYPE));
            Value sb;

            sb = builder.apply(newBuilder);
            var leftMethodDesc = MethodRef.method(SBC_TYPE, "append", SBC_TYPE, left.type());
            var leftInvoke = CoreOp.invoke(leftMethodDesc, sb, left);
            var rightMethodDesc = MethodRef.method(SBC_TYPE, "append", SBC_TYPE, right.type());
            var rightInvoke = CoreOp.invoke(rightMethodDesc, sb, right);
            builder.op(leftInvoke);
            builder.op(rightInvoke);
            return sb;

        }
    }
}