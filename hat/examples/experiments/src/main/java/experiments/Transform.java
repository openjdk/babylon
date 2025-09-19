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

package experiments;


import java.lang.reflect.Method;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.util.List;
import java.util.Map;

import static jdk.incubator.code.dialect.core.CoreType.FUNCTION_TYPE_VOID;

public class Transform {
        @CodeReflection
        public static void removeMe(int size, int x, int y) {

        }

        @CodeReflection
        public static void matrixMultiply(float[] a, float[] b, float[] c, int size) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    float sum = 0f;
                    for (int k = 0; k < size; k++) {
                        sum += a[i * size + k] * b[k * size + j];
                    }
                    removeMe(1, 2, 3);

                    c[i * size + j] = sum;

                }
            }
        }

        public static abstract class MyOp extends Op {
            private final String opName;
            private final TypeElement type;

            MyOp(String opName) {
                super(List.of());
                this.opName = opName;
                this.type = FUNCTION_TYPE_VOID;
            }

            MyOp(String opName, TypeElement type, List<Value> operands) {
                super(operands);
                this.opName = opName;
                this.type = type;
            }

            MyOp(String opName, TypeElement type, List<Value> operands, Map<String, Object> attributes) {
                super(operands);
                this.opName = opName;
                this.type = type;
            }

            MyOp(MyOp that, CopyContext cc) {
                super(that, cc);
                this.opName = that.opName;
                this.type = that.type;
            }

            @Override
            public TypeElement resultType() {
                return type;
            }
        }


        public static class RootOp extends MyOp {
            public RootOp() {
                super("Root");
            }

            public RootOp(MyOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public Op transform(CopyContext cc, OpTransformer ot) {
                return new RootOp(this, cc);
            }
        }

        public static void pre() {
        }

        ;

        public static void post() {
        }

        ;

        public static final MethodRef PRE = MethodRef.method(Transform.class, "pre", void.class);
        public static final MethodRef POST = MethodRef.method(Transform.class, "post", void.class);

        static public void main(String[] args) throws Exception {
            String methodName = "matrixMultiply";
            Method method = Transform.class.getDeclaredMethod(methodName, float[].class, float[].class, float[].class, int.class);

            CoreOp.FuncOp javaFunc = Op.ofMethod(method).get();

            CoreOp.FuncOp transformed = javaFunc.transform((builder, op) -> {
                if (op instanceof JavaOp.InvokeOp invokeOp) {
                    //  CopyContext cc = builder.context();
                    //  Block.Builder bb = builder;
                    // var invokePre = CoreOp.invoke(PRE);
                    RootOp rootOp = new RootOp();
                    // builder.op(rootOp);
                    //  builder.op(invokeOp);
                    //  builder.op(CoreOp.invoke(POST));
                } else {
                    builder.op(op);
                }
                return builder;
            });

            System.out.println(transformed.toText());

        }
}

