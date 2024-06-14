


    /*
     * Copyright (c) 2024 Intel Corporation. All rights reserved.
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


    import hat.buffer.S32Array;
    import hat.ops.HatPtrOp;
    import hat.optools.FuncOpWrapper;

    import java.lang.reflect.Method;
    import java.lang.reflect.code.CopyContext;
    import java.lang.reflect.code.Op;
    import java.lang.reflect.code.Value;
    import java.lang.reflect.code.type.JavaType;
    import java.lang.runtime.CodeReflection;
    import java.util.List;

    public class Ptr {


        @CodeReflection
        public static void addMul(S32Array s32Array, int add, int mul) {
            for (int i = 0; i < s32Array.length(); i++) {
                s32Array.array(i, (s32Array.array(i) + add) * mul);
            }
        }


        static public void main(String[] args) throws Exception {
            Method method = Ptr.class.getDeclaredMethod("addMul", S32Array.class, int.class, int.class);
            FuncOpWrapper funcOpWrapper = new FuncOpWrapper(method.getCodeModel().get());
            System.out.println(funcOpWrapper.toText());
            FuncOpWrapper transformedFuncOpWrapper = funcOpWrapper.transformIfaceInvokes((builder, invokeOpWrapper)->{
             //   builder.op(invokeOpWrapper.op());
                CopyContext cc = builder.context();
                List<Value> inputOperands = invokeOpWrapper.operands();
                List<Value> outputOperands = cc.getValues(inputOperands);
                Op.Result inputResult = invokeOpWrapper.result();
               // builder.op(invokeOpWrapper.op());
                Op.Result outputResult = builder.op(new HatPtrOp(JavaType.INT, outputOperands));
                cc.mapValue(inputResult, outputResult);

            });

          //  System.out.println(transformedFuncOpWrapper.toText());
            var loweredFuncOpWrapper =  transformedFuncOpWrapper.lower();
           // System.out.println(transformedFuncOpWrapper.toText());
            System.out.println(loweredFuncOpWrapper.toText());
             var ssa = loweredFuncOpWrapper.ssa();
            System.out.println(ssa.toText());
        }
    }

