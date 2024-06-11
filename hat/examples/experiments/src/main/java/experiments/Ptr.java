


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


    import hat.Accelerator;
    import hat.backend.Backend;
    import hat.buffer.S32Array;
    import hat.optools.FuncOpWrapper;
    import hat.optools.InvokeOpWrapper;
    import hat.util.Result;

    import java.lang.invoke.MethodHandles;
    import java.lang.management.OperatingSystemMXBean;
    import java.lang.reflect.Method;
    import java.lang.reflect.code.Block;
    import java.lang.reflect.code.Body;
    import java.lang.reflect.code.CopyContext;
    import java.lang.reflect.code.Op;
    import java.lang.reflect.code.OpTransformer;
    import java.lang.reflect.code.TypeElement;
    import java.lang.reflect.code.Value;
    import java.lang.reflect.code.interpreter.Interpreter;
    import java.lang.reflect.code.op.CoreOp;
    import java.lang.reflect.code.op.ExternalizableOp;
    import java.lang.reflect.code.type.JavaType;
    import java.lang.reflect.code.type.PrimitiveType;
    import java.lang.runtime.CodeReflection;
    import java.util.ArrayList;
    import java.util.List;
    import java.util.Map;

    public class Ptr {


        @CodeReflection
        public static void addMul(S32Array s32Array, int add, int mul) {
            for (int i = 0; i < s32Array.length(); i++) {
                s32Array.array(i, (s32Array.array(i) + add) * mul);
            }
        }

        public static abstract class HatOp extends ExternalizableOp {
            private final TypeElement type;

            HatOp(String opName) {
                super(opName, List.of());
                this.type = JavaType.INT;
            }

            HatOp(String opName, TypeElement type, List<Value> operands) {
                super(opName, operands);
                this.type = type;
            }

            HatOp(String opName, TypeElement type, List<Value> operands, Map<String, Object> attributes) {
                super(opName, operands);
                this.type = type;
            }

            HatOp(HatOp that, CopyContext cc) {
                super(that, cc);
                this.type = that.type;
            }

            @Override
            public TypeElement resultType() {
                return type;
            }

           // @Override
           // public Body body() {
          //      return body;
          //  }

           // public String functionName() {
             //   return functionName;
          //  }
        }
        public static class HatPtrOp extends HatOp  {
          //  InvokeOpWrapper ifaceInvokeOpWrapper;

            public HatPtrOp(TypeElement typeElement, List<Value> operands) {

                super("Ptr",typeElement,operands);
            }

          //  public HatPtrOp(InvokeOpWrapper ifaceInvokeOpWrapper) {
            //    this();
            //    this.ifaceInvokeOpWrapper = ifaceInvokeOpWrapper;
           // }

            public HatPtrOp(HatOp that, CopyContext cc) {
                super(that, cc);
            }

            @Override
            public Op transform(CopyContext cc, OpTransformer ot) {
                return new HatPtrOp(this, cc);
            }


           // @Override
           // public Block.Builder lower(Block.Builder builder, OpTransformer opTransformer) {

             //   builder.op(ifaceInvokeOpWrapper.op());
             //   return builder;
           // }
        }


        static public void main(String[] args) throws Exception {
            Method method = Ptr.class.getDeclaredMethod("addMul", S32Array.class, int.class, int.class);
            FuncOpWrapper funcOpWrapper = new FuncOpWrapper(method.getCodeModel().get());
            FuncOpWrapper transformedFuncOpWrapper = funcOpWrapper.transformIfaceInvokes((builder, invokeOpWrapper)->{
                CopyContext cc = builder.context();
                Value arg2 = cc.getValue(funcOpWrapper.parameter(2));
              //  Value receiver = cc.
               // Value receiver = cc.getValue(invokeOpWrapper.operandNAsValue(0));
                List<Value> operands = new ArrayList<>();
                operands.add(arg2);
                builder.op(new HatPtrOp( JavaType.INT,operands));
            });

            System.out.println(transformedFuncOpWrapper.toText());

            var loweredFuncOpWrapper =  transformedFuncOpWrapper.lower();
            System.out.println(transformedFuncOpWrapper.toText());
            Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
            var s32Array = S32Array.create(accelerator, 64);
            for (int i = 0; i < 64; i++) {
                s32Array.array(i, i);
            }

            Interpreter.invoke(MethodHandles.lookup(),loweredFuncOpWrapper.op(), s32Array, 2, 2);
            for (int i = 0; i < 6; i++) {
                System.out.println(s32Array.array(i));
            }
        }
    }

