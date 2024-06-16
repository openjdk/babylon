


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


    import hat.HatOps;
    import hat.buffer.KernelContext;
    import hat.buffer.S32Array;
    import hat.optools.FuncOpWrapper;
    import hat.optools.InvokeOpWrapper;

    import java.lang.reflect.Method;
    import java.lang.runtime.CodeReflection;

    public class Ptr {


        @CodeReflection
        public static void addMul(KernelContext kernelContext, S32Array s32Array, int add, int mul) {
            int x = kernelContext.x();
            s32Array.array(x, (s32Array.array(x) + add) * mul);
        }


        static public void main(String[] args) throws Exception {
            Method method = Ptr.class.getDeclaredMethod("addMul", KernelContext.class, S32Array.class, int.class, int.class);

            FuncOpWrapper funcOpWrapper = new FuncOpWrapper(method.getCodeModel().get());
            System.out.println(funcOpWrapper.toText());

            FuncOpWrapper transformedFuncOpWrapper = funcOpWrapper.findMapAndReplace(
                    (w)-> w instanceof InvokeOpWrapper invokeOpWrapper&& invokeOpWrapper.isIfaceBufferMethod(),  // Selector
                    (iw)-> (InvokeOpWrapper)iw,            // Mapper (so that wb.current() is type we want)
                    (wb) -> {
                if (wb.current().isIfaceBufferMethod()) {
                    if (wb.current().isIfaceAccessor()) {
                        if (wb.current().isKernelContextAccessor()) {
                            wb.replace(new HatOps.HatKernelContextOp(wb.resultType(), wb.operandValues()));
                        } else {
                            wb.replace(new HatOps.HatPtrLoadOp(wb.resultType(), wb.operandValues()));
                        }
                    } else {
                        wb.replace(new HatOps.HatPtrStoreOp(wb.resultType(), wb.operandValues()));
                    }
                }
            });
            System.out.println(transformedFuncOpWrapper.toText());
            var loweredFuncOpWrapper = transformedFuncOpWrapper.lower();
            System.out.println(loweredFuncOpWrapper.toText());
            var ssa = loweredFuncOpWrapper.ssa();
            System.out.println(ssa.toText());
        }
    }

