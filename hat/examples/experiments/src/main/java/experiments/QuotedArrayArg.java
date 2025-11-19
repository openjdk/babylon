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

import hat.Accelerator;
import hat.ComputeContext;
import hat.NDRange;
import hat.KernelContext;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

public class QuotedArrayArg {
    @CodeReflection
    public static void addScalerKernel(@MappableIface.RO KernelContext kc, @MappableIface.RO S32Array in, @MappableIface.WO S32Array out, int scaler) {
        out.array(kc.gix, in.array(kc.gix) + scaler);
    }

    @CodeReflection
    static public void addScalerCompute(final ComputeContext computeContext, S32Array in, S32Array out, int scaler) {
        computeContext.dispatchKernel(NDRange.of(in.length()), kc -> QuotedConstantArgs.addScalerKernel(kc, in, out, scaler));
    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());
        S32Array in = S32Array.create(accelerator, 32);
        in.fill((idx) -> idx);
        S32Array out = S32Array.create(accelerator, 32);
        int[] array = new int[]{1};
        int index = 0;
        // This works
        if (args.length == 0) {
            int lvar = array[index];
            accelerator.compute(computeContext -> QuotedConstantArgs.addScalerCompute(computeContext, in, out,lvar));
        }else {
            if (args.length == 1 && args[0].equals("passIndex")) {
                accelerator.compute(computeContext -> QuotedConstantArgs.addScalerCompute(computeContext, in, out, array[index]));
            }else if (args.length == 1 && args[0].equals("passZero")) {
                accelerator.compute(computeContext -> QuotedConstantArgs.addScalerCompute(computeContext, in, out, array[0]));
            }else{
                throw new IllegalArgumentException("Invalid args either no args, passIndex or passZero");
            }
        }
        for (int i = 0; i < in.length(); i++) {
            System.out.println("["+i+"]  in=" + in.array(i) + " out=" + out.array(i));
        }
    }

}
