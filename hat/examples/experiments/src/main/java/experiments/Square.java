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
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.S32Array;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;

import java.lang.invoke.MethodHandles;

import static hat.backend.Backend.FIRST;
import static optkl.OpHelper.Func.func;

public class Square {

    @Reflect
    public static int square(int v) {
        return v * v;
    }

    @Reflect
    public static void squareKernel(@RO KernelContext kc, @RW S32Array a) {
        int id = kc.gix;
        if (id < kc.gsx) {
            int value = a.array(id);
            a.array(id, square(value));
        }
    }

    @Reflect
    public static void squareCompute(ComputeContext cc, @RW S32Array s32Array) {
        cc.dispatchKernel(NDRange.of1D(s32Array.length()), kc -> squareKernel(kc, s32Array));
    }

     static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), FIRST);

        S32Array s32Array = S32Array.create(accelerator, 20);
        s32Array.fill(i -> i); // fill arrays with unity a[0] = 0, a[1] = 1 ....
        accelerator.compute((@Reflect Compute)
                cc -> Square.squareCompute(cc, s32Array));

        for (int i = 0; i < 20; i++) {
            System.out.println(i + "=" + s32Array.array(i));
        }
    }

}
