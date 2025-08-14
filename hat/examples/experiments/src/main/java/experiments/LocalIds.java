/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.LocalMesh1D;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

/**
 * How to test?
 * <code>
 *     HAT=SHOW_CODE java -cp job.jar hat.java exp ffi-opencl LocalIds
 * </code>
 */
public class LocalIds {

    @CodeReflection
    private static void assign(@RO KernelContext context, @RW S32Array arrayA, @RW S32Array arrayB, @RW S32Array arrayC) {
        int gix = context.gix;
        int lix = context.lix;
        int lsx = context.lsx;
        int bsx = context.bsx;
        arrayA.array(gix, lix);
        arrayB.array(gix, lsx);
        arrayC.array(gix, bsx);
    }

    @CodeReflection
    private static void mySimpleCompute(@RO ComputeContext cc,  @RW S32Array arrayA, @RW S32Array arrayB, @RW S32Array arrayC) {
        // 2 groups of 16 threads each
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(32), new LocalMesh1D(16));
        cc.dispatchKernel(computeRange, kc -> assign(kc, arrayA, arrayB, arrayC));
    }

    public static void main(String[] args) {
        System.out.println("Experiment: local IDs and local groups");

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 32;
        S32Array arrayA = S32Array.create(accelerator, size);
        S32Array arrayB = S32Array.create(accelerator, size);
        S32Array arrayC = S32Array.create(accelerator, size);

        // Set initial value to 0
        arrayA.fill(i -> 0);
        arrayB.fill(i -> 0);
        arrayC.fill(i -> 0);

        // Compute on the accelerator
        accelerator.compute( cc -> LocalIds.mySimpleCompute(cc, arrayA, arrayB, arrayC));

        System.out.println("Execution finished");
        System.out.println("Result Locals: ");
        for (int i = 0; i < arrayA.length(); i++) {
            System.out.println(arrayA.array(i));
        }

        System.out.println("Result Blocks: ");
        for (int i = 0; i < arrayB.length(); i++) {
            System.out.println(arrayB.array(i));
        }

        System.out.println("Result Block ID: ");
        for (int i = 0; i < arrayC.length(); i++) {
            System.out.println(arrayC.array(i));
        }
    }

}
