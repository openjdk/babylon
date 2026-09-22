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
package hat.test;

import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;

/**
 * Test to check with Java Flight Recording.
 */
public class TestStress {

    @Reflect
    public static void compute(F32Array arrayA, F32Array arrayB, F32Array arrayC, F32Array arrayD, F32Array arrayE, F32Array arrayF, F32Array arrayG, F32Array arrayH) {
        final int idx = KernelContext.GIX();
        // write on every buffer
        arrayA.array(idx, 0);
        arrayB.array(idx, 0);
        arrayC.array(idx, 0);
        arrayD.array(idx, 0);
        arrayE.array(idx, 0);
        arrayF.array(idx, 0);
        arrayG.array(idx, 0);
        arrayH.array(idx, 0);
    }

    @Reflect
    public static void compute(ComputeContext cc, F32Array arrayA, F32Array arrayB, F32Array arrayC, F32Array arrayD, F32Array arrayE, F32Array arrayF, F32Array arrayG, F32Array arrayH) {
        cc.dispatchKernel(NDRange.of1D(arrayA.length()), () -> TestStress.compute(arrayA, arrayB, arrayC, arrayD, arrayE, arrayF, arrayG, arrayH));
    }

    @HatTest
    public void stressMemoryTest() {
        var accelerator = new Accelerator(MethodHandles.lookup());
        final int size = Math.powExact(2, 16);
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);
        F32Array arrayC = F32Array.create(accelerator, size);
        F32Array arrayD = F32Array.create(accelerator, size);
        F32Array arrayE = F32Array.create(accelerator, size);
        F32Array arrayF = F32Array.create(accelerator, size);
        F32Array arrayG = F32Array.create(accelerator, size);
        F32Array arrayH = F32Array.create(accelerator, size);

        final int iterations = 1000;
        for (int i = 0; i < iterations; i++) {
            for (int j = 0; j < iterations; j++) {
                accelerator.compute((@Reflect Compute) cc -> compute(cc, arrayA, arrayB, arrayC, arrayD, arrayE, arrayF, arrayG, arrayH));
            }
        }
    }
}
