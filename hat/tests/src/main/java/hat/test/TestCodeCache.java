/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import hat.ComputeContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.Accelerator.Compute;
import static hat.KernelContext.GIX;

public class TestCodeCache {

    @Reflect
    public static void vectorAdd(F32Array arrayA, F32Array arrayB, F32Array arrayC) {
        if (GIX() < arrayA.length()) {
            var valueA = arrayA.array(GIX());
            var valueB = arrayB.array(GIX());
            arrayC.array(GIX(), (valueA + valueB));
        }
    }

    @Reflect
    public static void vectorAdd(ComputeContext cc, F32Array arrayA, F32Array arrayB, F32Array arrayC) {
        cc.dispatchKernel(NDRange.of1D(arrayA.length()), () -> vectorAdd(arrayA, arrayB, arrayC));
    }

    private static void initArrayRandom(F32Array arrayA) {
        Random r = new Random(19);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i, r.nextFloat(1));
        }
    }

    private static void check(F32Array arrayA, F32Array arrayB, F32Array arrayC) {
        for (int i = 0; i < arrayA.length(); i++) {
            HATAsserts.assertEquals(arrayA.array(i) + arrayB.array(i), arrayC.array(i), 0.01f);
        }
    }

    // Test run the same compute multiple times using the same input/output sizes, but different
    // I/O objects. We should hit the code-cache since we do not specialize based on sizes for the
    // SIMT model, just based on constants that are folded into the model.
    @HatTest
    public static void testCodeCache01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;

        final int numDispatches = 10;
        for (int i = 0; i < numDispatches; i++) {
            F32Array arrayA = F32Array.create(accelerator, size);
            F32Array arrayB = F32Array.create(accelerator, size);
            F32Array arrayC = F32Array.create(accelerator, size);
            initArrayRandom(arrayA);
            initArrayRandom(arrayB);

            // It should hit the code-cache
            accelerator.compute((@Reflect Compute) cc -> vectorAdd(cc, arrayA, arrayB, arrayC));
            check(arrayA, arrayB, arrayC);

        }
    }

    @HatTest
    public static void testCodeCache02() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;

        final int numDispatches = 10;
        int factor = 1;
        for (int i = 0; i < numDispatches; i++) {
            F32Array arrayA = F32Array.create(accelerator, size * factor);
            F32Array arrayB = F32Array.create(accelerator, size * factor);
            F32Array arrayC = F32Array.create(accelerator, size * factor);
            initArrayRandom(arrayA);
            initArrayRandom(arrayB);
            factor *= 2;

            // It should hit the code-cache
            accelerator.compute((@Reflect Compute) cc -> vectorAdd(cc, arrayA, arrayB, arrayC));
            check(arrayA, arrayB, arrayC);

        }
    }



}
