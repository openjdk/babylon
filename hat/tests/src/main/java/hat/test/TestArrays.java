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
import hat.ComputeContext;
import hat.NDRange;
import hat.Global1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestArrays {

    @CodeReflection
    public static int squareit(int v) {
        return  v * v;

    }

    @CodeReflection
    public static void squareKernel(@RO KernelContext kc, @RW S32Array array) {
        if (kc.gix < kc.gsx){
            int value = array.array(kc.gix);
            array.array(kc.gix, squareit(value));
        }
    }

    @CodeReflection
    public static void square(@RO ComputeContext cc, @RW S32Array array) {
        NDRange ndRange = NDRange.of(new Global1D(array.length()));
        cc.dispatchKernel(ndRange,
                kc -> squareKernel(kc, array)
        );
    }

    @CodeReflection
    public static void vectorAddition(@RO KernelContext kc, @RO S32Array arrayA, @RO S32Array arrayB, @RW S32Array arrayC) {
        if (kc.gix < kc.gsx) {
            int valueA = arrayA.array(kc.gix);
            int valueB = arrayB.array(kc.gix);
            arrayC.array(kc.gix, (valueA + valueB));
        }
    }

    @CodeReflection
    public static void vectorAdd(@RO ComputeContext cc, @RO S32Array arrayA, @RO S32Array arrayB, @RW S32Array arrayC) {
        NDRange ndRange = NDRange.of(new Global1D(arrayA.length()));
        cc.dispatchKernel(ndRange,
                kc -> vectorAddition(kc, arrayA, arrayB, arrayC)
        );
    }

    @CodeReflection
    public static void saxpy(@RO KernelContext kc, @RO F32Array arrayA, @RO F32Array arrayB, @RW F32Array arrayC, float alpha) {
        if (kc.gix < kc.gsx) {
            float valueA = arrayA.array(kc.gix);
            float valueB = arrayB.array(kc.gix);
            float result = alpha * valueA + valueB;
            arrayC.array(kc.gix, result);
        }
    }

    @CodeReflection
    public static void computeSaxpy(@RO ComputeContext cc, @RO F32Array arrayA, @RO F32Array arrayB, @RW F32Array arrayC, float alpha) {
        NDRange ndRange = NDRange.of(new Global1D(arrayA.length()));
        cc.dispatchKernel(ndRange,
                kc -> saxpy(kc, arrayA, arrayB, arrayC, alpha)
        );
    }

    @HatTest
    public static void testHelloHat() {
        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = S32Array.create(accelerator, size);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        // Blocking call
        accelerator.compute(cc -> TestArrays.square(cc, array));

        S32Array test = S32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, squareit(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), array.array(i));
        }
    }

    @HatTest
    public static void testVectorAddition() {
        final int size = 8192;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = S32Array.create(accelerator, size);
        var arrayB = S32Array.create(accelerator, size);
        var arrayC = S32Array.create(accelerator, size);

        // Initialize array
        arrayA.fill(i -> i);
        arrayB.fill(i -> 100 + i);

        accelerator.compute(cc ->
                TestArrays.vectorAdd(cc, arrayA, arrayB, arrayC));

        S32Array test = S32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, arrayA.array(i) + arrayB.array(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), arrayC.array(i));
        }
    }

    @HatTest
    public static void testVectorSaxpy() {
        final int size = 8192;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

        // Initialize array
        Random r = new Random(71);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        var alpha = 0.2f;
        accelerator.compute(cc ->
                TestArrays.computeSaxpy(cc, arrayA, arrayB, arrayC, alpha));

        F32Array test = F32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, alpha * arrayA.array(i) + arrayB.array(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), arrayC.array(i), 0.01f);
        }
    }

}
