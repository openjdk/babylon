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
import hat.buffer.S32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;

import static hat.KernelContext.GIX;
import static hat.KernelContext.GSX;

/**
 * How to run?
 * <code>HAT=MC,SW,TC java @.ffi-opencl-test hat.test.TestMinCopies</code>
 */
public class TestMinCopies {

    @Reflect
    public static void vectorAddition( S32Array arrayA, S32Array arrayB, S32Array arrayC) {
        if (GIX() < GSX()) {
            int valueA = arrayA.array(GIX());
            int valueB = arrayB.array(GIX());
            arrayC.array(GIX(), (valueA + valueB));
        }
    }

    @Reflect
    public static void vectorAdd(ComputeContext cc, S32Array arrayA, S32Array arrayB, S32Array arrayC) {
        cc.dispatchKernel(NDRange.of1D(arrayA.length()),() -> vectorAddition( arrayA, arrayB, arrayC));

        // Call to suggest a copy out on exit for the specified buffers
        cc.copyOutOnExit(arrayC);
    }

    @HatTest
    @Reflect
    public static void testVectorAddition() {
        final int size = 8192;
        var accelerator = new Accelerator(MethodHandles.lookup());
        var arrayA = S32Array.create(accelerator, size);
        var arrayB = S32Array.create(accelerator, size);
        var arrayC = S32Array.create(accelerator, size);

        arrayA.fill(i -> i);
        arrayB.fill(i -> 100 + i);

        for (int i = 0; i < 10; i++) {
            final int val = i;
            arrayA.fill(_ -> 100 + val);
            arrayB.fill(_ -> 100);

            accelerator.compute(cc -> vectorAdd(cc, arrayA, arrayB, arrayC));

            for (int k = 0; k < arrayA.length(); k++) {
                HATAsserts.assertEquals(arrayA.array(k) + arrayB.array(k), arrayC.array(k));
            }
        }

    }

}
