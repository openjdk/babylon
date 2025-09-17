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
package oracle.code.hat;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;

import static hat.ifacemapper.MappableIface.*;

public class TestConstants {

    public static final int CONSTANT = 100;

    @CodeReflection
    public static void vectorWithConstants(@RO KernelContext kc, @RO S32Array arrayA, @RO S32Array arrayB, @RW S32Array arrayC) {
        final int BM = 100;
        if (kc.x < kc.gsx) {
            final int valueA = arrayA.array(kc.x);
            final int valueB = arrayB.array(kc.x);
            arrayC.array(kc.x, (BM + valueA + valueB));
        }
    }

    @CodeReflection
    public static void vectorWithConstants(@RO ComputeContext cc, @RO S32Array arrayA, @RO S32Array arrayB, @RW S32Array arrayC) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(arrayA.length()));
        cc.dispatchKernel(computeRange, kc -> vectorWithConstants(kc, arrayA, arrayB, arrayC));
    }

    @HatTest
    public static void testConstants() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = S32Array.create(accelerator, size);
        var arrayB = S32Array.create(accelerator, size);
        var arrayC = S32Array.create(accelerator, size);

        arrayA.fill(i -> i);
        arrayB.fill(i -> 100 + i);

        accelerator.compute(cc ->
                TestConstants.vectorWithConstants(cc, arrayA, arrayB, arrayC));

        S32Array test = S32Array.create(accelerator, size);

        for (int i = 0; i < test.length(); i++) {
            test.array(i, CONSTANT + arrayA.array(i) + arrayB.array(i));
        }

        for (int i = 0; i < test.length(); i++) {
            HatAsserts.assertEquals(test.array(i), arrayC.array(i));
        }
    }
}
