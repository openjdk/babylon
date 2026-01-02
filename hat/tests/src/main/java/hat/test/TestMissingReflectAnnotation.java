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
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import jdk.incubator.code.Reflect;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.test.exceptions.HATExpectedFailureException;

import java.lang.invoke.MethodHandles;

public class TestMissingReflectAnnotation {

    @Reflect
    public static int squareit(int v) {
        return  v * v;

    }

    @Reflect
    public static void squareKernel(@RO KernelContext kc, @RW S32Array array) {
        if (kc.gix < kc.gsx){
            int value = array.array(kc.gix);
            array.array(kc.gix, squareit(value));
        }
    }

    public static void squareKernelWithoutReflectAnnotation(@RO KernelContext kc, @RW S32Array array) {
        if (kc.gix < kc.gsx){
            int value = array.array(kc.gix);
            array.array(kc.gix, squareit(value));
        }
    }

    @Reflect
    public static void square(@RO ComputeContext cc, @RW S32Array array) {
        cc.dispatchKernel(NDRange.of1D(array.length()),
                kc -> squareKernelWithoutReflectAnnotation(kc, array)
        );
    }

    public static void squareWithoutReflectAnnotation(@RO ComputeContext cc, @RW S32Array array) {
        cc.dispatchKernel(NDRange.of1D(array.length()),
                kc -> squareKernel(kc, array)
        );
    }

    @HatTest
    @Reflect
    public static void testComputeMethodWithoutReflectAnnotation() {
        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = S32Array.create(accelerator, size);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        try {
            accelerator.compute(cc -> TestMissingReflectAnnotation.squareWithoutReflectAnnotation(cc, array));
        } catch (RuntimeException e) {
            HATAsserts.assertEquals("Failed to create ComputeCallGraph (did you miss @Reflect annotation?).", e.getMessage());
            return;
        }
        throw new HATExpectedFailureException("Failed to create ComputeCallGraph (did you miss @Reflect annotation?).");
    }

    @HatTest
    @Reflect
    public static void testKernelMethodWithoutReflectAnnotation() {
        final int size = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var array = S32Array.create(accelerator, size);

        // Initialize array
        for (int i = 0; i < array.length(); i++) {
            array.array(i, i);
        }

        try {
            accelerator.compute(cc -> TestMissingReflectAnnotation.square(cc, array));
        } catch (RuntimeException e) {
            HATAsserts.assertTrue(e.getMessage().contains("Failed to create KernelCallGraph (did you miss @Reflect annotation?)."));
            return;
        }
        throw new HATExpectedFailureException("Failed to create KernelCallGraph (did you miss @Reflect annotation?).");
    }
}
