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
import hat.HATMath;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.F16;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.WO;

import java.lang.invoke.MethodHandles;
import java.util.Random;
import java.util.stream.IntStream;

public class TestHATMathLib {

    @Reflect
    private static void testMathLib01(@RO KernelContext kc, @RO F16Array a, @RO F16Array b, @WO F16Array c) {
        if (kc.gix < kc.gsx) {
            F16 ha = a.array(kc.gix);
            F16 hb = b.array(kc.gix);
            F16 result = HATMath.maxf16(ha, hb);
            F16 hC = c.array(kc.gix);
            hC.value(result.value());
        }
    }

    @Reflect
    private static void computeMathLib01(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @WO F16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib01(kernelContext, a, b, c));
    }

    @Reflect
    private static void testMathLib02(@RO KernelContext kc, @RO F32Array a, @RO F32Array b, @WO F32Array c) {
        if (kc.gix < kc.gsx) {
            float ha = a.array(kc.gix);
            float hb = b.array(kc.gix);
            float result = HATMath.maxf(ha, hb);
            c.array(kc.gix, result);
        }
    }

    @Reflect
    private static void computeMathLib02(@RO ComputeContext computeContext, @RO F32Array a, @RO F32Array b, @WO F32Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib02(kernelContext, a, b, c));
    }

    @Reflect
    private static void testMathLib03(@RO KernelContext kc, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        if (kc.gix < kc.gsx) {
            F16 ha = a.array(kc.gix);
            F16 hb = b.array(kc.gix);
            F16 hC = c.array(kc.gix);

            F16 result = HATMath.maxf16(ha, hb);
            result = HATMath.maxf16(result, hC);

            F16 hD = d.array(kc.gix);
            hD.value(result.value());
        }
    }

    @Reflect
    private static void computeMathLib03(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib03(kernelContext, a, b, c, d));
    }

    @Reflect
    private static void testMathLib04(@RO KernelContext kc, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        if (kc.gix < kc.gsx) {
            F16 ha = a.array(kc.gix);
            F16 hb = b.array(kc.gix);
            F16 hC = c.array(kc.gix);

            F16 init = F16.of(2.0f);
            F16 result = HATMath.maxf16(F16.add(ha, hb), F16.mul(hC, init));

            F16 hD = d.array(kc.gix);
            hD.value(result.value());
        }
    }

    @Reflect
    private static void computeMathLib04(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib04(kernelContext, a, b, c, d));
    }

    @Reflect
    private static void testMathLib05(@RO KernelContext kc, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        if (kc.gix < kc.gsx) {
            F16 ha = a.array(kc.gix);
            F16 hb = b.array(kc.gix);
            F16 hC = c.array(kc.gix);

            F16 init = F16.of(2.0f);
            F16 result = HATMath.maxf16(HATMath.maxf16(ha, hb), HATMath.maxf16(hC, init));

            F16 hD = d.array(kc.gix);
            hD.value(result.value());
        }
    }

    @Reflect
    private static void computeMathLib05(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RO F16Array c, @WO F16Array d) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib05(kernelContext, a, b, c, d));
    }

    @Reflect
    private static void testMathLib06(@RO KernelContext kc, @RO F16Array a, @WO F16Array b) {
        if (kc.gix < kc.gsx) {
            F16 ha = a.array(kc.gix);
            F16 result = HATMath.expf16(ha);
            F16 hB = b.array(kc.gix);
            hB.value(result.value());
        }
    }

    @Reflect
    private static void computeMathLib06(@RO ComputeContext computeContext, @RO F16Array a, @WO F16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib06(kernelContext, a, b));
    }

    @Reflect
    private static void testMathLib07(@RO KernelContext kc, @RO F32Array a, @WO F32Array b) {
        if (kc.gix < kc.gsx) {
            float fa = a.array(kc.gix);
            float result = HATMath.cosf(fa);
            b.array(kc.gix, result);
        }
    }

    @Reflect
    private static void computeMathLib07(@RO ComputeContext computeContext, @RO F32Array a, @WO F32Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib07(kernelContext, a, b));
    }

    @Reflect
    private static void testMathLib08(@RO KernelContext kc, @RO F32Array a, @WO F32Array b) {
        if (kc.gix < kc.gsx) {
            float fa = a.array(kc.gix);
            float result = HATMath.sinf(fa);
            b.array(kc.gix, result);
        }
    }

    @Reflect
    private static void computeMathLib08(@RO ComputeContext computeContext, @RO F32Array a, @WO F32Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib08(kernelContext, a, b));
    }

    @Reflect
    private static void testMathLib09(@RO KernelContext kc, @RO F32Array a, @WO F32Array b) {
        if (kc.gix < kc.gsx) {
            float fa = a.array(kc.gix);
            float result = HATMath.tanf(fa);
            b.array(kc.gix, result);
        }
    }

    @Reflect
    private static void computeMathLib09(@RO ComputeContext computeContext, @RO F32Array a, @WO F32Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib09(kernelContext, a, b));
    }

    @Reflect
    private static void testMathLib10(@RO KernelContext kc, @RO F32Array a, @WO F32Array b) {
        if (kc.gix < kc.gsx) {
            float fa = a.array(kc.gix);
            float result = HATMath.sqrtf(fa);
            b.array(kc.gix, result);
        }
    }

    @Reflect
    private static void computeMathLib10(@RO ComputeContext computeContext, @RO F32Array a, @WO F32Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestHATMathLib.testMathLib10(kernelContext, a, b));
    }


    @HatTest
    @Reflect
    public void testMathLib01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 128;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib01(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayC.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            HATAsserts.assertEquals(Math.max(fa, fb), F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib02() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 128;
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);
        F32Array arrayC = F32Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i, random.nextFloat());
            arrayB.array(i, random.nextFloat());
        }

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib02(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayC.length(); i++) {
            float val = arrayC.array(i);
            HATAsserts.assertEquals(Math.max(arrayA.array(i), arrayB.array(i)), val, 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 128;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);
        F16Array arrayD = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayC.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib03(computeContext, arrayA, arrayB, arrayC, arrayD));

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayD.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            float fc = Float.float16ToFloat(arrayC.array(i).value());
            float result = Math.max(Math.max(fa, fb), fc);
            HATAsserts.assertEquals(result, F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 128;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);
        F16Array arrayD = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayC.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib04(computeContext, arrayA, arrayB, arrayC, arrayD));

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayD.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            float fc = Float.float16ToFloat(arrayC.array(i).value());
            float result = Math.max((fa + fb), fc * 2.0f);
            HATAsserts.assertEquals(result, F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib05() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 128;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);
        F16Array arrayD = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayC.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib05(computeContext, arrayA, arrayB, arrayC, arrayD));

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayD.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            float fc = Float.float16ToFloat(arrayC.array(i).value());
            float result = Math.max(Math.max(fa, fb), Math.max(fc, 2.0f));
            HATAsserts.assertEquals(result, F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib06() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random random = new Random();
        IntStream.range(0, arrayA.length()).forEach(i -> arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value()));

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib06(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            float fa = (float) Math.exp(F16.f16ToFloat(arrayA.array(i)));
            HATAsserts.assertEquals(fa, F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib07() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);

        Random random = new Random();
        IntStream.range(0, arrayA.length()).forEach(i -> arrayA.array(i, random.nextFloat()));

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib07(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            float val = arrayB.array(i);
            float fa = (float) Math.cos(arrayA.array(i));
            HATAsserts.assertEquals(fa, val, 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib08() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);

        Random random = new Random();
        IntStream.range(0, arrayA.length()).forEach(i -> arrayA.array(i, random.nextFloat()));

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib08(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            float val = arrayB.array(i);
            float fa = (float) Math.sin(arrayA.array(i));
            HATAsserts.assertEquals(fa, val, 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib09() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);

        Random random = new Random();
        IntStream.range(0, arrayA.length()).forEach(i -> arrayA.array(i, random.nextFloat()));

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib09(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            float val = arrayB.array(i);
            float fa = (float) Math.tan(arrayA.array(i));
            HATAsserts.assertEquals(fa, val, 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void testMathLib10() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        F32Array arrayA = F32Array.create(accelerator, size);
        F32Array arrayB = F32Array.create(accelerator, size);

        Random random = new Random();
        IntStream.range(0, arrayA.length()).forEach(i -> arrayA.array(i, random.nextFloat()));

        accelerator.compute(computeContext -> TestHATMathLib.computeMathLib10(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            float val = arrayB.array(i);
            float fa = (float) Math.sqrt(arrayA.array(i));
            HATAsserts.assertEquals(fa, val, 0.001f);
        }
    }
}
