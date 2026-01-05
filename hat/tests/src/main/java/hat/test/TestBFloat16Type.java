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
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.types.BF16;
import hat.buffer.BF16Array;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAssertionError;
import hat.test.exceptions.HATAsserts;
import hat.test.exceptions.HATExpectedPrecisionError;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import optkl.ifacemapper.MappableIface.WO;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestBFloat16Type {

    @Reflect
    public static void kernel_copy(@RO KernelContext kernelContext, @RO BF16Array a, @WO BF16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            b.array(kernelContext.gix).value(ha.value());
        }
    }

    @Reflect
    public static void bf16_02(@RO KernelContext kernelContext, @RO BF16Array a, @RO BF16Array b, @WO BF16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 hb = b.array(kernelContext.gix);
            BF16 result = BF16.add(ha, hb);
            BF16 hc = c.array(kernelContext.gix);
            hc.value(result.value());
        }
    }

    @Reflect
    public static void bf16_03(@RO KernelContext kernelContext, @RO BF16Array a, @RO BF16Array b, @RW BF16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 hb = b.array(kernelContext.gix);

            BF16 result = BF16.add(ha, BF16.add(hb, hb));
            BF16 hC = c.array(kernelContext.gix);
            hC.value(result.value());
        }
    }

    @Reflect
    public static void bf16_04(@RO KernelContext kernelContext, @RO BF16Array a, @RO BF16Array b, @RW BF16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 hb = b.array(kernelContext.gix);

            BF16 r1 = BF16.mul(ha, hb);
            BF16 r2 = BF16.div(ha, hb);
            BF16 r3 = BF16.sub(ha, hb);
            BF16 r4 = BF16.add(r1, r2);
            BF16 r5 = BF16.add(r4, r3);
            BF16 hC = c.array(kernelContext.gix);
            hC.value(r5.value());
        }
    }

    @Reflect
    public static void bf16_05(@RO KernelContext kernelContext, @RW BF16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 initVal = BF16.of( 2.1f);
            ha.value(initVal.value());
        }
    }

    @Reflect
    public static void bf16_06(@RO KernelContext kernelContext, @RW BF16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 initVal = BF16.of(kernelContext.gix);
            BF16 ha = a.array(kernelContext.gix);
            ha.value(initVal.value());
        }
    }

    @Reflect
    public static void bf16_08(@RO KernelContext kernelContext, @RW BF16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 initVal = BF16.float2bfloat16(kernelContext.gix);
            BF16 ha = a.array(kernelContext.gix);
            ha.value(initVal.value());
        }
    }

    @Reflect
    public static void bf16_09(@RO KernelContext kernelContext, @RO BF16Array a, @WO BF16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            float f = BF16.bfloat162float(ha);
            BF16 result = BF16.float2bfloat16(f);
            BF16 hb = b.array(kernelContext.gix);
            hb.value(result.value());
        }
    }

    @Reflect
    public static void bf16_10(@RO KernelContext kernelContext, @RW BF16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 f16 = BF16.of(1.1f);
            float f = BF16.bfloat162float(f16);
            BF16 result = BF16.float2bfloat16(f);
            ha.value(result.value());
        }
    }

    public interface LocalArray extends DeviceType {
        BF16 array(int index);
        DeviceSchema<LocalArray> schema = DeviceSchema.of(LocalArray.class,
                builder -> builder.withArray("array", 1024)
                        .withDeps(BF16.class, bfloat16 -> bfloat16.withField("value")));

        static LocalArray  create(Accelerator accelerator) {
            return null;
        }

        static LocalArray createLocal() {
            return null;
        }
    }

    @Reflect
    public static void bf16_11(@RO KernelContext kernelContext, @RO BF16Array a, @RW BF16Array b) {
        LocalArray sm = LocalArray.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            BF16 ha = a.array(kernelContext.gix);

            sm.array(lix).value(ha.value());
            kernelContext.barrier();

            BF16 hb = sm.array(lix);
            b.array(kernelContext.gix).value(hb.value());
        }
    }

    @Reflect
    public static void bf16_12(@RO KernelContext kernelContext, @RO BF16Array a, @RO BF16Array b, @RW BF16Array c) {
        // Test the fluent API style
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 hb = b.array(kernelContext.gix);
            BF16 result = ha.add(hb);
            c.array(kernelContext.gix).value(result.value());
        }
    }

    @Reflect
    public static void bf16_13(@RO KernelContext kernelContext, @RO BF16Array a, @RO BF16Array b,  @RW BF16Array c) {
        // Test the fluent API style
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            BF16 hb = b.array(kernelContext.gix);
            BF16 result = ha.add(hb).sub(hb).mul(ha).div(ha);
            c.array(kernelContext.gix).value(result.value());
        }
    }

    @Reflect
    public static void bf16_14(@RO KernelContext kernelContext, @RO BF16Array a, @RW BF16Array b) {
        // Testing mixed float types
        if (kernelContext.gix < kernelContext.gsx) {
            BF16 ha = a.array(kernelContext.gix);
            float myFloat = 32.1f;
            BF16 result = BF16.add(myFloat, ha);
            b.array(kernelContext.gix).value(result.value());
        }
    }

    public interface PrivateArray extends DeviceType {
        BF16 array(int index);
        DeviceSchema<PrivateArray> schema = DeviceSchema.of(PrivateArray.class,
                builder -> builder.withArray("array", 256)
                        .withDeps(BF16.class, bfloat16 -> bfloat16.withField("value")));

        static PrivateArray  create(Accelerator accelerator) {
            return null;
        }

        static PrivateArray createPrivate() {
            return null;
        }
    }

    @Reflect
    public static void bf16_15(@RO KernelContext kernelContext, @RO BF16Array a, @RW BF16Array b) {
        PrivateArray privateArray = PrivateArray.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            BF16 ha = a.array(kernelContext.gix);
            privateArray.array(lix).value(ha.value());
            BF16 hb = privateArray.array(lix);
            b.array(kernelContext.gix).value(hb.value());
        }
    }

    @Reflect
    public static void bf16_16(@RO KernelContext kernelContext, @RW BF16Array a) {
        BF16 ha = a.array(0);
        BF16 hre = BF16.add(ha, ha);
        hre = BF16.add(hre, hre);
        a.array(0).value(hre.value());
    }

    @Reflect
    public static void bf16_17(@RO KernelContext kernelContext, @RW BF16Array a) {

        BF16 ha = a.array(0);
        PrivateArray privateArray = PrivateArray.createPrivate();
        privateArray.array(0).value(ha.value());

        // Obtain the value from private memory
        BF16 acc = privateArray.array(0);

        // compute
        acc = BF16.add(acc, acc);

        // store the result
        a.array(0).value(acc.value());
    }

    @Reflect
    public static void compute01(@RO ComputeContext computeContext, @RO BF16Array a, @WO BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.kernel_copy(kernelContext, a, b));
    }

    @Reflect
    public static void compute02(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @WO BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestBFloat16Type.bf16_02(kernelContext, a, b, c));
    }

    @Reflect
    public static void compute03(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @WO BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestBFloat16Type.bf16_03(kernelContext, a, b, c));
    }

    @Reflect
    public static void compute04(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @WO BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestBFloat16Type.bf16_04(kernelContext, a, b, c));
    }

    @Reflect
    public static void compute05(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_05(kernelContext, a));
    }

    @Reflect
    public static void compute06(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_06(kernelContext, a));
    }

    @Reflect
    public static void compute08(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_08(kernelContext, a));
    }

    @Reflect
    public static void compute09(@RO ComputeContext computeContext, @RW BF16Array a, @WO BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_09(kernelContext, a, b));
    }

    @Reflect
    public static void compute10(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_10(kernelContext, a));
    }

    @Reflect
    public static void compute11(@RO ComputeContext computeContext, @RO BF16Array a, @RW BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length(),16), kernelContext -> TestBFloat16Type.bf16_11(kernelContext, a, b));
    }

    @Reflect
    public static void compute12(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @RW BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_12(kernelContext, a, b, c));
    }

    @Reflect
    public static void compute13(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @RW BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_13(kernelContext, a, b, c));
    }

    @Reflect
    public static void compute14(@RO ComputeContext computeContext, @RO BF16Array a, @RW BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_14(kernelContext, a, b));
    }

    @Reflect
    public static void compute15(@RO ComputeContext computeContext, @RO BF16Array a, @RW BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.bf16_15(kernelContext, a, b));
    }

    @Reflect
    public static void compute16(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(1), kernelContext -> TestBFloat16Type.bf16_16(kernelContext, a));
    }

    @Reflect
    public static void compute17(@RO ComputeContext computeContext, @RW BF16Array a) {
        computeContext.dispatchKernel(NDRange.of1D(1), kernelContext -> TestBFloat16Type.bf16_17(kernelContext, a));
    }

    @HatTest
    @Reflect
    public void test_bfloat16_01() {
        final int size = 256;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        for (int i = 0; i < size; i++) {
            arrayA.array(i).value(BF16.float2bfloat16(i).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute01(computeContext, arrayA, arrayB));

        for (int i = 0; i < size; i++) {
            BF16 result = arrayB.array(i);
            HATAsserts.assertEquals((float)i, BF16.bfloat162float(result), 0.001f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_02() {
        final int size = 256;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute02(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < size; i++) {
            BF16 result = arrayC.array(i);
            BF16 a = arrayA.array(i);
            BF16 b = arrayB.array(i);
            float res = BF16.bfloat162float(a) + BF16.bfloat162float(b);
            HATAsserts.assertEquals(res, BF16.bfloat162float(result), 0.001f);
        }
    }
    @HatTest
    @Reflect
    public void test_bfloat16_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
            arrayB.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute03(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayC.length(); i++) {
            BF16 val = arrayC.array(i);
            float fa = BF16.bfloat162float(arrayA.array(i));
            float fb = BF16.bfloat162float(arrayB.array(i));
            try {
                HATAsserts.assertEquals((fa + fb + fb), BF16.bfloat162float(val), 0.01f);
            } catch (HATAssertionError hae) {
                throw new HATExpectedPrecisionError(hae.getMessage());
            }
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
            arrayB.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> {
            TestBFloat16Type.compute04(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            BF16 gotResult = arrayC.array(i);

            // CPU Computation
            BF16 ha = arrayA.array(i);
            BF16 hb = arrayB.array(i);
            BF16 r1 = BF16.mul(ha, hb);
            BF16 r2 = BF16.div(ha, hb);
            BF16 r3 = BF16.sub(ha, hb);
            BF16 r4 = BF16.add(r1, r2);
            BF16 r5 = BF16.add(r4, r3);

            try {
                HATAsserts.assertEquals(BF16.bfloat162float(r5), BF16.bfloat162float(gotResult), 0.01f);
            } catch (HATAssertionError hatAssertionError) {
                throw new HATExpectedPrecisionError(hatAssertionError.getMessage());
            }
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_05() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestBFloat16Type.compute05(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            BF16 val = arrayA.array(i);
            HATAsserts.assertEquals(2.1f, BF16.bfloat162float(val), 0.01f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_06() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 512;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestBFloat16Type.compute06(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            BF16 val = arrayA.array(i);
            try {
                HATAsserts.assertEquals(i, BF16.bfloat162float(val), 0.01f);
            } catch (HATAssertionError hatAssertionError) {
                throw new HATExpectedPrecisionError(hatAssertionError.getMessage());
            }

        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_07() {
        // Test CPU Implementation of BF16
        BF16 a = BF16.of(2.5f);
        BF16 b = BF16.of(3.5f);
        BF16 c = BF16.add(a, b);
        HATAsserts.assertEquals((2.5f + 3.5f), BF16.bfloat162float(c), 0.01f);

        BF16 d = BF16.sub(a, b);
        HATAsserts.assertEquals((2.5f - 3.5f), BF16.bfloat162float(d), 0.01f);

        BF16 e = BF16.mul(a, b);
        HATAsserts.assertEquals((2.5f * 3.5f), BF16.bfloat162float(e), 0.01f);

        BF16 f = BF16.div(a, b);
        HATAsserts.assertEquals((2.5f / 3.5f), BF16.bfloat162float(f), 0.01f);
    }

    @HatTest
    @Reflect
    public void test_bfloat16_08() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestBFloat16Type.compute08(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            BF16 val = arrayA.array(i);
            HATAsserts.assertEquals(i, BF16.bfloat162float(val), 0.01f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_09() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute09(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 val = arrayB.array(i);
            HATAsserts.assertEquals(BF16.bfloat162float(arrayA.array(i)), BF16.bfloat162float(val), 0.01f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_10() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);

        accelerator.compute(computeContext -> TestBFloat16Type.compute10(computeContext, arrayA));

        for (int i = 0; i < arrayA.length(); i++) {
            BF16 val = arrayA.array(i);
            HATAsserts.assertEquals(1.1f, BF16.bfloat162float(val), 0.01f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_11() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute11(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_12() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
            arrayB.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute12(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 result = arrayC.array(i);
            HATAsserts.assertEquals(BF16.bfloat162float(BF16.add(arrayA.array(i), arrayB.array(i))), BF16.bfloat162float(result), 0.01f);
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_13() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
            arrayB.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute13(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 result = arrayC.array(i);
            try {
                HATAsserts.assertEquals(BF16.bfloat162float(arrayA.array(i)), BF16.bfloat162float(result), 0.01f);
             } catch (HATAssertionError hatAssertionError) {
                throw new HATExpectedPrecisionError(hatAssertionError.getMessage());
            }
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_14() {
        // Testing mixed types
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute14(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 result = arrayB.array(i);
            try {
                HATAsserts.assertEquals(BF16.bfloat162float(arrayA.array(i)) + 32.1f, BF16.bfloat162float(result), 0.1f);
            } catch (HATAssertionError hatAssertionError) {
                throw new HATExpectedPrecisionError(hatAssertionError.getMessage());
            }
        }
    }

    @HatTest
    @Reflect
    public void test_bfloat16_15() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestBFloat16Type.compute15(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            BF16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    // Check accumulators
    @HatTest
    @Reflect
    public void test_bfloat16_16() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1;
        BF16Array arrayA = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        arrayA.array(0).value(BF16.float2bfloat16(10).value());

        accelerator.compute(computeContext -> TestBFloat16Type.compute16(computeContext, arrayA));

        BF16 val = arrayA.array(0);
        HATAsserts.assertEquals(40.0f, BF16.bfloat162float(val), 0.01f);
    }

    // Check accumulators in private memory
    @HatTest
    @Reflect
    public void test_bfloat16_17() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1;
        BF16Array arrayA = BF16Array.create(accelerator, size);

        Random r = new Random(73);
        arrayA.array(0).value(BF16.float2bfloat16(10).value());

        accelerator.compute(computeContext -> TestBFloat16Type.compute17(computeContext, arrayA));

        BF16 val = arrayA.array(0);
        HATAsserts.assertEquals(20.0f, BF16.bfloat162float(val), 0.01f);
    }

}
