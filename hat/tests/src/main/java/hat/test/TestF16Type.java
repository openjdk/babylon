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
import hat.buffer.F16;
import hat.buffer.F16Array;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;
import hat.test.engine.HATExpectedFailureException;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestF16Type {

    @CodeReflection
    public static void copy01(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            b.array(kernelContext.gix).value(ha.value());
        }
    }

    @CodeReflection
    public static void f16Ops_02(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);

            F16 result = F16.add(ha, hb);
            F16 hC = c.array(kernelContext.gix);
            hC.value(result.value());
        }
    }

    @CodeReflection
    public static void f16Ops_03(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);

            F16 result = F16.add(ha, F16.add(hb, hb));
            F16 hC = c.array(kernelContext.gix);
            hC.value(result.value());
        }
    }

    @CodeReflection
    public static void f16Ops_04(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);

            F16 r1 = F16.mul(ha, hb);
            F16 r2 = F16.div(ha, hb);
            F16 r3 = F16.sub(ha, hb);
            F16 r4 = F16.add(r1, r2);
            F16 r5 = F16.add(r4, r3);
            F16 hC = c.array(kernelContext.gix);
            hC.value(r5.value());
        }
    }

    @CodeReflection
    public static void f16Ops_05(@RO KernelContext kernelContext, @RW F16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 initVal = F16.of( 2.1f);
            ha.value(initVal.value());
        }
    }

    @CodeReflection
    public static void f16Ops_06(@RO KernelContext kernelContext, @RW F16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 initVal = F16.of( kernelContext.gix);
            F16 ha = a.array(kernelContext.gix);
            ha.value(initVal.value());
        }
    }

    @CodeReflection
    public static void f16Ops_08(@RO KernelContext kernelContext, @RW F16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 initVal = F16.floatToF16(kernelContext.gix);
            F16 ha = a.array(kernelContext.gix);
            ha.value(initVal.value());
        }
    }

    @CodeReflection
    public static void f16Ops_09(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            float f = F16.f16ToFloat(ha);
            F16 result = F16.floatToF16(f);
            F16 hb = b.array(kernelContext.gix);
            hb.value(result.value());
        }
    }

    @CodeReflection
    public static void f16Ops_10(@RO KernelContext kernelContext, @RO F16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 f16 = F16.of(1.1f);
            float f = F16.f16ToFloat(f16);
            F16 result = F16.floatToF16(f);
            ha.value(result.value());
        }
    }

    public interface DeviceLocalArray extends DeviceType {
        F16 array(int index);
        //void array(int index, F16 value);

        DeviceSchema<DeviceLocalArray> schema = DeviceSchema.of(DeviceLocalArray.class,
builder -> builder.withArray("array", 1024)
                        .withDeps(F16.class, half -> half.withField("value")));

        static DeviceLocalArray create(Accelerator accelerator) {
            return null;
        }

        static DeviceLocalArray createLocal() {
            return null;
        }
    }

    @CodeReflection
    public static void f16Ops_11(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        DeviceLocalArray sm = DeviceLocalArray.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            F16 ha = a.array(kernelContext.gix);

            // store into local memory
            sm.array(lix).value(ha.value());
            kernelContext.barrier();

            F16 hb = sm.array(lix);
            b.array(kernelContext.gix).value(hb.value());
        }
    }

    @CodeReflection
    public static void f16Ops_12(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b,  @RW F16Array c) {
        // Test the fluent API style
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);
            F16 result = ha.add(hb);
            c.array(kernelContext.gix).value(result.value());
        }
    }

    @CodeReflection
    public static void f16Ops_13(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b,  @RW F16Array c) {
        // Test the fluent API style
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);
            F16 result = ha.add(hb).sub(hb).mul(ha).div(ha);
            c.array(kernelContext.gix).value(result.value());
        }
    }

    @CodeReflection
    public static void f16Ops_14(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        // Testing mixed float types
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            float myFloat = 32.1f;
            F16 result = F16.add(myFloat, ha);
            b.array(kernelContext.gix).value(result.value());
        }
    }

    interface DevicePrivateArray extends DeviceType {
        F16 array(int index);
        //void array(int index, F16 value);

        DeviceSchema<DevicePrivateArray> schema = DeviceSchema.of(DevicePrivateArray.class,
                builder -> builder.withArray("array", 1024)
                        .withDeps(F16.class, half -> half.withField("value")));

        static DevicePrivateArray create(Accelerator accelerator) {
            return null;
        }

        static DevicePrivateArray createPrivate() {
            return null;
        }
    }

    @CodeReflection
    public static void f16Ops_15(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        DevicePrivateArray privateArray = DevicePrivateArray.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            F16 ha = a.array(kernelContext.gix);

            // store into the private object
            privateArray.array(lix).value(ha.value());

            F16 hb = privateArray.array(lix);
            b.array(kernelContext.gix).value(hb.value());
        }
    }

    interface DevicePrivateArray2 extends DeviceType {
        F16 array(int index);
        void array(int index, F16 value);

        DeviceSchema<DevicePrivateArray2> schema = DeviceSchema.of(DevicePrivateArray2.class,
                builder -> builder.withArray("array", 1024)
                        .withDeps(F16.class, half -> half.withField("value")));

        static DevicePrivateArray2 create(Accelerator accelerator) {
            return null;
        }

        static DevicePrivateArray2 createPrivate() {
            return null;
        }
    }

    @CodeReflection
    public static void f16Ops_16(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        DevicePrivateArray2 privateArray = DevicePrivateArray2.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            F16 ha = a.array(kernelContext.gix);

            // This is expected to fail on the GPU due to the assigment of different types.
            // ha is a typed F16Impl, which is a subtype of F16.
            // While in Java, this is correct, because F16Impl is an implementation of F16,
            // the GPU code is not aware of this inheritance, then we end up assigning values
            // from different types.
            privateArray.array(lix, ha);

            F16 hb = privateArray.array(lix);
            b.array(kernelContext.gix).value(hb.value());
        }
    }

    @CodeReflection
    public static void f16Ops_17(@RO KernelContext kernelContext, @RW F16Array a) {
        F16 ha = a.array(0);
        F16 hre = F16.add(ha, ha);
        hre = F16.add(hre, hre);
        a.array(0).value(hre.value());
    }

    @CodeReflection
    public static void f16Ops_18(@RO KernelContext kernelContext, @RW F16Array a) {

        F16 ha = a.array(0);
        DevicePrivateArray2 privateArray = DevicePrivateArray2.createPrivate();
        privateArray.array(0).value(ha.value());

        // Obtain the value from private memory
        F16 acc = privateArray.array(0);

        // compute
        acc = F16.add(acc, acc);

        // store the result
        a.array(0).value(acc.value());
    }

    @CodeReflection
    public static void compute01(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.copy01(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute02(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_02(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute03(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_03(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute04(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_04(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute05(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_05(kernelContext, a));
    }

    @CodeReflection
    public static void compute06(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_06(kernelContext, a));
    }

    @CodeReflection
    public static void compute08(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_08(kernelContext, a));
    }

    @CodeReflection
    public static void compute09(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_09(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute10(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_10(kernelContext, a));
    }

    @CodeReflection
    public static void compute11(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()), NDRange.Local1D.of(16));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_11(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute12(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_12(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute13(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_13(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute14(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_14(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute15(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()), NDRange.Local1D.of(16));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_15(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute16(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(a.length()), NDRange.Local1D.of(16));
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_16(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute17(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(1);
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_17(kernelContext, a));
    }

    @CodeReflection
    public static void compute18(@RO ComputeContext computeContext, @RW F16Array a) {
        NDRange ndRange = NDRange.of(1);
        computeContext.dispatchKernel(ndRange, kernelContext -> TestF16Type.f16Ops_18(kernelContext, a));
    }

    @HatTest
    public void testF16_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(i).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute01(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            HATAsserts.assertEquals((float)i, F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    public void testF16_02() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute02(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayC.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            HATAsserts.assertEquals((fa + fb), F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    public void testF16_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute03(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            F16 val = arrayC.array(i);
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            HATAsserts.assertEquals((fa + fb + fb), F16.f16ToFloat(val), 0.001f);
        }
    }

    @HatTest
    public void testF16_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(random.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute04(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            short gotResult = arrayC.array(i).value();

            // CPU Computation
            F16 ha = arrayA.array(i);
            F16 hb = arrayB.array(i);
            F16 r1 = F16.mul(ha, hb);
            F16 r2 = F16.div(ha, hb);
            F16 r3 = F16.sub(ha, hb);
            F16 r4 = F16.add(r1, r2);
            F16 r5 = F16.add(r4, r3);

            HATAsserts.assertEquals(Float.float16ToFloat(r5.value()), Float.float16ToFloat(gotResult), 0.01f);
        }
    }

    @HatTest
    public void testF16_05() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute05(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            short val = arrayA.array(i).value();
            HATAsserts.assertEquals(2.1f, Float.float16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_06() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute06(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            short val = arrayA.array(i).value();
            HATAsserts.assertEquals(i, Float.float16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_07() {
        // Test CPU Implementation of F16
        F16 a = F16.of(2.5f);
        F16 b = F16.of(3.5f);
        F16 c = F16.add(a, b);
        HATAsserts.assertEquals((2.5f + 3.5f), Float.float16ToFloat(c.value()), 0.01f);

        F16 d = F16.sub(a, b);
        HATAsserts.assertEquals((2.5f - 3.5f), Float.float16ToFloat(d.value()), 0.01f);

        F16 e = F16.mul(a, b);
        HATAsserts.assertEquals((2.5f * 3.5f), Float.float16ToFloat(e.value()), 0.01f);

        F16 f = F16.div(a, b);
        HATAsserts.assertEquals((2.5f / 3.5f), Float.float16ToFloat(f.value()), 0.01f);
    }

    @HatTest
    public void testF16_08() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 256;
        F16Array arrayA = F16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(0.0f).value());
        }

        accelerator.compute(computeContext -> {
            TestF16Type.compute08(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            short val = arrayA.array(i).value();
            HATAsserts.assertEquals(i, Float.float16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_09() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute09(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    @HatTest
    public void testF16_10() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        F16Array arrayA = F16Array.create(accelerator, size);

        accelerator.compute(computeContext -> TestF16Type.compute10(computeContext, arrayA));

        for (int i = 0; i < arrayA.length(); i++) {
            F16 val = arrayA.array(i);
            HATAsserts.assertEquals(1.1f, F16.f16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_11() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute11(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    @HatTest
    public void testF16_12() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute12(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 result = arrayC.array(i);
            HATAsserts.assertEquals(F16.f16ToFloat(F16.add(arrayA.array(i), arrayB.array(i))), F16.f16ToFloat(result), 0.01f);
        }
    }

    @HatTest
    public void testF16_13() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);
        F16Array arrayC = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
            arrayB.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute13(computeContext, arrayA, arrayB, arrayC));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 result = arrayC.array(i);
            HATAsserts.assertEquals(F16.f16ToFloat(arrayA.array(i)), F16.f16ToFloat(result), 0.01f);
        }
    }

    @HatTest
    public void testF16_14() {
        // Testing mixed types
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute14(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 result = arrayB.array(i);
            HATAsserts.assertEquals(F16.f16ToFloat(arrayA.array(i)) + 32.1f, F16.f16ToFloat(result), 0.1f);
        }
    }

    @HatTest
    public void testF16_15() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(computeContext -> TestF16Type.compute15(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    //@HatTest
    public void testF16_16() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 256;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.floatToF16(r.nextFloat()).value());
        }

        try {
            accelerator.compute(computeContext -> TestF16Type.compute16(computeContext, arrayA, arrayB));
        } catch (RuntimeException e) {
            throw new HATExpectedFailureException("Incompatible types in expression `privateArray.array(lix, ha);`");
        }

        for (int i = 0; i < arrayB.length(); i++) {
            F16 val = arrayB.array(i);
            HATAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

    @HatTest
    public void testF16_17() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1;
        F16Array arrayA = F16Array.create(accelerator, size);

        Random r = new Random(73);
        arrayA.array(0).value(F16.floatToF16(10).value());

        accelerator.compute(computeContext -> TestF16Type.compute17(computeContext, arrayA));

        F16 val = arrayA.array(0);
        HATAsserts.assertEquals(40.0f, F16.f16ToFloat(val), 0.01f);
    }

    @HatTest
    public void testF16_18() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1;
        F16Array arrayA = F16Array.create(accelerator, size);

        Random r = new Random(73);
        arrayA.array(0).value(F16.floatToF16(10).value());

        accelerator.compute(computeContext -> TestF16Type.compute18(computeContext, arrayA));

        F16 val = arrayA.array(0);
        HATAsserts.assertEquals(20.0f, F16.f16ToFloat(val), 0.01f);
    }

}
