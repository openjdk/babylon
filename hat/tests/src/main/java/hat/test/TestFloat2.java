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
import hat.buffer.F32ArrayPadded;
import hat.buffer.Float2;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestFloat2 {

    @CodeReflection
    public static void vectorOps01(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vC = Float2.add(vA, vB);
            c.storeFloat2View(vC, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps02(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2.MutableImpl vA = a.float2View(index * 2);
            float scaleX = vA.x() * 10.0f;
            vA.x(scaleX);
            b.storeFloat2View(vA, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps03(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            // Obtain a view of the input data as a float4 and
            // store that view in private memory
            Float2 vA = a.float2View(index * 2);

            // operate with the float4
            float scaleX = vA.x() * 10.0f;
            float scaleY = vA.y() * 20.0f;

            // Create a float4 within the device code
            Float2 vResult = Float2.of(scaleX, scaleY);

            // store the float4 from private memory to global memory
            b.storeFloat2View(vResult, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps04(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2.MutableImpl vA = a.float2View(index * 2);
            vA.x(vA.x() * 10.0f);
            vA.y(vA.y() * 20.0f);
            b.storeFloat2View(vA, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps05(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vC = vA.add(vB).add(vB);
            c.storeFloat2View(vC, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps06(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vD = Float2.sub(vA, vB);
            Float2 vC = vA.sub(vB);
            c.storeFloat2View(vC, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps07(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vC = vA.add(vB).sub(vB);
            c.storeFloat2View(vC, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps08(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vC = vA.add(vB).mul(vA).div(vB);
            c.storeFloat2View(vC, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps09(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        // Checking composition
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2 vB = b.float2View(index * 2);
            Float2 vC = vA.add(vA.mul(vB));
            c.storeFloat2View(vC, index * 2);
        }
    }

    private interface SharedArray extends DeviceType {
        void array(long index, float value);
        float array(long index);
        DeviceSchema<SharedArray> schema = DeviceSchema.of(SharedArray.class,
                arr -> arr.withArray("array", 1024));
        static SharedArray create(Accelerator accelerator) {
            return null;
        }
        static SharedArray createLocal() {
            return null;
        }
        default Float2 float2View(int index) {
            return null;
        }
        default void storeFloat2View(Float2 float4, int index) {
        }
    }

    @CodeReflection
    public static void vectorOps10(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedArray sm = SharedArray.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;
            Float2 vA = a.float2View(index * 2);
            sm.storeFloat2View(vA, lix * 2);
            kernelContext.barrier();
            Float2 r = sm.float2View(lix * 2);
            b.storeFloat2View(r, index * 2);
        }
    }

    private interface PrivateMemory extends DeviceType {
        void array(long index, float value);
        float array(long index);
        DeviceSchema<PrivateMemory> schema = DeviceSchema.of(PrivateMemory.class,
                arr -> arr.withArray("array", 4));
        static PrivateMemory create(Accelerator accelerator) {
            return null;
        }
        static PrivateMemory createPrivate() {
            return null;
        }
        default Float2 float2View(int index) {
            return null;
        }
        default void storeFloat2View(Float2 float4, int index) {
        }
    }

    @CodeReflection
    public static void vectorOps11(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        PrivateMemory pm = PrivateMemory.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            pm.storeFloat2View(vA, 0);
            kernelContext.barrier();
            Float2 r = pm.float2View(0);
            b.storeFloat2View(r, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps12(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedArray sm = SharedArray.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;
            Float2 vA = a.float2View(index * 2);
            sm.array(lix * 2 + 0, vA.x());
            sm.array(lix * 2 + 1, vA.y());
            kernelContext.barrier();
            Float2 r = sm.float2View(lix * 2);
            b.storeFloat2View(r, index * 2);
        }
    }

    @CodeReflection
    public static void vectorOps14(@RO KernelContext kernelContext, @RW F32ArrayPadded a) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 vA = a.float2View(index * 2);
            Float2.MutableImpl vB = Float2.makeMutable(vA);
            vB.x(10.0f);
            a.storeFloat2View(vB, index * 2);
        }
    }


    @CodeReflection
    public static void vectorOps15(@RO KernelContext kernelContext, @RW F32ArrayPadded a) {
        // in this sample, we don't perform the vload, but rather the vstore directly
        // from a new float2.
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float2 result = Float2.of(1.0f, 2.0f);
            a.storeFloat2View(result, index * 2);
        }
    }

    @CodeReflection
    public static void computeGraph01(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2), NDRange.Local1D.of(128));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps01(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph02(@RO ComputeContext cc, @RW F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps02(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph03(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps03(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph04(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps04(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph05(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps05(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph06(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps06(kernelContext, a, b, c));
    }


    @CodeReflection
    public static void computeGraph07(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps07(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph08(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps08(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph09(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps09(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph10(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps10(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph11(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps11(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph12(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps12(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph14(@RO ComputeContext cc, @RW F32ArrayPadded a, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps14(kernelContext, a));
    }

    @CodeReflection
    public static void computeGraph15(@RO ComputeContext cc, @RW F32ArrayPadded a, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 2 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps15(kernelContext, a));
    }


    @HatTest
    public void testFloat2_01() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph01(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }

    }

    @HatTest
    public void testFloat2_02() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph02(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 2) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1)), arrayB.array(i + 1), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_03() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph03(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 2) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_04() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph04(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 2) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_05() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph05(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_06() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph06(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((arrayA.array(i) - arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_07() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph07(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_08() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph08(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            float val = (((arrayA.array(i) + arrayB.array(i)) * arrayA.array(i)) / arrayB.array(i));
            HATAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_09() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);
        var arrayC = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph09(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            float val = (arrayA.array(i) + (arrayB.array(i)) * arrayA.array(i));
            HATAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_10() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph10(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_11() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph11(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_12() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph12(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_13() {
        // Test the CPU implementation of Float4
        Float2 vA = Float2.of(1, 2);
        Float2 vB = Float2.of(3, 4);
        Float2 vC = Float2.add(vA, vB);
        Float2 expectedSum = Float2.of(
                vA.x() + vB.x(),
                vA.y() + vB.y());

        HATAsserts.assertEquals(expectedSum, vC, 0.001f);

        Float2 vD = Float2.sub(vA, vB);
        Float2 expectedSub = Float2.of(
                vA.x() - vB.x(),
                vA.y() - vB.y());
        HATAsserts.assertEquals(expectedSub, vD, 0.001f);

        Float2 vE = Float2.mul(vA, vB);
        Float2 expectedMul = Float2.of(
                vA.x() * vB.x(),
                vA.y() * vB.y());
        HATAsserts.assertEquals(expectedMul, vE, 0.001f);

        Float2 vF = Float2.div(vA, vB);
        Float2 expectedDiv = Float2.of(
                vA.x() / vB.x(),
                vA.y() / vB.y());
        HATAsserts.assertEquals(expectedDiv, vF, 0.001f);
    }

    @HatTest
    public void testFloat2_14() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph14(cc, arrayA, size));

        for (int i = 0; i < size; i += 2) {
            HATAsserts.assertEquals(10.0f, arrayA.array(i), 0.001f);
        }
    }

    @HatTest
    public void testFloat2_15() {
        final int size = 2048;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(73);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestFloat2.computeGraph15(cc, arrayA, size));

        Float2 v = Float2.of(1.0f, 2.0f);
        for (int i = 0; i < size; i += 2) {
            HATAsserts.assertEquals(v.x(), arrayA.array(i), 0.001f);
            HATAsserts.assertEquals(v.y(), arrayA.array(i + 1), 0.001f);
        }
    }
}