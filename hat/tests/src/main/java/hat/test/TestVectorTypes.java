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

import hat.*;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.F32ArrayPadded;
import hat.buffer.Float4;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestVectorTypes {

    @CodeReflection
    public static void vectorOps01(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = Float4.add(vA, vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps02(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            float scaleX = vA.x() * 10.0f;
            vA.x(scaleX);
            b.storeFloat4View(vA, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps03(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            // Obtain a view of the input data as a float4 and
            // store that view in private memory
            Float4 vA = a.float4View(index * 4);

            // operate with the float4
            float scaleX = vA.x() * 10.0f;
            float scaleY = vA.y() * 20.0f;
            float scaleZ = vA.z() * 30.0f;
            float scaleW = vA.w() * 40.0f;

            // Create a float4 within the device code
            Float4 vResult = Float4.of(scaleX, scaleY, scaleZ, scaleW);

            // store the float4 from private memory to global memory
            b.storeFloat4View(vResult, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps04(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            vA.x(vA.x() * 10.0f);
            vA.y(vA.y() * 20.0f);
            vA.z(vA.z() * 30.0f);
            vA.w(vA.w() * 40.0f);
            b.storeFloat4View(vA, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps05(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).add(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps06(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vD = Float4.sub(vA, vB);
            Float4 vC = vA.sub(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps07(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).sub(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps08(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).mul(vA).div(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps09(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        // Checking composition
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vA.mul(vB));
            c.storeFloat4View(vC, index * 4);
        }
    }

    private interface SharedMemory extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<SharedMemory> schema = Schema.of(SharedMemory.class,
                arr -> arr.array("array", 1024));
        static SharedMemory create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static SharedMemory createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
        default Float4 float4View(int index) {
            return null;
        }
        default void storeFloat4View(Float4 float4, int index) {
        }
    }

    @CodeReflection
    public static void vectorOps10(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedMemory sm = SharedMemory.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;
            Float4 vA = a.float4View(index * 4);
            sm.storeFloat4View(vA, lix * 4);
            kernelContext.barrier();
            Float4 r = sm.float4View(lix * 4);
            b.storeFloat4View(r, index * 4);
        }
    }

    private interface PrivateMemory extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<PrivateMemory> schema = Schema.of(PrivateMemory.class,
                arr -> arr.array("array", 4));
        static PrivateMemory create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static PrivateMemory createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
        default Float4 float4View(int index) {
            return null;
        }
        default void storeFloat4View(Float4 float4, int index) {
        }
    }

    @CodeReflection
    public static void vectorOps11(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        PrivateMemory pm = PrivateMemory.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            pm.storeFloat4View(vA, 0);
            kernelContext.barrier();
            Float4 r = pm.float4View(0);
            b.storeFloat4View(r, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps12(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedMemory sm = SharedMemory.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;
            Float4 vA = a.float4View(index * 4);
            sm.array(lix * 4 + 0, vA.x());
            sm.array(lix * 4 + 1, vA.y());
            sm.array(lix * 4 + 2, vA.z());
            sm.array(lix * 4 + 3, vA.w());
            kernelContext.barrier();
            Float4 r = sm.float4View(lix * 4);
            b.storeFloat4View(r, index * 4);
        }
    }

    @CodeReflection
    public static void computeGraph01(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4), new LocalMesh1D(128));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps01(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph02(@RO ComputeContext cc, @RW F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps02(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph03(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps03(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph04(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps04(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph05(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps05(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph06(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps06(kernelContext, a, b, c));
    }


    @CodeReflection
    public static void computeGraph07(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps07(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph08(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps08(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph09(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps09(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph10(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps10(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph11(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps11(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph12(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps12(kernelContext, a, b));
    }

    @HatTest
    public void testVectorTypes01() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph01(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HatAsserts.assertEquals((arrayA.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }

    }

    @HatTest
    public void testVectorTypes02() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph02(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HatAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 1)), arrayB.array(i + 1), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 2)), arrayB.array(i + 2), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 3)), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes03() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph03(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HatAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 2) * 30.0f), arrayB.array(i + 2), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 3) * 40.0f), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes04() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph04(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HatAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 2) * 30.0f), arrayB.array(i + 2), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 3) * 40.0f), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes05() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph05(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals((arrayA.array(i) + arrayB.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes06() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph06(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals((arrayA.array(i) - arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes07() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph07(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals(arrayA.array(i), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes08() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph08(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            float val = (((arrayA.array(i) + arrayB.array(i)) * arrayA.array(i)) / arrayB.array(i));
            HatAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes09() {
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

        accelerator.compute(cc -> TestVectorTypes.computeGraph09(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            float val = (arrayA.array(i) + (arrayB.array(i)) * arrayA.array(i));
            HatAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes10() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph10(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes11() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph11(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes12() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> TestVectorTypes.computeGraph12(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HatAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void testVectorTypes13() {
        // Test the CPU implementation of Float4
        Float4 vA = Float4.of(1, 2, 3, 4);
        Float4 vB = Float4.of(4, 3, 2, 1);
        Float4 vC = Float4.add(vA, vB);
        Float4 expectedSum = Float4.of(vA.x() + vB.x(),
                vA.y() + vB.y(),
                vA.z() + vB.z(),
                vA.w() + vB.w()
                );
        HatAsserts.assertEquals(expectedSum, vC, 0.001f);

        Float4 vD = Float4.sub(vA, vB);
        Float4 expectedSub = Float4.of(
                vA.x() - vB.x(),
                vA.y() - vB.y(),
                vA.z() - vB.z(),
                vA.w() - vB.w()
        );
        HatAsserts.assertEquals(expectedSub, vD, 0.001f);

        Float4 vE = Float4.mul(vA, vB);
        Float4 expectedMul = Float4.of(
                vA.x() * vB.x(),
                vA.y() * vB.y(),
                vA.z() * vB.z(),
                vA.w() * vB.w()
        );
        HatAsserts.assertEquals(expectedMul, vE, 0.001f);

        Float4 vF = Float4.div(vA, vB);
        Float4 expectedDiv = Float4.of(
                vA.x() / vB.x(),
                vA.y() / vB.y(),
                vA.z() / vB.z(),
                vA.w() / vB.w()
        );
        HatAsserts.assertEquals(expectedDiv, vF, 0.001f);
    }
}

