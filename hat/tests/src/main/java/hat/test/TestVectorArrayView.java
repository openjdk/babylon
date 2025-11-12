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
import hat.buffer.*;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestVectorArrayView {

    @CodeReflection
    public static void vectorOps01(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 floatA = vA[index * 4];
            Float4 floatB = vB[index * 4];
            Float4 res = Float4.add(floatA, floatB);
            vC[index * 4] = res;
        }
    }

    @CodeReflection
    public static void vectorOps01WithFloat4s(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 vAFloat = vA[index * 4];
            Float4 vBFloat = vB[index * 4];
            vC[index * 4] = Float4.add(vAFloat, vBFloat);
        }
    }

    @CodeReflection
    public static void vectorOps01WithSeparateAdd(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 res = Float4.add(vA[index * 4], vB[index * 4]);
            vC[index * 4] = res;
        }
    }

    @CodeReflection
    public static void vectorOps02(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4.MutableImpl[] vArr = a.float4ArrayView();
            Float4.MutableImpl[] bArr = b.float4ArrayView();
            Float4.MutableImpl vA = vArr[index * 4];
            float scaleX = vA.x() * 10.0f;
            vA.x(scaleX);
            bArr[index * 4] = vA;
        }
    }

    @CodeReflection
    public static void vectorOps03(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4.MutableImpl[] vA = a.float4ArrayView();
            Float4.MutableImpl[] vB = b.float4ArrayView();
            Float4.MutableImpl vAFloat = vA[index * 4];
            float scaleX = vAFloat.x() * 10.0f;
            float scaleY = vAFloat.y() * 20.0f;
            float scaleZ = vAFloat.z() * 30.0f;
            float scaleW = vAFloat.w() * 40.0f;
            vAFloat.x(scaleX);
            vAFloat.y(scaleY);
            vAFloat.z(scaleZ);
            vAFloat.w(scaleW);
            vB[index * 4] = vAFloat;
        }
    }

    @CodeReflection
    public static void vectorOps04(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4.MutableImpl[] vA = a.float4ArrayView();
            Float4.MutableImpl[] vB = b.float4ArrayView();
            Float4.MutableImpl vAFloat = vA[index * 4];
            vAFloat.x(vAFloat.x() * 10.0f);
            vAFloat.y(vAFloat.y() * 20.0f);
            vAFloat.z(vAFloat.z() * 30.0f);
            vAFloat.w(vAFloat.w() * 40.0f);
            vB[index * 4] = vAFloat;
        }
    }

    @CodeReflection
    public static void vectorOps05(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 floatA = vA[index * 4];
            Float4 floatB = vB[index * 4];
            Float4 temp = floatA.add(floatB);
            Float4 res = temp.add(floatB);
            vC[index * 4] = res;
        }
    }

    @CodeReflection
    public static void vectorOps06(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 floatA = vA[index * 4];
            Float4 floatB = vB[index * 4];
            Float4 vD = Float4.sub(floatA, floatB);
            Float4 vE = Float4.sub(floatA, floatB);
            vC[index * 4] = vE;
        }
    }

    @CodeReflection
    public static void vectorOps07(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vAArray = a.float4ArrayView();
            Float4[] vBArray = b.float4ArrayView();
            Float4[] vCArray = c.float4ArrayView();

            Float4 vA = vAArray[index * 4];
            Float4 vB = vBArray[index * 4];
            Float4 vC = vA.add(vB);
            Float4 vD = vC.sub(vB);
            vCArray[index * 4] = vD;
        }
    }

    @CodeReflection
    public static void vectorOps08(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vAArray = a.float4ArrayView();
            Float4[] vBArray = b.float4ArrayView();
            Float4[] vCArray = c.float4ArrayView();

            Float4 vA = vAArray[index * 4];
            Float4 vB = vBArray[index * 4];
            Float4 vC = vA.add(vB);
            Float4 vD = vC.mul(vA);
            Float4 vE = vD.div(vB);
            vCArray[index * 4] = vE;
        }
    }

    @CodeReflection
    public static void vectorOps09(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        // Checking composition
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4[] vAArray = a.float4ArrayView();
            Float4[] vBArray = b.float4ArrayView();
            Float4[] vCArray = c.float4ArrayView();

            Float4 vA = vAArray[index * 4];
            Float4 vB = vBArray[index * 4];
            Float4 temp = vA.mul(vB);
            Float4 vC = vA.add(temp);
            vCArray[index * 4] = vC;
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
        default Float4.MutableImpl[] float4LocalArrayView() {
            return null;
        }
    }

    @CodeReflection
    public static void vectorOps10(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedMemory sm = SharedMemory.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;

            Float4[] aArr = a.float4ArrayView();
            Float4[] bArr = b.float4ArrayView();
            Float4[] smArr = sm.float4LocalArrayView();

            Float4 vA = aArr[index * 4];
            smArr[lix * 4] = vA;
            kernelContext.barrier();
            Float4 r = smArr[lix * 4];
            bArr[index * 4] = r;
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
        default Float4[] float4PrivateArrayView() {
            return null;
        }
    }

    @CodeReflection
    public static void vectorOps11(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        PrivateMemory pm = PrivateMemory.createPrivate();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] aArr = a.float4ArrayView();
            Float4[] bArr = b.float4ArrayView();
            Float4[] pmArr = pm.float4PrivateArrayView();

            Float4 vA = aArr[index * 4];
            pmArr[0] = vA;
            kernelContext.barrier();
            Float4 r = pmArr[0];
            bArr[index * 4] = r;
        }
    }

    @CodeReflection
    public static void vectorOps12(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        SharedMemory sm = SharedMemory.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            int lix = kernelContext.lix;
            Float4.MutableImpl[] aArr = a.float4ArrayView();
            Float4.MutableImpl[] bArr = b.float4ArrayView();
            Float4.MutableImpl[] smArr = sm.float4LocalArrayView();

            Float4.MutableImpl vA = aArr[index * 4];
            Float4.MutableImpl smVector = smArr[lix * 4];
            smVector.x(vA.x());
            smVector.y(vA.y());
            smVector.z(vA.z());
            smVector.w(vA.w());
            smArr[lix * 4] = smVector;
            kernelContext.barrier();
            Float4.MutableImpl r = smArr[lix * 4];
            bArr[index * 4] = r;
        }
    }

    @CodeReflection
    public static void computeGraph01(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4), NDRange.Local1D.of(128));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps01(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph01WithFloat4s(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4), NDRange.Local1D.of(128));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps01WithFloat4s(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph01WithSeparateAdd(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4), NDRange.Local1D.of(128));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps01WithSeparateAdd(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph02(@RO ComputeContext cc, @RW F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps02(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph03(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps03(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph04(@RO ComputeContext cc, @RO F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps04(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph05(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps05(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph06(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps06(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph07(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps07(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph08(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps08(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph09(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps09(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph10(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps10(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph11(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps11(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph12(@RO ComputeContext cc, @RO F32ArrayPadded a,  @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size/4));
        cc.dispatchKernel(ndRange, kernelContext -> vectorOps12(kernelContext, a, b));
    }

    @HatTest
    public void TestVectorArrayView01() {
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

        accelerator.compute(cc -> computeGraph01(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }

    }

    // @HatTest
    // public void TestVectorArrayView01WithFloat4s() {
    //     final int size = 1024;
    //     var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
    //     var arrayA = F32ArrayPadded.create(accelerator, size);
    //     var arrayB = F32ArrayPadded.create(accelerator, size);
    //     var arrayC = F32ArrayPadded.create(accelerator, size);
    //
    //     Random r = new Random(19);
    //     for (int i = 0; i < size; i++) {
    //         arrayA.array(i, r.nextFloat());
    //         arrayB.array(i, r.nextFloat());
    //     }
    //
    //     accelerator.compute(cc -> computeGraph01WithFloat4s(cc, arrayA, arrayB, arrayC, size));
    //
    //     for (int i = 0; i < size; i++) {
    //         HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
    //     }
    //
    // }
    //
    // @HatTest
    // public void TestVectorArrayView01WithSeparateAdd() {
    //     final int size = 1024;
    //     var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
    //     var arrayA = F32ArrayPadded.create(accelerator, size);
    //     var arrayB = F32ArrayPadded.create(accelerator, size);
    //     var arrayC = F32ArrayPadded.create(accelerator, size);
    //
    //     Random r = new Random(19);
    //     for (int i = 0; i < size; i++) {
    //         arrayA.array(i, r.nextFloat());
    //         arrayB.array(i, r.nextFloat());
    //     }
    //
    //     accelerator.compute(cc -> computeGraph01WithSeparateAdd(cc, arrayA, arrayB, arrayC, size));
    //
    //     for (int i = 0; i < size; i++) {
    //         HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
    //     }
    //
    // }

    @HatTest
    public void TestVectorArrayView02() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph02(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1)), arrayB.array(i + 1), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 2)), arrayB.array(i + 2), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 3)), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView03() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph03(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 2) * 30.0f), arrayB.array(i + 2), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 3) * 40.0f), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView04() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph04(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 4) {
            HATAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 1) * 20.0f), arrayB.array(i + 1), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 2) * 30.0f), arrayB.array(i + 2), 0.001f);
            HATAsserts.assertEquals((arrayA.array(i + 3) * 40.0f), arrayB.array(i + 3), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView05() {
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

        accelerator.compute(cc -> computeGraph05(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals((arrayA.array(i) + arrayB.array(i) + arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView06() {
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

        accelerator.compute(cc -> computeGraph06(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals((arrayA.array(i) - arrayB.array(i)), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView07() {
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

        accelerator.compute(cc -> computeGraph07(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView08() {
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

        accelerator.compute(cc -> computeGraph08(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            float val = (((arrayA.array(i) + arrayB.array(i)) * arrayA.array(i)) / arrayB.array(i));
            HATAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView09() {
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

        accelerator.compute(cc -> computeGraph09(cc, arrayA, arrayB, arrayC, size));

        for (int i = 0; i < size; i ++) {
            float val = (arrayA.array(i) + (arrayB.array(i)) * arrayA.array(i));
            HATAsserts.assertEquals(val, arrayC.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView10() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph10(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView11() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph11(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }

    @HatTest
    public void TestVectorArrayView12() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32ArrayPadded.create(accelerator, size);
        var arrayB = F32ArrayPadded.create(accelerator, size);

        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            arrayA.array(i, r.nextFloat());
            arrayB.array(i, r.nextFloat());
        }

        accelerator.compute(cc -> computeGraph12(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i ++) {
            HATAsserts.assertEquals(arrayA.array(i), arrayB.array(i), 0.001f);
        }
    }
}
