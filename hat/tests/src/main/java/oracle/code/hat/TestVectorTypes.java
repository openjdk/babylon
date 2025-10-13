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

import hat.*;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.buffer.Float4;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestVectorTypes {

    @CodeReflection
    public static void vectorOps01(@RO KernelContext kernelContext, @RO F32Array a, @RO F32Array b, @RW F32Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = Float4.add(vA, vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps02(@RO KernelContext kernelContext, @RO F32Array a, @RW F32Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            float scaleX = vA.x() * 10.0f;
            vA.x(scaleX);
            b.storeFloat4View(vA, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps03(@RO KernelContext kernelContext, @RO F32Array a, @RW F32Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            float scaleX = vA.x() * 10.0f;
            float scaleY = vA.y() * 20.0f;
            float scaleZ = vA.z() * 30.0f;
            float scaleW = vA.w() * 40.0f;
            vA.x(scaleX);
            vA.y(scaleY);
            vA.z(scaleZ);
            vA.w(scaleW);
            b.storeFloat4View(vA, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps04(@RO KernelContext kernelContext, @RO F32Array a, @RW F32Array b) {
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
    public static void vectorOps05(@RO KernelContext kernelContext, @RO F32Array a, @RO F32Array b, @RW F32Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).add(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps06(@RO KernelContext kernelContext, @RO F32Array a, @RO F32Array b, @RW F32Array c) {
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
    public static void vectorOps07(@RO KernelContext kernelContext, @RO F32Array a, @RO F32Array b, @RW F32Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).sub(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void vectorOps08(@RO KernelContext kernelContext, @RO F32Array a, @RO F32Array b, @RW F32Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;
            Float4 vA = a.float4View(index * 4);
            Float4 vB = b.float4View(index * 4);
            Float4 vC = vA.add(vB).mul(vA).div(vB);
            c.storeFloat4View(vC, index * 4);
        }
    }

    @CodeReflection
    public static void computeGraph01(@RO ComputeContext cc, @RO F32Array a, @RO F32Array b, @RW F32Array c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4), new LocalMesh1D(128));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps01(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph02(@RO ComputeContext cc, @RW F32Array a, @RW F32Array b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps02(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph03(@RO ComputeContext cc, @RO F32Array a, @RW F32Array b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps03(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph04(@RO ComputeContext cc, @RO F32Array a, @RW F32Array b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps04(kernelContext, a, b));
    }

    @CodeReflection
    public static void computeGraph05(@RO ComputeContext cc, @RO F32Array a, @RO F32Array b, @RW F32Array c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps05(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph06(@RO ComputeContext cc, @RO F32Array a, @RO F32Array b, @RW F32Array c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps06(kernelContext, a, b, c));
    }


    @CodeReflection
    public static void computeGraph07(@RO ComputeContext cc, @RO F32Array a, @RO F32Array b, @RW F32Array c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps07(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph08(@RO ComputeContext cc, @RO F32Array a, @RO F32Array b, @RW F32Array c,  int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size/4));
        cc.dispatchKernel(computeRange, kernelContext -> TestVectorTypes.vectorOps08(kernelContext, a, b, c));
    }

    @HatTest
    public void testVectorTypes01() {
        final int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

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
        var arrayA = F32Array.create(accelerator, size);
        var arrayB = F32Array.create(accelerator, size);
        var arrayC = F32Array.create(accelerator, size);

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


}
