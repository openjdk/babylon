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
import hat.ifacemapper.MappableIface.WO;
import hat.ifacemapper.Schema;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;
import jdk.incubator.code.CodeReflection;

import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public class TestFloatArrayView {

    @CodeReflection
    public static void vectorOps01(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            vC[index * 4] = Float4.add(vA[index * 4], vB[index * 4]);
        }
    }

    @CodeReflection
    public static void vectorOps02(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vArr = a.float4ArrayView();
            Float4[] bArr = b.float4ArrayView();
            Float4 vA = vArr[index * 4];
            float scaleX = vA.x() * 10.0f;
            vA.x(scaleX);
            bArr[index * 4] = vA;
        }
    }

    @CodeReflection
    public static void vectorOps03(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RW F32ArrayPadded b) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4 vAFloat = vA[index * 4];
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

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4 vAFloat = vA[index * 4];
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
            vC[index * 4] = vA[index * 4].add(vB[index * 4]).add(vB[index * 4]);
        }
    }

    @CodeReflection
    public static void vectorOps06(@RO KernelContext kernelContext, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c) {
        if (kernelContext.gix < kernelContext.gsx) {
            int index = kernelContext.gix;

            Float4[] vA = a.float4ArrayView();
            Float4[] vB = b.float4ArrayView();
            Float4[] vC = c.float4ArrayView();
            Float4 vD = Float4.sub(vA[index * 4], vB[index * 4]);
            vC[index * 4] = Float4.sub(vA[index * 4], vB[index * 4]);
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
}
