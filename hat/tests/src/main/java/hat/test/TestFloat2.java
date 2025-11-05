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
import hat.buffer.Buffer;
import hat.buffer.F32ArrayPadded;
import hat.buffer.Float2;
import hat.buffer.Float4;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;
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
    public static void computeGraph01(@RO ComputeContext cc, @RO F32ArrayPadded a, @RO F32ArrayPadded b, @RW F32ArrayPadded c, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(new NDRange.Global1D(size/2), new NDRange.Local1D(128));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps01(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void computeGraph02(@RO ComputeContext cc, @RW F32ArrayPadded a, @RW F32ArrayPadded b, int size) {
        // Note: we need to launch N threads / vectorWidth -> size / 4 for this example
        NDRange ndRange = NDRange.of(new NDRange.Global1D(size/2));
        cc.dispatchKernel(ndRange, kernelContext -> TestFloat2.vectorOps02(kernelContext, a, b));
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

        accelerator.compute(cc -> TestFloat2.computeGraph01(cc, arrayA, arrayB, arrayC, size));

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

        accelerator.compute(cc -> TestFloat2.computeGraph02(cc, arrayA, arrayB, size));

        for (int i = 0; i < size; i += 2) {
            HatAsserts.assertEquals((arrayA.array(i + 0) * 10.0f), arrayB.array(i + 0), 0.001f);
            HatAsserts.assertEquals((arrayA.array(i + 1)), arrayB.array(i + 1), 0.001f);
        }
    }

}

