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
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.buffer.F16Array.F16;

public class TestFP16Type {

    @CodeReflection
    public static void copy01(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16Array.F16 ha = a.array(kernelContext.gix);
            F16Array.F16 hb = b.array(kernelContext.gix);

            // The following expression does not work
            //b.array(kernelContext.gix).value(ha.value());

            hb.value(ha.value());
        }
    }

    @CodeReflection
    public static void fp16Ops_02(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16Array.F16 ha = a.array(kernelContext.gix);
            F16Array.F16 hb = b.array(kernelContext.gix);

            F16Array.F16 result = F16.add(ha, hb);
            F16Array.F16 hC = c.array(kernelContext.gix);
            hC.value(result.value());
        }
    }

    @CodeReflection
    public static void fp16Ops_03(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16Array.F16 ha = a.array(kernelContext.gix);
            F16Array.F16 hb = b.array(kernelContext.gix);

            F16Array.F16 result = F16.add(ha, F16.add(hb, hb));
            F16Array.F16 hC = c.array(kernelContext.gix);
            hC.value(result.value());
        }
    }

    @CodeReflection
    public static void fp16Ops_04(@RO KernelContext kernelContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16Array.F16 ha = a.array(kernelContext.gix);
            F16Array.F16 hb = b.array(kernelContext.gix);

            F16Array.F16 r1 = F16.mul(ha, hb);
            F16Array.F16 r2 = F16.div(ha, hb);
            F16Array.F16 r3 = F16.sub(ha, hb);
            F16Array.F16 r4 = F16.add(r1, r2);
            F16Array.F16 r5 = F16.add(r4, r3);
            F16Array.F16 hC = c.array(kernelContext.gix);
            hC.value(r5.value());
        }
    }

    @CodeReflection
    public static void fp16Ops_05(@RO KernelContext kernelContext, @RW F16Array a) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16Array.F16 ha = a.array(kernelContext.gix);
            F16Array.F16 initVal = F16.init( 2.1f);
            ha.value(initVal.value());
        }
    }

    @CodeReflection
    public static void compute01(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestFP16Type.copy01(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute02(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestFP16Type.fp16Ops_02(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute03(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestFP16Type.fp16Ops_03(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute04(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestFP16Type.fp16Ops_04(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute05(@RO ComputeContext computeContext, @RW F16Array a) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestFP16Type.fp16Ops_05(kernelContext, a));
    }

    @HatTest
    public void testF16_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        F16Array arrayB = F16Array.create(accelerator, size);

        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.float2half(i));
        }

        accelerator.compute(computeContext -> TestFP16Type.compute01(computeContext, arrayA, arrayB));

        for (int i = 0; i < arrayB.length(); i++) {
            short val = arrayB.array(i).value();
            HatAsserts.assertEquals((float)i, F16.half2float(val), 0.001f);
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
            arrayA.array(i).value(F16.float2half(random.nextFloat()));
            arrayB.array(i).value(F16.float2half(random.nextFloat()));
        }

        accelerator.compute(computeContext -> {
            TestFP16Type.compute02(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            short val = arrayC.array(i).value();
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            HatAsserts.assertEquals((fa + fb), F16.half2float(val), 0.001f);
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
            arrayA.array(i).value(F16.float2half(random.nextFloat()));
            arrayB.array(i).value(F16.float2half(random.nextFloat()));
        }

        accelerator.compute(computeContext -> {
            TestFP16Type.compute03(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            short val = arrayC.array(i).value();
            float fa = Float.float16ToFloat(arrayA.array(i).value());
            float fb = Float.float16ToFloat(arrayB.array(i).value());
            HatAsserts.assertEquals((fa + fb + fb), F16.half2float(val), 0.001f);
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
            arrayA.array(i).value(F16.float2half(random.nextFloat()));
            arrayB.array(i).value(F16.float2half(random.nextFloat()));
        }

        accelerator.compute(computeContext -> {
            TestFP16Type.compute04(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            short val = arrayC.array(i).value();
            F16Array.F16 ha = arrayA.array(i);
            F16Array.F16 hb = arrayB.array(i);

            float fa = Float.float16ToFloat(ha.value());
            float fb = Float.float16ToFloat(hb.value());
            float r1 = fa * fb;
            float r2 = fa / fb;
            float r3 = fa - fb;
            float r4 = r1 + r2;
            float r5 = r4 + r3;
            HatAsserts.assertEquals(r5, Float.float16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_05() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        F16Array arrayA = F16Array.create(accelerator, size);
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(F16.float2half(0.0f));
        }

        accelerator.compute(computeContext -> {
            TestFP16Type.compute05(computeContext, arrayA);
        });

        for (int i = 0; i < arrayA.length(); i++) {
            short val = arrayA.array(i).value();
            HatAsserts.assertEquals(2.1f, Float.float16ToFloat(val), 0.01f);
        }
    }

}
