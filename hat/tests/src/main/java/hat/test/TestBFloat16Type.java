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
import hat.buffer.BF16;
import hat.buffer.BF16Array;
import hat.buffer.F16;
import hat.buffer.F16Array;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.ifacemapper.MappableIface.*;

public class TestBFloat16Type {

    @Reflect
    public static void kenrel_copy(@RO KernelContext kernelContext, @RO BF16Array a, @WO BF16Array b) {
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
    public static void compute01(@RO ComputeContext computeContext, @RO BF16Array a, @WO BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.kenrel_copy(kernelContext, a, b));
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

    @HatTest
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
    public void test_bfloat16_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
        BF16Array arrayA = BF16Array.create(accelerator, size);
        BF16Array arrayB = BF16Array.create(accelerator, size);
        BF16Array arrayC = BF16Array.create(accelerator, size);

        Random random = new Random();
        for (int i = 0; i < arrayA.length(); i++) {
            arrayA.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
            arrayB.array(i).value(BF16.float2bfloat16(random.nextFloat()).value());
        }

        accelerator.compute(computeContext -> {
            TestBFloat16Type.compute03(computeContext, arrayA, arrayB, arrayC);
        });

        for (int i = 0; i < arrayC.length(); i++) {
            BF16 val = arrayC.array(i);
            float fa = BF16.bfloat162float(arrayA.array(i));
            float fb = BF16.bfloat162float(arrayB.array(i));
            HATAsserts.assertEquals((fa + fb + fb), BF16.bfloat162float(val), 0.01f);
        }
    }

    @HatTest
    public void test_bfloat16_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 16;
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

            HATAsserts.assertEquals(BF16.bfloat162float(r5), BF16.bfloat162float(gotResult), 0.01f);
        }
    }
}
