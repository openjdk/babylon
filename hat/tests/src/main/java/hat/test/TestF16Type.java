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
import hat.LocalMesh1D;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.F16;
import hat.buffer.F16Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestF16Type {

    @CodeReflection
//    @Kernel("""
//            HAT_KERNEL void copy01(
//                HAT_GLOBAL_MEM KernelContext_t* kernelContext,
//                HAT_GLOBAL_MEM F16Array_t* a,
//                HAT_GLOBAL_MEM F16Array_t* b
//            ){
//                if(HAT_GIX<HAT_GSX){
//                    HAT_GLOBAL_MEM F16_t* ha = &a->array[(long)HAT_GIX];
//                    HAT_GLOBAL_MEM F16_t* hb = &b->array[(long)HAT_GIX];
//                    (&b->array[(long)HAT_GIX])->value=ha->value;
//                }
//                return;
//            }
//            """)
    public static void copy01(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        if (kernelContext.gix < kernelContext.gsx) {
            F16 ha = a.array(kernelContext.gix);
            F16 hb = b.array(kernelContext.gix);
            // The following expression does not work
            b.array(kernelContext.gix).value(ha.value());
            //hb.value(ha.value());
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

    private interface MyLocalArray extends Buffer {
        void array(long index, F16 value);
        F16 array(long index);
        Schema<MyLocalArray> schema = Schema.of(MyLocalArray.class,
                        arr -> arr.array("array", 1024));

        static MyLocalArray create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static MyLocalArray createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    @CodeReflection
    public static void f16Ops_11(@RO KernelContext kernelContext, @RO F16Array a, @RW F16Array b) {
        MyLocalArray sm = MyLocalArray.createLocal();
        if (kernelContext.gix < kernelContext.gsx) {
            int lix = kernelContext.lix;
            F16 ha = a.array(kernelContext.gix);
            // store a value into shared memory

            sm.array(lix, ha);

            kernelContext.barrier();

            // read a value from shared
            F16 hb = sm.array(lix);

            b.array(kernelContext.gix).value(hb.value());
        }
    }

    @CodeReflection
    public static void compute01(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.copy01(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute02(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_02(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute03(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_03(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute04(@RO ComputeContext computeContext, @RO F16Array a, @RO F16Array b, @RW F16Array c) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_04(kernelContext, a, b, c));
    }

    @CodeReflection
    public static void compute05(@RO ComputeContext computeContext, @RW F16Array a) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_05(kernelContext, a));
    }

    @CodeReflection
    public static void compute06(@RO ComputeContext computeContext, @RW F16Array a) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_06(kernelContext, a));
    }

    @CodeReflection
    public static void compute08(@RO ComputeContext computeContext, @RW F16Array a) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_08(kernelContext, a));
    }

    @CodeReflection
    public static void compute09(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_09(kernelContext, a, b));
    }

    @CodeReflection
    public static void compute10(@RO ComputeContext computeContext, @RW F16Array a) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_10(kernelContext, a));
    }

    @CodeReflection
    public static void compute11(@RO ComputeContext computeContext, @RO F16Array a, @RW F16Array b) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(a.length()), new LocalMesh1D(16));
        computeContext.dispatchKernel(computeRange, kernelContext -> TestF16Type.f16Ops_11(kernelContext, a, b));
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
            HatAsserts.assertEquals((float)i, F16.f16ToFloat(val), 0.001f);
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
            HatAsserts.assertEquals((fa + fb), F16.f16ToFloat(val), 0.001f);
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
            HatAsserts.assertEquals((fa + fb + fb), F16.f16ToFloat(val), 0.001f);
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

            HatAsserts.assertEquals(Float.float16ToFloat(r5.value()), Float.float16ToFloat(gotResult), 0.01f);
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
            HatAsserts.assertEquals(2.1f, Float.float16ToFloat(val), 0.01f);
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
            HatAsserts.assertEquals(i, Float.float16ToFloat(val), 0.01f);
        }
    }

    @HatTest
    public void testF16_07() {
        // Test CPU Implementation of F16
        F16 a = F16.of(2.5f);
        F16 b = F16.of(3.5f);
        F16 c = F16.add(a, b);
        HatAsserts.assertEquals((2.5f + 3.5f), Float.float16ToFloat(c.value()), 0.01f);

        F16 d = F16.sub(a, b);
        HatAsserts.assertEquals((2.5f - 3.5f), Float.float16ToFloat(d.value()), 0.01f);

        F16 e = F16.mul(a, b);
        HatAsserts.assertEquals((2.5f * 3.5f), Float.float16ToFloat(e.value()), 0.01f);

        F16 f = F16.div(a, b);
        HatAsserts.assertEquals((2.5f / 3.5f), Float.float16ToFloat(f.value()), 0.01f);
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
            HatAsserts.assertEquals(i, Float.float16ToFloat(val), 0.01f);
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
            HatAsserts.assertEquals(arrayA.array(i).value(), val.value());
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
            HatAsserts.assertEquals(1.1f, F16.f16ToFloat(val), 0.01f);
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
            HatAsserts.assertEquals(arrayA.array(i).value(), val.value());
        }
    }

}
