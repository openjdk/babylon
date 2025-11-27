package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.BF16;
import hat.buffer.BF16Array;
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
    public static void compute01(@RO ComputeContext computeContext, @RO BF16Array a, @WO BF16Array b) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()), kernelContext -> TestBFloat16Type.kenrel_copy(kernelContext, a, b));
    }

    @Reflect
    public static void compute02(@RO ComputeContext computeContext, @RO BF16Array a, @RO BF16Array b, @WO BF16Array c) {
        computeContext.dispatchKernel(NDRange.of1D(a.length()),
                kernelContext -> TestBFloat16Type.bf16_02(kernelContext, a, b, c));
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
}
