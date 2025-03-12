package tst;
import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.ffi.OpenCLBackend;
import hat.buffer.S32Array;
import static hat.ifacemapper.MappableIface.*;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

import static hat.backend.ffi.Config.*;

public class MinBufferTest {
    public static class Compute {
        @CodeReflection
        public static void inc(@RO KernelContext kc, @RW S32Array s32Array, int len) {
            if (kc.x < kc.maxX) {
                s32Array.array(kc.x, s32Array.array(kc.x) + 1);
            }
        }

        @CodeReflection
        public static void add(ComputeContext cc, @RW S32Array s32Array, int len, int n) {
            for (int i = 0; i < n; i++) {
                cc.dispatchKernel(len, kc -> inc(kc, s32Array, len));
                System.out.println(i);//s32Array.array(0));
            }
        }
    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
                new OpenCLBackend(of(
                      //  TRACE(),
                        TRACE_COPIES(),
                        GPU(),
                        MINIMIZE_COPIES()
                ))

        );
        int len = 10000000;
        int valueToAdd = 10;
        S32Array s32Array = S32Array.create(accelerator, len,i->i);
        accelerator.compute(
                cc -> Compute.add(cc, s32Array, len, valueToAdd)
        );
        // Quite an expensive way of adding 20 to each array alement
        for (int i = 0; i < 20; i++) {
            System.out.println(i + "=" + s32Array.array(i));
        }
    }
}
