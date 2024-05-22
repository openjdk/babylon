package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;

import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class ForTests {

    public static class Compute {

        @CodeReflection
        static void breakAndContinue(KernelContext kc, F32Array a) {
            long i = kc.x;
            long size = kc.maxX;
            outer:
            for (long j = 0; j < size; j++) {
                float sum = 0f;
                for (long k = 0; k < size; k++) {
                    if (k == 6) {
                        sum += 3;
                        break outer;
                    } else if (k == 4) {
                        sum += 2;
                        continue outer;
                    } else if (k == 0) {
                        sum += 0;
                    } else {
                        sum += 4;
                    }
                    sum++;
                }
            }
        }

        @CodeReflection
        static void counted(KernelContext kc, F32Array a) {
            for (int j = 0; j < a.length(); j = j + 1) {
                float sum = j;
            }
        }

        @CodeReflection
        static void tuple(KernelContext kc, F32Array a) {
            for (int j = 1, i = 2, k = 3; j < a.length(); k += 1, i += 2, j += 3) {
                float sum = k + i + j;
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a) {
            computeContext.dispatchKernel(a.length(), (kc) -> counted(kc, a));
            computeContext.dispatchKernel(a.length(), (kc) -> tuple(kc, a));
            computeContext.dispatchKernel(a.length(), (kc) -> breakAndContinue(kc, a));
        }

    }

    public static void main(String[] args) {

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),
              //  Backend.JAVA_MULTITHREADED
                (backend) -> backend.getClass().getSimpleName().startsWith("OpenCL")
        );
        var a = F32Array.create(accelerator, 100);
        accelerator.compute(
                cc -> Compute.compute(cc, a)
        );

    }

}
