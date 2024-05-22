package experiments.spirv;

import hat.Accelerator;
import hat.KernelContext;
import hat.ComputeContext;
import hat.backend.Backend;
import hat.buffer.F32Array;

import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class GetBackend {

    public static void getSpirvBakend() {
        Backend spirvBackend = Backend.getBackend((backend) -> {
            return backend.getClass().getSimpleName().equals("SpirvBackend");
        });
    }

    public static void getSpirvAccelerator() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), (backend) -> {
            return backend.getClass().getSimpleName().equals("SpirvBackend");
        });
    }

    public static class MatrixMultiply {

        /*
          Original loop was
          for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    float sum = 0f;
                    for (int k = 0; k < size; k++) {
                        sum += a[i * size + k] * b[k * size + j];
                        sum += a[i * size + k] * b[k * size + j];
                    }
                    c[i * size + j] = sum;
                }
            }

           Converted to hat kernel

                for (int j = 0; j < kid.max; j++) {
                    float sum = 0f;
                    for (int k = 0; k < kid.max; k++) {
                        sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                        sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                    }
                    c[kid.x * kid.max + j] = sum;
                }

           We don't allow heap array access. So we use F32Array iface mapped segment

            Converted to hat kernel

                for (int j = 0; j < kid.max; j++) {
                    float sum = 0f;
                    for (int k = 0; k < kid.max; k++) {
                        //sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                        sum += a.array(kid.x * kid.max + k)*b.array(k * kid.max + j]);
                        //sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                        sum += a.array(kid.x * kid.max + k) * b.array(k * kid.max + j);
                    }
                    //c[kid.x * kid.max + j] = sum;
                    c.array(kid.x * kid.max + j, sum);
                }

         */
        @CodeReflection
        static void kernel(KernelContext kid, F32Array a, F32Array b, F32Array c) {
            for (int j = 0; j < kid.maxX; j++) {
                float sum = 0f;
                for (int k = 0; k < kid.maxX; k++) {
                    //sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                    sum += a.array(kid.x * kid.maxX + k)*b.array(k * kid.maxX + j);
                    //sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                    sum += a.array(kid.x * kid.maxX + k) * b.array(k * kid.maxX + j);
                }
                //c[kid.x * kid.max + j] = sum;
                c.array(kid.x * kid.maxX + j, sum);
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a, F32Array b, F32Array c, int size) {
            computeContext.dispatchKernel(size*size, kc->MatrixMultiply.kernel(kc,  a, b, c));
        }

    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), (backend) ->
            backend.getClass().getSimpleName().startsWith("Spirv")
        );
        var a =F32Array.create(accelerator, 100);
        var b =F32Array.create(accelerator, 100);
        var c =F32Array.create(accelerator, 100);
        accelerator.compute(cc->MatrixMultiply.compute(cc, a,b,c,100));
    }

}
