package experiments.spirv;

import hat.Accelerator;
import hat.KernelContext;
import hat.ComputeContext;
import hat.backend.Backend;
import hat.buffer.F32Array;

import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;
import java.util.stream.IntStream;

public class MatrixMultiply {

    public static class MatrixMultiplyCompute {

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

        /*
         I used this doc to map kid.XXXXX  references to SPIRV built ins.
         https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_Env.html#_built_in_variables
         */

        @CodeReflection
        static void matmul(KernelContext kc, F32Array a, F32Array b, F32Array c, int sz) {
            //long size = kc.maxX; // There is probably a SPIRV call or intrinsic or const for this
                                  //   OpenCL kc.max -> get_global_size(0)
                                  //   CUDA   kc.max -> blockDim.x*gridDim.x
                                  //   SPIRV  kc.max -> builtin GlobalSize.x?

            //long i = kc.x;       // There is probably a SPIRV call or intrinsic or const for this
                                  //   OpenCL kc.x -> get_global_id(0)
                                  //   CUDA   kc.x -> blockIdx.x*blockDim.x+threadIdx.x
                                  //   SPIRV  kc.x -> builtin GlobalInvocationId.x?
            long i= kc.x;
            long size = sz;

            for (long j = 0; j < size; j++) {
                float sum = 0f;
                for (long k = 0; k < size; k++) {
                    //sum += a[kc.x * kc.max + k] * b[k * kc.max + j];
                    sum += a.array(i * size + k) * b.array( k * size + j);
                }
                //c[kc.x * kc.max + j] = sum;
                c.array( i * size + j, sum);
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a, F32Array b, F32Array c, int size) {

                computeContext.dispatchKernel(
                        size*size,                // range is passed as int and creation internalized
                        (kid)->matmul(kid,a,b,c,size));  // kid is Kid1D has kid.x and kid.maxX

                /* A 2D dispatch - not supported yet
                computeContext.dispatchKernel(
                        size, size,                // 2D range now can be passed as two int's and creation internalized
                        (kid)->kernel(kid,a,b,c)); // kid now a Kid2D now has kid.x,kid.y,kid.maxX,

                 */

        }



    }

    public static void main(String[] args) {
        boolean newProposedAPI = true;
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        // final int size = 100; // breaks!!!!
        int size = 10;  // works
        float[] arrA = new float[size*size];
        float[] arrB = new float[size*size];
        var a = F32Array.create(accelerator, arrA);
        var b = F32Array.create(accelerator, arrB);
        var c = F32Array.create(accelerator, new float[size*size]);
        System.out.print(c.schema());
        accelerator.compute(
                cc->MatrixMultiplyCompute.compute(cc,a,b,c,size)
        );




    }

}
