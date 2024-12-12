/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package experiments.spirv;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.buffer.SchemaBuilder;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CodeReflection;

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
            long i = kc.x;
            long size = sz;

            for (long j = 0; j < size; j++) {
                float sum = 0f;
                for (long k = 0; k < size; k++) {
                    //sum += a[kc.x * kc.max + k] * b[k * kc.max + j];
                    sum += a.array(i * size + k) * b.array(k * size + j);
                }
                //c[kc.x * kc.max + j] = sum;
                c.array(i * size + j, sum);
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a, F32Array b, F32Array c, int size) {

            computeContext.dispatchKernel(
                    size * size,                // range is passed as int and creation internalized
                    (kid) -> matmul(kid, a, b, c, size));  // kid is Kid1D has kid.x and kid.maxX

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
        float[] arrA = new float[size * size];
        float[] arrB = new float[size * size];
        var a = F32Array.create(accelerator, arrA.length);
        var b = F32Array.create(accelerator, arrB.length);
        var c = F32Array.create(accelerator, size * size);
        System.out.print(SchemaBuilder.schema(c));
        accelerator.compute(
                cc -> MatrixMultiplyCompute.compute(cc, a, b, c, size)
        );


    }

}
