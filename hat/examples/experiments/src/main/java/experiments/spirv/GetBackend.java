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

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.CodeReflection;

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
                    sum += a.array(kid.x * kid.maxX + k) * b.array(k * kid.maxX + j);
                    //sum += a[kid.x * kid.max + k] * b[k * kid.max + j];
                    sum += a.array(kid.x * kid.maxX + k) * b.array(k * kid.maxX + j);
                }
                //c[kid.x * kid.max + j] = sum;
                c.array(kid.x * kid.maxX + j, sum);
            }
        }

        @CodeReflection
        static void compute(ComputeContext computeContext, F32Array a, F32Array b, F32Array c, int size) {
            computeContext.dispatchKernel(size * size, kc -> MatrixMultiply.kernel(kc, a, b, c));
        }

    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), (backend) ->
                backend.getClass().getSimpleName().startsWith("Spirv")
        );
        var a = F32Array.create(accelerator, 100);
        var b = F32Array.create(accelerator, 100);
        var c = F32Array.create(accelerator, 100);
        accelerator.compute(cc -> MatrixMultiply.compute(cc, a, b, c, 100));
    }

}
