/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import hat.annotations.Kernel;
import hat.annotations.Preformatted;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Reflect;

import static hat.NDRange.Global2D;
import static hat.NDRange.Local2D;
import static hat.NDRange.NDRange1D;
import static hat.NDRange.NDRange2D;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.NDRange.of2D;

/**
 * How to run?
 *
 * <code>
 *     HAT=SHOW_CODE java -cp hat/job.jar hat.java test ffi-cuda hat.test.TestTensors
 * </code>
 *
 */
public class TestTensors {

    @Reflect
//    @Kernel("""
//            #include <mma.h>
//            using namespace nvcuda;
//            HAT_KERNEL void matrixMultiplyKernel2DLIF16(
//                HAT_GLOBAL_MEM KernelContext_t* kc,
//                HAT_GLOBAL_MEM F16Array_t* matrixA,
//                HAT_GLOBAL_MEM F16Array_t* matrixB,
//                HAT_GLOBAL_MEM F32Array_t* matrixC,
//                int size
//            ){
//                int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//                int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
//                int lda = 1024;
//                int ldb = 1024;
//                int ldc = 1024;
//
//                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
//                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
//                wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
//
//                wmma::fill_fragment(acc_frag, 0.0f);
//
//                half *a = (half *)matrixA;
//                half *b = (half *)matrixB;
//                float *c = (float *)matrixC;
//
//                const int headSize = 2; // header is one `int`, which is 2 `half` floats.
//
//                        for(int i = 0; i<size; i += 16){
//                            int aRow = warpM * 16;
//                            int aCol = i;
//
//                            int bRow = i;
//                            int bCol = warpN * 16;
//
//                            if (aRow < 1024 && aCol < 1024 && bRow < 1024 && bCol < 1024) {
//                                // Load tensor
//                                wmma::load_matrix_sync(a_frag, a + headSize + aRow + aCol * lda, lda);
//                                wmma::load_matrix_sync(b_frag, b + headSize + bRow + bCol * ldb, ldb);
//
//                                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//                            }
//                        }
//                        int cRow = warpM * 16;
//                        int cCol = warpN * 16;
//
//                        const int headerC = 1; // the header for the output is 1 integer.
//                        wmma::store_matrix_sync(c + headerC + cRow + cCol * ldc, acc_frag, ldc, wmma::mem_col_major);
//                return;
//            }
//            """)
    @Kernel("""
            HAT_KERNEL void matrixMultiplyKernel2DLIF16(
                            HAT_GLOBAL_MEM KernelContext_t* kc,
                            HAT_GLOBAL_MEM F16Array_t* matrixA,
                            HAT_GLOBAL_MEM F16Array_t* matrixB,
                            HAT_GLOBAL_MEM F32Array_t* matrixC,
                            int size
                        ){
                int WMMA_M = 16;
                int WMMA_N = 16;
                int WMMA_K = 16;
                int warpM = HAT_GIX;
                int warpN = HAT_GIY;
                int lda = 1024;
                int ldb = 1024;
                int ldc = 1024;

                // wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
                //      => Tensor tensorA = Tensor.create(Tensor.FIRST, Tensor.Shape(16, 16, 16), F16.class);
                F16_t a_frag[256];

                //wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                //      -> Tensor tensorB = Tensor.create(Tensor.SECOND, Tensor.Shape(16, 16, 16), F16.class);
                F16_t b_frag[256];

                // wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
                float acc[16][16];

                //wmma::fill_fragment(acc_frag, 0.0f);
                for (int m = 0; m < WMMA_M; m++)
                    for (int n = 0; n < WMMA_N; n++)
                        acc[m][n] = 0.0f;

                // this loop remains the same
                for(int i = 0; i<size; i=i+WMMA_K){
                    int aRow = warpM * WMMA_M;
                    int aCol = i;
                    int bRow = i;
                    int bCol = warpN*WMMA_N;

                    // wmma::load_matrix_sync(a_frag, a + headSize + aRow + aCol * lda, lda);
                    for (int m = 0; m < WMMA_M; m++) {
                        int rowA = aRow + m;
                        for (int n = 0; n < WMMA_N; n++) {
                            int colA = aCol + n;
                            int idxA = rowA + colA * lda;
                            HAT_GLOBAL_MEM F16Impl_t* ha = &matrixA->array[idxA];
                            F16_t r = (F16_t){ha->value};
                            a_frag[m * WMMA_M + n] = r;
                        }
                    }

                    // wmma::load_matrix_sync(b_frag, b + headSize + bRow + bCol * ldb, ldb);
                    for (int m = 0; m < WMMA_M; m++) {
                        int rowB = bRow + m;
                        for (int n = 0; n < WMMA_N; n++) {
                            int colB = bCol + n;
                            int idxB = rowB + colB * ldb;
                            HAT_GLOBAL_MEM F16Impl_t* hb = &matrixB->array[idxB];
                            F16_t r = (F16_t){hb->value};
                            b_frag[m * WMMA_M + n] = r;
                        }
                    }

                    // wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                    for (int m = 0; m < WMMA_M; m++) {
                        for (int n = 0; n < WMMA_N; n++) {
                            float sum = acc[m][n];
                            for (int k = 0; k < WMMA_K; k++) {
                                F16_t ha = a_frag[m * WMMA_M + k];
                                F16_t hb = b_frag[k * WMMA_M + n];

                                F16_t result = (F16_t){(ha.value * hb.value)};
                                sum += (float)(result.value);
                            }
                            acc[m][n] = sum;
                        }
                    }
                }

                int cRow = warpM*WMMA_M;
                int cCol = warpN*WMMA_N;
                // wmma::store_sync
                for (int m = 0; m < WMMA_M; m++) {
                    int rowC = cRow + m;
                    if (rowC >= size) continue;
                    for (int n = 0; n < WMMA_N; n++) {
                        int colC = cCol + n;
                        if (colC >= size) continue;
                        int idxC = (cRow + m) + (cCol + n) * ldc;  // Almost same index
                        matrixC->array[idxC] = acc[m][n];
                    }
                }
            }
            """)
    public static void matrixMultiplyKernel2DLIF16(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int size) {
        final int WMMA_M = 16;
        final int WMMA_N = 16;
        final int WMMA_K = 16;
        int warpM = kc.gix / kc.warpSize;
        int warpN = kc.giy;

        final int lda = 1024;
        final int ldb = 1024;
        final int ldc = 1024;

        Tensor tensorA = Tensor.create(Tensor.FIRST, Tensor.Shape(16, 16, 16), F16.class, Tensor.ofColumnMajor());
        Tensor tensorB = Tensor.create(Tensor.SECOND, Tensor.Shape(16, 16, 16), F16.class, Tensor.ofColumnMajor());
        Tensor acc = Tensor.create(Tensor.ACC, Tensor.Shape(16, 16, 16), float.class);

        Tensor.fill(acc, 0.0f);

        for (int i = 0; i < size; i += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = i;

            int bRow = i;
            int bCol = warpN * WMMA_N;

            if (aRow < lda && aCol < lda && bRow < ldb && bCol < ldb) {

                tensorA = Tensor.load(matrixA, aRow, aCol, lda);
                tensorB = Tensor.load(matrixB, bRow, bCol, ldb);

                // acc = tensorA * tensorB + acc
                Tensor.mma(acc, tensorA, tensorB, acc);
            }
        }
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;
        Tensor.store(matrixC, cRow, cCol, acc, ldc, Tensor.ofColumnMajor());
    }

    @Reflect
    public static void matrixMultiply2DLIF16(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int globalSize) {
        // var ndRange = of2D(2048, 64, 128, 4);  // When we launch using the CUDA backend
        // For the OpenCL backend: [ (size / tile), (size / tile) ]
        var ndRange = NDRange2D.of(Global2D.of(64, 64), Local2D.of(16, 4));
        cc.dispatchKernel(ndRange, kc -> matrixMultiplyKernel2DLIF16(kc, matrixA, matrixB, matrixC, globalSize));
    }

    private static void runSequential(F16Array matrixA, F16Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    F16 a = matrixA.array((long) j * size + k);
                    F16 b = matrixB.array((long) k * size + i);
                    F16 mul = F16.mul(a, b);
                    sum += F16.f16ToFloat(mul);
                }
                matrixC.array((long) j * size + i, sum);
            }
        }
    }

    @HatTest
    @Reflect
    public void test_tensors_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;

        F16Array matrixAHalf = F16Array.create(accelerator, size * size);
        F16Array matrixBHalf = F16Array.create(accelerator, size * size);
        F32Array matrixC = F32Array.create(accelerator, size * size);
        F32Array resultSequential = F32Array.create(accelerator, size * size);

        Random r = new Random(19);
        for (int j = 0; j < matrixAHalf.length(); j++) {
            matrixAHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixBHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(cc -> matrixMultiply2DLIF16(cc, matrixAHalf, matrixBHalf, matrixC, size));

        runSequential(matrixAHalf, matrixBHalf, resultSequential, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final int index = j * size + i;
                float expectedValue = resultSequential.array(index);
                float gotValue = matrixC.array(index);
                HATAsserts.assertEquals(expectedValue, gotValue, 0.1f);
            }
        }
    }
}
