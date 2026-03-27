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
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Reflect;
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
        cc.dispatchKernel(of2D(2048, 64, 128, 4), kc -> matrixMultiplyKernel2DLIF16(kc, matrixA, matrixB, matrixC, globalSize));
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

        for (int i = 0; i < 10; i++) {
            accelerator.compute(cc -> matrixMultiply2DLIF16(cc, matrixAHalf, matrixBHalf, matrixC, size));
        }

        runSequential(matrixAHalf, matrixBHalf, resultSequential, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final int index = j * size + i;
                float expectedValue = resultSequential.array(index);
                float gotValue = matrixC.array(index);
                //IO.println(gotValue + " vs " + expectedValue);
                HATAsserts.assertEquals(expectedValue, gotValue, 0.1f);
            }
        }
    }
}
