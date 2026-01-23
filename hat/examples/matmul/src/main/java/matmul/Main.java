/*
 * Copyright (c) 2024-2025, Oracle and/or its affiliates. All rights reserved.
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
package matmul;

import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange.Global2D;
import hat.NDRange.Local2D;
import hat.backend.Backend;
import hat.types.F16;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.buffer.F32ArrayPadded;
import hat.types.Float4;
import hat.device.DeviceSchema;
import hat.device.NonMappableIface;
import optkl.ifacemapper.MappableIface.WO;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;
import java.util.stream.IntStream;

import static hat.NDRange.NDRange2D;
import static hat.NDRange.of1D;
import static hat.NDRange.of2D;
import static optkl.ifacemapper.MappableIface.RO;

/**
 * Canonical example for Matrix Multiply.
 *
 * <p>How to run?</p>
 *
 * <p>For 2D Configuration:
 * <code>
 * java @hat/run ffi-opencl matmul 2D
 * </code>
 * </p>
 *
 * <p> For 1D Configuration
 * <code>
 * java @hat/run ffi-opencl matmul 1D
 * </code>
 * </p>
 */
public class Main {

    private static final boolean CHECK_RESULT = true;

    private static final int NUM_ITERATIONS = 10;

    /**
     * Naive Matrix Multiplication implemented in 2D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @Reflect
    public static void matrixMultiplyKernel2D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.giy < kc.gsy) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.gix * size + k) * matrixB.array(k * size + kc.giy));
                }
                matrixC.array(kc.gix * size + kc.giy, acc);
            }
        }
    }

    /**
     * Naive Matrix Multiplication implemented in 2D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @Reflect
    public static void matrixMultiplyKernel2DLI(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.giy < kc.gsy) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.giy * size + k) * matrixB.array(k * size + kc.gix));
                }
                matrixC.array(kc.giy * size + kc.gix, acc);
            }
        }
    }

    private interface MyLocalArrayFixedSize extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<MyLocalArrayFixedSize> schema = DeviceSchema.of(MyLocalArrayFixedSize.class,
                myPrivateArray -> myPrivateArray.withArray("array", 256));// It is a bound schema, so we fix the size here


        static MyLocalArrayFixedSize create(Accelerator accelerator) {
            return null;
        }

        static MyLocalArrayFixedSize createLocal(Accelerator accelerator) {
            return null;
        }

        static MyLocalArrayFixedSize createLocal() {
            return null;
        }
    }

    @Reflect
    public static void matrixMultiplyKernel2DTiling(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {

        final int tileSize = 16;
        MyLocalArrayFixedSize tileA = MyLocalArrayFixedSize.createLocal();
        MyLocalArrayFixedSize tileB = MyLocalArrayFixedSize.createLocal();

        int groupIndexX = kc.bix;
        int groupIndexY = kc.biy;
        int localIdx = kc.lix;
        int localIdy = kc.liy;

        // we identify the row and column
        int row = groupIndexY * tileSize + localIdy;
        int col = groupIndexX * tileSize + localIdx;

        // Compute matrix-vector and accumulate the result over the tiles
        float sum = 0.0f;
        for (int tile = 0; tile < (size / tileSize); tile++) {
            // Copy from global to shared memory
            tileA.array((long) localIdy * tileSize + localIdx, matrixA.array((long) row * size + tile * tileSize + localIdx));
            tileB.array((long) localIdy * tileSize + localIdx, matrixB.array((tile * tileSize + localIdy) * size + col));

            // Apply a barrier for the local group: we need to guarantee that all threads that belong
            // to the same group reach this point before doing the partial reduction
            kc.barrier();

            // compute partial reductions over the tile
            for (int k = 0; k < tileSize; k++) {
                sum += (tileA.array((long) localIdy * tileSize + k) * tileB.array(k * tileSize + localIdx));
            }

            // A new local barrier for all threads that belong to the same group before loading a new tile into
            // share memory. With the following barrier, we can ensure that all threads within the same workgroup
            // finished the compute for the partial reduction
            kc.barrier();
        }

        // copy result from shared memory to global memory
        matrixC.array((long) row * size + col, sum);
    }

    private interface SharedMemory extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<SharedMemory> schema = DeviceSchema.of(SharedMemory.class,
                arr -> arr.withArray("array", 1024));

        static SharedMemory create(Accelerator accelerator) {
            return null;
        }

        static SharedMemory createLocal() {
            return null;
        }
    }

    private interface PrivateArray extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<PrivateArray> schema = DeviceSchema.of(PrivateArray.class,
                arr -> arr.withArray("array", 16));

        static PrivateArray create(Accelerator accelerator) {
            return null;
        }

        static PrivateArray createPrivate() {
            return null;
        }
    }

    private interface FlatPrivate extends NonMappableIface {
        void array(long index, float value);

        float array(long index);

        DeviceSchema<FlatPrivate> schema = DeviceSchema.of(FlatPrivate.class,
                arr -> arr.withArray("array", 4));

        static FlatPrivate create(Accelerator accelerator) {
            return null;
        }

        static FlatPrivate createPrivate() {
            return null;
        }
    }

    /**
     * Algorithm for MatMul using 2D Cache (shared memory), Loop Tiling and 2D Register Tiling.
     *
     * <p>
     * We want to probe that HAT can represent more complex optimisations, and make use of the
     * different levels of the GPU's memory hierarchy, such as shared memory (as in CUDA shared memory),
     * and private memory. This code has been tested on NVIDIA A10 GPUs.
     * </p>
     *
     * <p>
     * The code has been adapted from CUDA to HAT based on the algorithms presented here:
     * {@url https://siboehm.com/articles/22/CUDA-MMM}
     * </p>
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @Reflect
    public static void matrixMultiplyKernel2DRegisterTiling(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {

        // Configuration for the kernel: Keep in mind that if you change the following parameters,
        // also change the scheduling (global and local work sizes).
        final int BM = 64;
        final int BN = 64;
        final int BK = 16;
        final int TM = 4;
        final int TN = 4;

        int bx = kc.bix;
        int by = kc.biy;

        int totalResultsBlockTile = BM * BN;
        final int numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

        final int linearLocalId = kc.liy * kc.lsx + kc.lix;
        final int threadCol = kc.lix;
        final int threadRow = kc.liy;

        SharedMemory tileA = SharedMemory.createLocal();
        SharedMemory tileB = SharedMemory.createLocal();

        int aFrom = by * BM * size;
        int bFrom = bx * BN;
        int v = bx * BN;
        int cFrom = (by * BM * size) + (v);

        final int innerRowA = linearLocalId / BK;
        final int innerColA = linearLocalId % BK;

        final int strideA = numThreadsBlockTile / BK;
        final int innerRowB = linearLocalId / BN;
        final int innerColB = linearLocalId % BN;

        int strideB = numThreadsBlockTile / BN;

        // Declarations of the arrays in private memory to perform register tiling
        PrivateArray threadResults = PrivateArray.createPrivate();
        FlatPrivate regM = FlatPrivate.createPrivate();
        FlatPrivate regN = FlatPrivate.createPrivate();

        // initialize values
        for (int i = 0; i < (TN * TN); i++) {
            threadResults.array(i, 0.0f);
        }

        // Each thread loops over the tiles
        for (int bkIdx = 0; bkIdx < size; bkIdx += BK) {

            // A) Load data into shared memory for array A
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                tileA.array((innerRowA + loadOffset) * BK + innerColA,
                        matrixA.array(((innerRowA + loadOffset) * size + innerColA) + aFrom));
            }

            // B) Load data matrixB into shared memory for array B
            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                tileB.array((innerRowB + loadOffset) * BN + innerColB,
                        matrixB.array(((innerRowB + loadOffset) * size + innerColB) + bFrom));
            }
            kc.barrier();

            aFrom += (BK);
            int f = BK * size;
            bFrom += f;

            // Per-thread, we load the data from the shared memory into register for both
            // array A and array B (matrix A and B), and then perform the reduction within
            // the small region in private memory.
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                // block into registers
                for (int i = 0; i < TM; i++) {
                    regM.array(i, tileA.array((threadRow * TM + i) * BK + dotIdx));
                }
                for (int i = 0; i < TN; i++) {
                    regN.array(i, tileB.array(dotIdx * BN + threadCol * TN + i));
                }
                for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                        float val = regM.array(resIdxM) * regN.array(resIdxN);
                        float acc = threadResults.array(resIdxM * TN + resIdxN);
                        acc += val;
                        threadResults.array((resIdxM * TN + resIdxN), (acc));
                    }
                }
            }
            kc.barrier();
        }

        // Finally, we store the results of the reductions for the whole 2D register block into global memory.
        // Essentially, each thread compute a small block of TM * TN sub-block size.
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                float value = threadResults.array(resIdxM * TN + resIdxN);
                matrixC.array((((threadRow * TM + resIdxM) * size + threadCol * TN + resIdxN) + (cFrom)), value);
            }
        }
    }

    /**
     * Algorithm for MatMul using 2D Cache (shared memory), Loop Tiling and 2D Register Tiling + Vector Loads/Stores from/to
     * global memory to shared memory.
     *
     * <p>
     * We want to probe that HAT can represent more complex optimisations, and make use of the
     * different levels of the GPU's memory hierarchy, such as shared memory (as in CUDA shared memory),
     * and private memory. This code has been tested on NVIDIA A10 GPUs.
     * </p>
     *
     * <p>
     * The code has been adapted from CUDA to HAT based on the algorithms presented here:
     * {@url https://siboehm.com/articles/22/CUDA-MMM}
     * </p>
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @Reflect
    public static void matrixMultiplyKernel2DRegisterTilingVectorized(@RO KernelContext kc, @RO F32ArrayPadded matrixA, @RO F32ArrayPadded matrixB, @WO F32ArrayPadded matrixC, int size) {

        // Configuration for the kernel: Keep in mind that if you change the following parameters,
        // also change the scheduling (global and local work sizes).
        final int M = size;
        final int N = size;
        final int K = size;
        final int BM = 64;
        final int BN = 64;
        final int BK = 16;
        final int TM = 4;
        final int TN = 4;

        int bx = kc.bix;
        int by = kc.biy;

        final int linearLocalId = kc.liy * kc.lsx + kc.lix;
        final int threadCol = kc.lix;
        final int threadRow = kc.liy;

        SharedMemory tileA = SharedMemory.createLocal();
        SharedMemory tileB = SharedMemory.createLocal();

        int aFrom = by * BM * size;
        int bFrom = bx * BN;
        int v = bx * BN;
        int cFrom = (by * BM * size) + (v);

        final int innerRowA = linearLocalId / (BK / 4);
        final int innerColA = linearLocalId % (BK / 4);
        final int innerRowB = linearLocalId / (BN / 4);
        final int innerColB = linearLocalId % (BN / 4);

        // Declarations of the arrays in private memory to perform register tiling
        PrivateArray threadResults = PrivateArray.createPrivate();
        FlatPrivate regM = FlatPrivate.createPrivate();
        FlatPrivate regN = FlatPrivate.createPrivate();

        // initialize values
        for (int i = 0; i < (TN * TN); i++) {
            threadResults.array(i, 0.0f);
        }

        final int extraCols = 0;

        // Each thread loops over the tiles
        for (int bkIdx = 0; bkIdx < size; bkIdx += BK) {

            Float4 loadA = matrixA.float4View((innerRowA * K + innerColA * 4) + aFrom);
            tileA.array((innerColA * 4 + 0) * BM + innerRowA, loadA.x());
            tileA.array((innerColA * 4 + 1) * BM + innerRowA, loadA.y());
            tileA.array((innerColA * 4 + 2) * BM + innerRowA, loadA.z());
            tileA.array((innerColA * 4 + 3) * BM + innerRowA, loadA.w());

            Float4 loadB = matrixB.float4View((innerRowB * N + innerColB * 4) + bFrom);
            tileB.array(innerRowB * (BN + extraCols) + innerColB * 4 + 0, loadB.x());
            tileB.array(innerRowB * (BN + extraCols) + innerColB * 4 + 1, loadB.y());
            tileB.array(innerRowB * (BN + extraCols) + innerColB * 4 + 2, loadB.z());
            tileB.array(innerRowB * (BN + extraCols) + innerColB * 4 + 3, loadB.w());

            kc.barrier();

            aFrom += (BK);
            int f = BK * size;
            bFrom += f;

            // Per-thread, we load the data from the shared memory into register for both
            // array A and array B (matrix A and B), and then perform the reduction within
            // the small region in private memory.
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                // block into registers
                for (int i = 0; i < TM; i++) {
                    regM.array(i, tileA.array(dotIdx * BM + threadRow * TM + i));
                }
                for (int i = 0; i < TN; i++) {
                    regN.array(i, tileB.array(dotIdx * (BN + extraCols) + threadCol * TN + i));
                }
                for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                        float val = regM.array(resIdxM) * regN.array(resIdxN);
                        float acc = threadResults.array(resIdxM * TN + resIdxN);
                        acc += val;
                        threadResults.array((resIdxM * TN + resIdxN), (acc));
                    }
                }
            }
            kc.barrier();
        }

        // Finally, we store the results of the reductions for the whole 2D register block into global memory.
        // Essentially, each thread compute a small block of TM * TN sub-block size.
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                float value = threadResults.array(resIdxM * TN + resIdxN);
                matrixC.array((((threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN) + (cFrom)), value);
            }
        }
    }

    private interface SharedMemoryHalf extends NonMappableIface {
        F16 array(int index);

        DeviceSchema<SharedMemoryHalf> schema = DeviceSchema.of(SharedMemoryHalf.class,
                arr -> arr.withArray("array", 1024)
                        .withDeps(F16.class, half -> half.withField("value")));

        static SharedMemoryHalf create(Accelerator accelerator) {
            return null;
        }

        static SharedMemoryHalf createLocal() {
            return null;
        }
    }

    private interface PrivateArrayHalf extends NonMappableIface {
        F16 array(int index);

        DeviceSchema<PrivateArrayHalf> schema = DeviceSchema.of(PrivateArrayHalf.class,
                arr -> arr.withArray("array", 16)
                        .withDeps(F16.class, half -> half.withField("value")));

        static PrivateArrayHalf create(Accelerator accelerator) {
            return null;
        }

        static PrivateArrayHalf createPrivate() {
            return null;
        }
    }

    private interface FlatPrivateHalf extends NonMappableIface {
        F16 array(int index);

        DeviceSchema<FlatPrivateHalf> schema = DeviceSchema.of(FlatPrivateHalf.class,
                arr -> arr.withArray("array", 4)
                        .withDeps(F16.class, half -> half.withField("value")));

        static FlatPrivateHalf create(Accelerator accelerator) {
            return null;
        }

        static FlatPrivateHalf createPrivate() {
            return null;
        }
    }

    @Reflect
    public static void matrixMultiplyKernel2DRegisterTilingHalf(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F16Array matrixC, int size) {

        // Configuration for the kernel: Keep in mind that if you change the following parameters,
        // also change the scheduling (global and local work sizes).
        final int BM = 64;
        final int BN = 64;
        final int BK = 16;
        final int TM = 4;
        final int TN = 4;

        int bx = kc.bix;
        int by = kc.biy;

        int totalResultsBlockTile = BM * BN;
        final int numThreadsBlockTile = totalResultsBlockTile / (TM * TN);

        final int linearLocalId = kc.liy * kc.lsx + kc.lix;
        final int threadCol = kc.lix;
        final int threadRow = kc.liy;

        SharedMemoryHalf tileA = SharedMemoryHalf.createLocal();
        SharedMemoryHalf tileB = SharedMemoryHalf.createLocal();

        int aFrom = by * BM * size;
        int bFrom = bx * BN;
        int v = bx * BN;
        int cFrom = (by * BM * size) + (v);

        final int innerRowA = linearLocalId / BK;
        final int innerColA = linearLocalId % BK;

        final int strideA = numThreadsBlockTile / BK;
        final int innerRowB = linearLocalId / BN;
        final int innerColB = linearLocalId % BN;

        int strideB = numThreadsBlockTile / BN;

        // Declarations of the arrays in private memory to perform register tiling
        PrivateArrayHalf threadResults = PrivateArrayHalf.createPrivate();
        FlatPrivateHalf regM = FlatPrivateHalf.createPrivate();
        FlatPrivateHalf regN = FlatPrivateHalf.createPrivate();

        // initialize values
        for (int i = 0; i < (TN * TN); i++) {
            F16 init = F16.of(0.0f);
            threadResults.array(i).value(init.value());
        }

        // Each thread loops over the tiles
        for (int bkIdx = 0; bkIdx < size; bkIdx += BK) {

            // A) Load data into shared memory for array A
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                F16 ha = matrixA.array(((innerRowA + loadOffset) * size + innerColA) + aFrom);
                tileA.array((innerRowA + loadOffset) * BK + innerColA).value(ha.value());
            }

            // B) Load data matrixB into shared memory for array B
            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                F16 hb = matrixB.array(((innerRowB + loadOffset) * size + innerColB) + bFrom);
                tileB.array((innerRowB + loadOffset) * BN + innerColB).value(hb.value());
            }
            kc.barrier();

            aFrom += (BK);
            int f = BK * size;
            bFrom += f;

            // Per-thread, we load the data from the shared memory into register for both
            // array A and array B (matrix A and B), and then perform the reduction within
            // the small region in private memory.
            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
                // block into registers
                for (int i = 0; i < TM; i++) {
                    F16 ha = tileA.array((threadRow * TM + i) * BK + dotIdx);
                    regM.array(i).value(ha.value());
                }
                for (int i = 0; i < TN; i++) {
                    F16 hb = tileB.array(dotIdx * BN + threadCol * TN + i);
                    regN.array(i).value(hb.value());
                }
                for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
                    for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                        F16 privA = regM.array(resIdxM);
                        F16 privB = regN.array(resIdxN);
                        F16 mul = F16.mul(privA, privB);
                        F16 acc = threadResults.array(resIdxM * TN + resIdxN);
                        acc = F16.add(acc, mul);
                        threadResults.array((resIdxM * TN + resIdxN)).value(acc.value());
                    }
                }
            }
            kc.barrier();
        }

        // Finally, we store the results of the reductions for the whole 2D register block into global memory.
        // Essentially, each thread compute a small block of TM * TN sub-block size.
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                F16 result = threadResults.array(resIdxM * TN + resIdxN);
                matrixC.array((((threadRow * TM + resIdxM) * size + threadCol * TN + resIdxN) + (cFrom))).value(result.value());
            }
        }
    }

    @Reflect
    public static float compute(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, int size, int j) {
        float acc = 0.0f;
        for (int k = 0; k < size; k++) {
            acc += (matrixA.array(kc.gix * size + k) * matrixB.array(k * size + j));
        }
        return acc;
    }

    /**
     * Naive Matrix Multiplication implemented in 1D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @Reflect
    public static void matrixMultiplyKernel1D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            for (int j = 0; j < size; j++) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.gix * size + k) * matrixB.array(k * size + j));
                }
                matrixC.array(kc.gix * size + j, acc);
            }
        }
    }

    /**
     * 1D Matrix Multiply with function calls passing the kernel context ID. This is just for testing purposes.
     */
    @Reflect
    public static void matrixMultiplyKernel1DWithFunctionCalls(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            for (int j = 0; j < size; j++) {
                float acc = compute(kc, matrixA, matrixB, size, j);
                matrixC.array(kc.gix * size + j, acc);
            }
        }
    }

    @Reflect
    public static void matrixMultiply1D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of1D(globalSize,16),
                kc -> matrixMultiplyKernel1D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    final static int BLOCK_SIZE = 16;

    @Reflect
    public static void matrixMultiply1DWithFunctionCalls(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int size) {
        cc.dispatchKernel(of1D(size,16),
                kc -> matrixMultiplyKernel1DWithFunctionCalls(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @Reflect
    public static void matrixMultiply2D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(globalSize, globalSize,BLOCK_SIZE, BLOCK_SIZE),
                kc -> matrixMultiplyKernel2D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void matrixMultiply2DLI(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(globalSize, globalSize,BLOCK_SIZE, BLOCK_SIZE),
                kc -> matrixMultiplyKernel2DLI(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void matrixMultiply2DTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(globalSize, globalSize, BLOCK_SIZE, BLOCK_SIZE),
                kc -> matrixMultiplyKernel2DTiling(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void matrixMultiply2DRegisterTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(256, 256,BLOCK_SIZE, BLOCK_SIZE),
                kc -> matrixMultiplyKernel2DRegisterTiling(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void matrixMultiply2DRegisterTilingVectorizedAccesses(@RO ComputeContext cc, @RO F32ArrayPadded matrixA, @RO F32ArrayPadded matrixB, @WO F32ArrayPadded matrixC, int globalSize) {
        cc.dispatchKernel(of2D(256, 256,BLOCK_SIZE, BLOCK_SIZE),
                kc -> matrixMultiplyKernel2DRegisterTilingVectorized(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void matrixMultiply2DRegisterTilingHalf(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F16Array matrixC, int globalSize) {
        var range = NDRange2D.of(Global2D.of(256, 256), Local2D.of(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(range,
                kc -> matrixMultiplyKernel2DRegisterTilingHalf(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    private static void runSequential(F32Array matrixA, F32Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    float a = matrixA.array((long) i * size + k);
                    float b = matrixB.array((long) k * size + j);
                    sum += a * b;
                }
                matrixC.array((long) i * size + j, sum);
            }
        }
    }

    private static void runSequential(F32ArrayPadded matrixA, F32ArrayPadded matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    float a = matrixA.array((long) i * size + k);
                    float b = matrixB.array((long) k * size + j);
                    sum += a * b;
                }
                matrixC.array((long) i * size + j, sum);
            }
        }
    }

    private static void runSequential(F16Array matrixA, F16Array matrixB, F16Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                F16 sum = F16.of(0.0f);
                for (int k = 0; k < size; k++) {
                    F16 a = matrixA.array((long) i * size + k);
                    F16 b = matrixB.array((long) k * size + j);
                    sum = F16.add(sum, F16.mul(a, b));
                }
                matrixC.array((long) i * size + j).value(sum.value());
            }
        }
    }

    /**
     * Configuration to use in this example to represent a
     * 1D range or 2D range.
     */
    private enum Configuration {
        _MT,  // Runs the Multi-thread Java code on the host side (no HAT)
        _1D,   //
        _1DFC, // 1D with multiple function calls: This is just for testing
        _2D,   //
        _2DLI,
        _2DTILING,
        _2DREGISTER_TILING,
        _2DREGISTER_TILING_VECTORIZED,
        _2DREGISTER_TILING_FP16,
    }

    /**
     * Run a 2D version by default.
     *
     * @param args args: <"1D"|"2D"> for 1D dispatch
     *
     */
    static void main(String[] args) {
        Configuration configuration = Configuration._2DTILING;
        if (args.length > 0) {
            configuration = switch (args[0]) {
                case "MT" -> Configuration._MT;
                case "1D" -> Configuration._1D;
                case "1DFC" -> Configuration._1DFC;
                case "2D" -> Configuration._2D;
                case "2DLI" -> Configuration._2DLI;
                case "2DTILING" -> Configuration._2DTILING;
                case "2DREGISTERTILING" -> Configuration._2DREGISTER_TILING;
                case "2DREGISTERTILING_V" -> Configuration._2DREGISTER_TILING_VECTORIZED;
                case "2DREGISTERTILING_FP16" -> Configuration._2DREGISTER_TILING_FP16;
                default -> configuration;
            };
        }

        System.out.println("[INFO] NDRangeConfiguration: " + configuration);

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);
        System.out.println(accelerator);

        final int size = 1024;
        IO.println("[INFO] Starting Matrix Multiplication with size: " + size + "x" + size);
        F32Array matrixA;
        F32Array matrixB;
        F32Array matrixC;
        F32ArrayPadded matrixAPad;
        F32ArrayPadded matrixBPad;
        F32ArrayPadded matrixCPad;
        F16Array matrixAHalf;
        F16Array matrixBHalf;
        F16Array matrixCHalf;

        Random r = new Random(19);
        if (configuration == Configuration._2DREGISTER_TILING_VECTORIZED) {
            matrixBHalf = null;
            matrixAHalf = null;
            matrixCHalf = null;
            matrixC = null;
            matrixB = null;
            matrixA = null;
            matrixAPad = F32ArrayPadded.create(accelerator, size * size);
            matrixBPad = F32ArrayPadded.create(accelerator, size * size);
            matrixCPad = F32ArrayPadded.create(accelerator, size * size);
            for (int j = 0; j < matrixAPad.length(); j++) {
                matrixAPad.array(j, r.nextFloat());
                matrixBPad.array(j, r.nextFloat());
            }
        } else {
            matrixCPad = null;
            matrixBPad = null;
            matrixAPad = null;
            if (configuration == Configuration._2DREGISTER_TILING_FP16) {
                matrixC = null;
                matrixB = null;
                matrixA = null;
                matrixAHalf = F16Array.create(accelerator, size * size);
                matrixBHalf = F16Array.create(accelerator, size * size);
                matrixCHalf = F16Array.create(accelerator, size * size);
                for (int j = 0; j < matrixAHalf.length(); j++) {
                    matrixAHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
                    matrixBHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
                }
            } else {
                matrixBHalf = null;
                matrixAHalf = null;
                matrixCHalf = null;
                matrixA = F32Array.create(accelerator, size * size);
                matrixB = F32Array.create(accelerator, size * size);
                matrixC = F32Array.create(accelerator, size * size);
                for (int j = 0; j < matrixA.length(); j++) {
                    matrixA.array(j, r.nextFloat());
                    matrixB.array(j, r.nextFloat());
                }
            }
        }


        F32Array resultSeq = null;
        F16Array resultSeqHalf = null;
        if (configuration == Configuration._2DREGISTER_TILING_FP16) {
            resultSeqHalf = F16Array.create(accelerator, size * size);
        } else {
            resultSeq = F32Array.create(accelerator, size * size);
        }

        // Run Seq for reference
        if (configuration == Configuration._2DREGISTER_TILING_VECTORIZED) {
            runSequential(matrixAPad, matrixBPad, resultSeq, size);
        } else if (configuration == Configuration._2DREGISTER_TILING_FP16) {
            runSequential(matrixAHalf, matrixBHalf, resultSeqHalf, size);
        } else {
            runSequential(matrixA, matrixB, resultSeq, size);
        }

        for (int it = 0; it < NUM_ITERATIONS; it++) {
            long start = System.nanoTime();
            switch (configuration) {
                case _MT -> runMultiThreadedWithStreams(matrixA, matrixB, matrixC, size);
                case _1D -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply1D(cc, matrixA, matrixB, matrixC, size));
                case _1DFC -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply1DWithFunctionCalls(cc, matrixA, matrixB, matrixC, size));
                case _2D -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2D(cc, matrixA, matrixB, matrixC, size));
                case _2DLI -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2DLI(cc, matrixA, matrixB, matrixC, size));
                case _2DTILING -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2DTiling(cc, matrixA, matrixB, matrixC, size));
                case _2DREGISTER_TILING -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2DRegisterTiling(cc, matrixA, matrixB, matrixC, size));
                case _2DREGISTER_TILING_VECTORIZED -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2DRegisterTilingVectorizedAccesses(cc, matrixAPad, matrixBPad, matrixCPad, size));
                case _2DREGISTER_TILING_FP16 -> accelerator.compute((@Reflect Compute) cc ->
                        matrixMultiply2DRegisterTilingHalf(cc, matrixAHalf, matrixBHalf, matrixCHalf, size));
                default -> throw new RuntimeException("Unknown configuration: " + configuration);
            }

            long end = System.nanoTime();
            System.out.println("Elapsed Time: " + (end - start) + " ns");

            // If the check is ON, then check first and lat iterations
            if (it == 0 || it == (NUM_ITERATIONS - 1) && CHECK_RESULT) {
                // Check result for the first iteration
                boolean isCorrect = true;
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        float expectedValue;
                        if (configuration == Configuration._2DREGISTER_TILING_FP16) {
                            expectedValue = F16.f16ToFloat(resultSeqHalf.array(i * size + j));
                        } else {
                            expectedValue = resultSeq.array(i * size + j);
                        }
                        float gotValue;
                        if (configuration == Configuration._2DREGISTER_TILING_VECTORIZED) {
                            gotValue = matrixCPad.array(i * size + j);
                        } else if (configuration == Configuration._2DREGISTER_TILING_FP16) {
                            gotValue = F16.f16ToFloat(matrixCHalf.array(i * size + j));
                        } else {
                            gotValue = matrixC.array(i * size + j);
                        }
                        if (Math.abs(expectedValue - gotValue) > 0.01f) {
                            IO.println(expectedValue + " != " + gotValue);
                            isCorrect = false;
                            break;
                        }
                    }
                    if (!isCorrect) {
                        break;
                    }
                }

                if (isCorrect) {
                    System.out.println("Result is correct!");
                } else {
                    System.out.println("Result is wrong!");
                }
            }
        }
    }

    private static void runMultiThreadedWithStreams(F32Array matrixA, F32Array matrixB, F32Array matrixC, int size) {
        IntStream.range(0, size).parallel().forEach(i -> {
            IntStream.range(0, size).parallel().forEach(j -> {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    sum += matrixA.array(i * size + k) * matrixB.array(k * size + j);
                }
                matrixC.array(i * size + j, sum);
            });
        });
    }
}
