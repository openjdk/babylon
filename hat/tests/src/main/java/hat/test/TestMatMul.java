/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
import hat.NDRange;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F16;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.buffer.F32ArrayPadded;
import hat.buffer.Float4;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;
import jdk.incubator.code.CodeReflection;

import java.util.Random;

import static hat.ifacemapper.MappableIface.RO;
import static hat.ifacemapper.MappableIface.RW;

public class TestMatMul {

    private static final int SIZE = 256;

    @CodeReflection
    public static void matrixMultiplyKernel2D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.gix < kc.gsy) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.gix * size + k) * matrixB.array(k * size + kc.giy));
                }
                matrixC.array(kc.gix * size + kc.giy, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiplyKernel2DLI(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
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

    @CodeReflection
    public static void matrixMultiplyKernel2DLIF16(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @RW F16Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.giy < kc.gsy) {
                F16 acc = F16.of(0.0f);
                for (int k = 0; k < size; k++) {
                    F16 valA = matrixA.array(kc.giy * size + k);
                    F16 valB = matrixB.array(k * size + kc.gix);
                    F16 valc = F16.mul(valA, valB);
                    acc = F16.add(acc, valc);
                }
                F16 resultC = matrixC.array(kc.giy * size + kc.gix);
                resultC.value(acc.value());
            }
        }
    }

    private interface MyLocalArrayFixedSize extends DeviceType {
        void array(long index, float value);
        float array(long index);

        DeviceSchema<MyLocalArrayFixedSize> schema = DeviceSchema.of(MyLocalArrayFixedSize.class,
                myPrivateArray -> myPrivateArray
                        .withArray("array", 256));

        static MyLocalArrayFixedSize create(Accelerator accelerator) {
            return null;
        }

        static MyLocalArrayFixedSize createLocal() {
            return null;
        }
    }

    @CodeReflection
    public static void matrixMultiplyKernel2DTiling(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {

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
        for (int tile = 0; tile < (size/tileSize); tile++) {
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

    @CodeReflection
    public static float compute(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, int size, int j) {
        float acc = 0.0f;
        for (int k = 0; k < size; k++) {
            acc += (matrixA.array(kc.gix * size + k) * matrixB.array(k * size + j));
        }
        return acc;
    }

    @CodeReflection
    public static void matrixMultiplyKernel1D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
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

    @CodeReflection
    public static void matrixMultiplyKernel1DWithFunctionCalls(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            for (int j = 0; j < size; j++) {
                float acc = compute(kc, matrixA, matrixB, size, j);
                matrixC.array(kc.gix * size + j, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiply1D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(globalSize), NDRange.Local1D.of(16));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel1D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    final static int BLOCK_SIZE = 16;

    @CodeReflection
    public static void matrixMultiply1DWithFunctionCalls(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(size));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel1DWithFunctionCalls(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(globalSize, globalSize), NDRange.Local2D.of(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DLI(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(globalSize, globalSize), NDRange.Local2D.of(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DLI(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DLIF16(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @RW F16Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(globalSize, globalSize), NDRange.Local2D.of(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DLIF16(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(globalSize, globalSize), NDRange.Local2D.of(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DTiling(kc, matrixA, matrixB, matrixC, globalSize)
        );
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

    private static void runSequential(F32ArrayPadded matrixA, F32ArrayPadded matrixB, F32ArrayPadded matrixC, final int size) {
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

    @HatTest
    public void testMatrixMultiply1D() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply1D(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    @HatTest
    public void testMatrixMultiply1DWithFunctionCalls() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply1DWithFunctionCalls(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }


    @HatTest
    public void testMatrixMultiply2D() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2D(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    @HatTest
    public void testMatrixMultiply2DLI() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DLI(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    @HatTest
    public void testMatrixMultiply2DLIF16() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F16Array.create(accelerator, size * size);
        var matrixB = F16Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F16Array.create(accelerator, size * size);
        var resultSeq = F16Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixB.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DLIF16(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(
                        Float.float16ToFloat(resultSeq.array(i * size + j).value()),
                        Float.float16ToFloat(matrixC.array(i * size + j).value()),
                        0.01f);
            }
        }
    }

    @HatTest
    public void testMatrixMultiply2DTiling() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = SIZE;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DTiling(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    private interface SharedMemory extends DeviceType {
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
        default void storeFloat4View(Float4 float4, int index) {
        }
    }

    private interface PrivateArray extends DeviceType {
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

    private interface FlatPrivate extends DeviceType {
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

    // Code ported from the HAT example module.
    @CodeReflection
    public static void matrixMultiplyKernel2DRegisterTiling(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {

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
                    regM.array(i,  tileA.array((threadRow * TM + i) * BK + dotIdx));
                }
                for (int i = 0; i < TN; i++) {
                    regN.array(i,  tileB.array(dotIdx * BN + threadCol * TN + i));
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

    // Code ported from the HAT example module.
    @CodeReflection
    public static void matrixMultiplyKernel2DRegisterTilingVectorized(@RO KernelContext kc, @RO F32ArrayPadded matrixA, @RO F32ArrayPadded matrixB, @RW F32ArrayPadded matrixC, int size) {

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
                    regM.array(i,  tileA.array(dotIdx * BM + threadRow * TM + i));
                }
                for (int i = 0; i < TN; i++) {
                    regN.array(i,  tileB.array(dotIdx * (BN + extraCols) + threadCol * TN + i));
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

    @CodeReflection
    public static void matrixMultiply2DRegisterTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, final int size) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(256, 256), NDRange.Local2D.of(16, 16));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DRegisterTiling(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DRegisterTilingVectorized(@RO ComputeContext cc, @RO F32ArrayPadded matrixA, @RO F32ArrayPadded matrixB, @RW  F32ArrayPadded matrixC, final int size) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(256, 256), NDRange.Local2D.of(16, 16));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DRegisterTilingVectorized(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @HatTest
    public void testMatMul2DRegisterTiling() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = 1024;
        var matrixA = F32Array.create(accelerator, size * size);
        var matrixB = F32Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32Array.create(accelerator, size * size);
        var resultSeq = F32Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DRegisterTiling(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    @HatTest
    public void testMatMul2DRegisterTilingVectorized() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = 1024;
        var matrixA = F32ArrayPadded.create(accelerator, size * size);
        var matrixB = F32ArrayPadded.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F32ArrayPadded.create(accelerator, size * size);
        var resultSeq = F32ArrayPadded.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);

        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DRegisterTilingVectorized(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HATAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    private interface SharedMemoryHalf extends DeviceType {
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

    private interface PrivateArrayHalf extends DeviceType {
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

    private interface FlatPrivateHalf extends DeviceType {
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

    // Taking from the HAT Examples module
    @CodeReflection
    public static void matrixMultiplyKernel2DRegisterTilingHalf(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @RW F16Array matrixC, int size) {
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

        PrivateArrayHalf threadResults = PrivateArrayHalf.createPrivate();
        FlatPrivateHalf regM = FlatPrivateHalf.createPrivate();
        FlatPrivateHalf regN = FlatPrivateHalf.createPrivate();

        for (int i = 0; i < (TN * TN); i++) {
            F16 init = F16.of(0.0f);
            threadResults.array(i).value(init.value());
        }

        for (int bkIdx = 0; bkIdx < size; bkIdx += BK) {
            for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
                F16 ha = matrixA.array(((innerRowA + loadOffset) * size + innerColA) + aFrom);
                tileA.array((innerRowA + loadOffset) * BK + innerColA).value(ha.value());
            }
            for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
                F16 hb = matrixB.array(((innerRowB + loadOffset) * size + innerColB) + bFrom);
                tileB.array((innerRowB + loadOffset) * BN + innerColB).value(hb.value());
            }
            kc.barrier();

            aFrom += (BK);
            int f = BK * size;
            bFrom += f;

            for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
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
                        F16 acc2 = F16.add(acc, mul);   // FIXME: this is a partial fix until we support expressions such as: acc = acc <OP> val
                        threadResults.array((resIdxM * TN + resIdxN)).value(acc2.value());
                    }
                }
            }
            kc.barrier();
        }
        for (int resIdxM = 0; resIdxM < TM; resIdxM++) {
            for (int resIdxN = 0; resIdxN < TN; resIdxN++) {
                F16 result = threadResults.array(resIdxM * TN + resIdxN);
                matrixC.array((((threadRow * TM + resIdxM) * size + threadCol * TN + resIdxN) + (cFrom))).value(result.value());
            }
        }
    }

    @CodeReflection
    public static void matrixMultiply2DRegisterTilingHalf(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @RW F16Array matrixC, int globalSize) {
        NDRange ndRange = NDRange.of(NDRange.Global2D.of(256, 256), NDRange.Local2D.of(16, 16));
        cc.dispatchKernel(ndRange,
                kc -> matrixMultiplyKernel2DRegisterTilingHalf(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @HatTest
    public void matrixMultiply2DRegisterTilingHalf() {
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int size = 1024;
        var matrixA = F16Array.create(accelerator, size * size);
        var matrixB = F16Array.create(accelerator, size * size);

        // Matrix for the results
        var matrixC = F16Array.create(accelerator, size * size);
        var resultSeq = F16Array.create(accelerator, size * size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int j = 0; j < matrixA.length(); j++) {
            matrixA.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixB.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        accelerator.compute(cc ->
                TestMatMul.matrixMultiply2DRegisterTilingHalf(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                HATAsserts.assertEquals(F16.f16ToFloat(resultSeq.array(i * size + j)),
                                        F16.f16ToFloat(matrixC.array(i * size + j)),
                                        0.01f);
            }
        }
    }
}
