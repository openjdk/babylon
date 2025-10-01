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
package oracle.code.hat;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.GlobalMesh2D;
import hat.KernelContext;
import hat.LocalMesh2D;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.F32Array;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.ifacemapper.MappableIface.*;

public class TestMatMul {

    private static final int SIZE = 256;

    @CodeReflection
    public static void matrixMultiplyKernel2D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            if (kc.y < kc.maxY) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + kc.y));
                }
                matrixC.array(kc.x * size + kc.y, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiplyKernel2DLI(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            if (kc.y < kc.maxY) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.y * size + k) * matrixB.array(k * size + kc.x));
                }
                matrixC.array(kc.y * size + kc.x, acc);
            }
        }
    }

    private interface MyLocalArrayFixedSize extends Buffer {
        void array(long index, float value);
        float array(long index);

        Schema<MyLocalArrayFixedSize> schema = Schema.of(MyLocalArrayFixedSize.class,
                myPrivateArray -> myPrivateArray
                        // It is a bound schema, so we fix the size here
                        .array("array", 256));

        static MyLocalArrayFixedSize create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }

        static MyLocalArrayFixedSize createLocal(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }

        static MyLocalArrayFixedSize createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
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
            acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + j));
        }
        return acc;
    }

    @CodeReflection
    public static void matrixMultiplyKernel1D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            for (int j = 0; j < size; j++) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + j));
                }
                matrixC.array(kc.x * size + j, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiplyKernel1DWithFunctionCalls(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            for (int j = 0; j < size; j++) {
                float acc = compute(kc, matrixA, matrixB, size, j);
                matrixC.array(kc.x * size + j, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiply1D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(globalSize));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel1D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    final static int BLOCK_SIZE = 16;

    @CodeReflection
    public static void matrixMultiply1DWithFunctionCalls(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel1DWithFunctionCalls(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh2D(globalSize, globalSize), new LocalMesh2D(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel2D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DLI(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh2D(globalSize, globalSize), new LocalMesh2D(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel2DLI(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh2D(globalSize, globalSize), new LocalMesh2D(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel2DTiling(kc, matrixA, matrixB, matrixC, globalSize)
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    private interface SharedWithPad extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<SharedWithPad> schema = Schema.of(SharedWithPad.class,
                arr -> arr.array("array", 1088));   // The size is BLOCK_M * (TILE_K + 1) to mitigate memory bank conflicts
        static SharedWithPad create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static SharedWithPad createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    private interface SharedWithPadB extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<SharedWithPadB> schema = Schema.of(SharedWithPadB.class,
                arr -> arr.array("array", 1040));    // The size is TILE_K * (BLOCk_M + 1) to mitigate memory bank conflicts
        static SharedWithPadB create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static SharedWithPadB createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    private interface PrivateAcc extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<PrivateAcc> schema = Schema.of(PrivateAcc.class,
                arr -> arr.array("array", 16));
        static PrivateAcc create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static PrivateAcc createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    private interface PrivateReg extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<PrivateReg> schema = Schema.of(PrivateReg.class,
                arr -> arr.array("array", 4));
        static PrivateReg create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static PrivateReg createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    // Ported from the HAT examples module.
    @CodeReflection
    public static void matrixMultiplyKernel2DRegisterTilingPortable(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {

        // Configuration for this kernel
        final int WG_M = 16;
        final int WG_N = 16;
        final int REG_M = 4;
        final int REG_N = 4;
        final int TILE_K = 16;
        final int BLOCK_M = WG_M * REG_M;
        final int BLOCK_N = WG_N * REG_N;

        // We compute squares matrices for simplification. But the code is
        // prepared to compute any compatible matmul sizes
        final int M = size;
        final int N = size;
        final int K = size;
        final int lda = size;
        final int ldb = size;
        final int ldc = size;

        final int lid_m = kc.lix;
        final int lid_n = kc.liy;
        final int gid_m = kc.bix;
        final int gid_n = kc.biy;

        // starting index of the tile in the matrix
        final int blockRow = gid_m * BLOCK_M;
        final int blockCol = gid_n * BLOCK_N;

        // Within the block, each thread computes REG_M x REG_N micro-tile
        // start at (blockRow + lid_m * REG_M) and (blockCol + lid_n * REG_N)
        final int cRowBase = blockRow + lid_m * REG_M;
        final int cColBase = blockCol + lid_n * REG_N;

        // Accumulators in private memory. This is set to REG_M x REG_N values
        PrivateAcc acc = PrivateAcc.createPrivate();
        for (int i = 0; i < REG_M; i++) {
            for (int j = 0; j < REG_N; j++) {
                acc.array(i * REG_M + j, 0.0f);
            }
        }

        // Shared memory to store tiles of matrixA and matrixB
        SharedWithPad sharedA = SharedWithPad.createLocal();
        SharedWithPadB sharedB = SharedWithPadB.createLocal();

        // Padding scales to access local memory.
        // We add +1 to mitigate memory bank conflicts.
        final int padA = TILE_K + 1;
        final int padB = BLOCK_M + 1;
        for (int tileIndex = 0; tileIndex < K; tileIndex += TILE_K) {
            // Load A tile (Block_M x Tile_K)
            for (int i = 0; i < REG_M; i++) {
                final int aRow = cRowBase + i;
                for (int kk = lid_n; kk < TILE_K; kk += WG_N) {
                    final int aCol = tileIndex + kk;
                    float valA = 0.0f;
                    if (aRow < M && aCol < K) {
                        valA = matrixA.array(aRow * lda + aCol);
                    }
                    sharedA.array(((aRow - blockRow) * padA) + kk, valA);
                }
            }

            // Load B tile (Tile_K x Block_N)
            for (int j = 0; j < REG_N; j++) {
                final int bCol = cColBase + j;
                for (int kk = lid_m; kk < TILE_K; kk += WG_M) {
                    final int bRow = tileIndex + kk;
                    float valB = 0.0f;
                    if (bRow < K && bCol < N) {
                        valB = matrixB.array(bRow * ldb + bCol);
                    }
                    sharedB.array((kk * padB) + (bCol - blockCol), valB);
                }
            }
            kc.barrier();

            // Compute the tile acc += sharedA[:,k] * sharedB[k,:]
            for (int kk = 0; kk < TILE_K && (tileIndex + kk) < K; kk++) {
                // Load the A and B operands into registers to reuse for REG_M x REG_N accumulations
                PrivateReg aReg = PrivateReg.createPrivate();
                PrivateReg bReg = PrivateReg.createPrivate();

                // Fetch a column kk of sharedA to private memory
                for (int i = 0; i < REG_M; i++) {
                    final int aRowL = (cRowBase + i) - blockRow;
                    float valPrivA = 0.0f;
                    if ((cRowBase + i) < M) {
                        valPrivA = sharedA.array(aRowL * padA + kk);
                    }
                    aReg.array(i, valPrivA);
                }

                // Fetch a row kk of sharedB to private memory
                for (int j = 0; j < REG_N; j++) {
                    final int bColL = (cColBase + j) - blockCol;
                    float valPrivB = 0.0f;
                    if ((cColBase + j) < N) {
                        valPrivB = sharedB.array(kk * padB + bColL);
                    }
                    bReg.array(j, valPrivB);
                }

                // FMA over the register tile
                for (int i = 0; i < REG_M; i++) {
                    final float a_ik = aReg.array(i);
                    for (int j = 0; j < REG_N; j++) {
                        float valRes = acc.array(i * REG_M + j);
                        valRes += a_ik * bReg.array(j);
                        acc.array(i * REG_M + j, valRes);
                    }
                }
            }
            kc.barrier();
        }
        // Write back the result of the register tile into matrixC
        for (int i = 0; i < REG_M; i++) {
            final int row = cRowBase + i;
            if (row < M) {
                for (int j = 0; j < REG_N; j++) {
                    final int col = cColBase + j;
                    if (col < N) {
                        matrixC.array(row * ldc + col, acc.array(i * REG_M + j));
                    }
                }
            }
        }
    }

    @CodeReflection
    public static void matrixMultiply2DRegisterTilingPortable(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int globalSize) {
        ComputeRange cudaRange = new ComputeRange(new GlobalMesh2D(256, 256), new LocalMesh2D(16, 16));
        cc.dispatchKernel(cudaRange,
                kc -> matrixMultiplyKernel2DRegisterTilingPortable(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @HatTest
    public void testMatrixMultiply2DRegisterTiling() {
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
                TestMatMul.matrixMultiply2DRegisterTilingPortable(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }

    private interface SharedMemory extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<SharedMemory> schema = Schema.of(SharedMemory.class,
                arr -> arr.array("array", 1024));
        static SharedMemory create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static SharedMemory createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    private interface PrivateArray extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<PrivateArray> schema = Schema.of(PrivateArray.class,
                arr -> arr.array("array", 16));
        static PrivateArray create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static PrivateArray createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }

    private interface FlatPrivate extends Buffer {
        void array(long index, float value);
        float array(long index);
        Schema<FlatPrivate> schema = Schema.of(FlatPrivate.class,
                arr -> arr.array("array", 4));
        static FlatPrivate create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static FlatPrivate createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
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

    @CodeReflection
    public static void matrixMultiply2DRegisterTiling(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, final int size) {
        ComputeRange cudaRange = new ComputeRange(new GlobalMesh2D(256, 256), new LocalMesh2D(16, 16));
        cc.dispatchKernel(cudaRange,
                kc -> matrixMultiplyKernel2DRegisterTiling(kc, matrixA, matrixB, matrixC, size)
        );
    }


    @HatTest
    public void testMatMul2DRegisterTilingV2() {
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
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }
}
