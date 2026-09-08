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
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.NDRange;
import hat.TileContext;
import hat.TileOp;
import hat.backend.Backend;
import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF32;

import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

/**
 * How to run?
 *
 * <p>
 *     <code>
 *         java @.ffi-opencl-test hat.test.TestTileAPI
 *     </code>
 * </p>
 *
 */
public class TestTileAPI {

    // ================================================================================================================
    // Expressing Vector Addition
    // ================================================================================================================
    @Reflect
    public static void vectorAddTile(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        final var pid = TileContext.BIDX();
        var tileA = TileContext.load(inputA, pid, tileSize);
        var tileB = TileContext.load(inputB, pid, tileSize);
        var result = TileOp.add(tileA, tileB);
        TileContext.store(output, pid, result);
    }

    @Reflect
    public static void vectorAddTile(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.m(), tileSize), () -> vectorAddTile(inputA, inputB, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_00() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        final int tileSize = 32;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);

        // Fill data
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        TensorF32 result = TensorF32.create(accelerator, size);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));

        // change the tile size
        final int newTileSize = 16;
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, newTileSize));

        // Alternate the tile size
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, newTileSize));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    @HatTest
    public void test_hat_tile_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = Math.powExact(2, 12);
        final int tile_size = 64;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);
        accelerator.compute( (@Reflect Compute)computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tile_size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    // ================================================================================================================
    // Expressing MatMul
    // ================================================================================================================
    public static final int GROUP_SIZE_M = 8;

    @Reflect
    public static void matmul(Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int num_tiles) {

        // Calculate bidx and bidy using swizzle
        final int bid = TileContext.BIDX();
        final int num_bid_m = TileOp.ceildiv(M, tm);
        final int num_bid_n = TileOp.ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = TileOp.min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / group_size_m;

        // Calculate the total number of tiles
        //final int num_tiles = TileOp.numTiles(inputA, 1, TileContext.shape(tm, tk));

        // declare the accumulator using the shapes describes as arguments
        var accumulator = TileOp.zeros(tm, tn);

        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }
        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tileMatmul(ComputeContext computeContext, Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of1D(M * N, tm * tn),
                () -> matmul(inputA, inputB, output, tm, tn, tk, M, N, numTiles));
    }

    private static void runSequential(Tensor2DF32 matrixA, Tensor2DF32 matrixB, Tensor2DF32 matrixC, final int size) {
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

    private void checkResult(Tensor2DF32 expected, Tensor2DF32 obtained) {
        for (int i = 0; i < expected.m(); i++) {
            for (int j = 0; j < obtained.n(); j++) {
                HATAsserts.assertEquals(expected.array(i * obtained.n() + j), obtained.array(i * obtained.n() + j), 0.01f);
            }
        }
    }

    @HatTest
    public void test_hat_tile_02() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 1024;

        Tensor2DF32 matrixA = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixB = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixC = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixSeq = Tensor2DF32.create(accelerator, size, size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int j = 0; j < size * size; j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        final int tm = 64;
        final int tn = 64;
        final int tk = 16;
        final int numTiles = (size + tk - 1) / tk;
        accelerator.compute( (@Reflect Compute)computeContext -> {
            tileMatmul(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size, numTiles);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    // ================================================================================================================
    // Expressing Reductions
    // ================================================================================================================
    // This example launches one grid (1, 1, 1).
    @Reflect
    public static void tileReduction(TensorF32 input, TensorF32 output, final int tileSize) {

        // Obtain the number of tiles
        final int numTiles = TileOp.numTiles(input, 0, tileSize);

        // Initialize a tile
        var acc = TileOp.full(TileContext.shape(1), 0.0f);

        // Perform the sum for all blocks of tiles
        for (int i = 0; i < numTiles; i++) {
            // load tile
            var tileA = TileContext.load(input, i, tileSize);
            // Perform a sum over the tile
            var res = TileOp.sum(tileA, 0);
            // Store the result into the accumulator
            acc = TileOp.add(acc, res);
        }

        // Store the final accumulator into global memory
        TileContext.store(output, 0, acc);
    }

    @Reflect
    public static void tileReduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(1, tileSize), () -> tileReduction(input, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tileSize = 64;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, 1);

        // fill input
        Random r = new Random();
        for (int k = 0; k < size; k++) {
            input.array(k, r.nextFloat(1));
        }

        accelerator.compute( (@Reflect Compute)computeContext ->
                tileReduction(computeContext, input, result, tileSize));

        float acc = 0.0f;
        for (int k = 0; k < size; k++) {
            acc += input.array(k);
        }

        HATAsserts.assertEquals(acc, result.array(0), 0.01f);
    }

    // Matrix transpose example
    @Reflect
    public static void transposeKernel(Tensor2DF32 inputMatrix, Tensor2DF32 transposedMatrix, final int tm, final int tn) {
        // In this example we get a 2D block.
        // The block id 0 maps to a row from the input matrix.
        // the block id 1 maps to a column from the input matrix.
        final int bidx = TileContext.BIDX();
        final int bidy = TileContext.BIDY();

        // Load the tile with shape tm x tn into memory
        var inputTile = TileContext.load(inputMatrix, TileContext.index(bidx, bidy), TileContext.shape(tm, tn));

        // compute the transpose function.
        var transposedTile = TileOp.transpose(inputTile);

        // store the resulting transposedTile into global memory.
        // Note that the index used are swapped.
        TileContext.store(transposedMatrix, TileContext.index(bidy, bidx), transposedTile);
    }

    @Reflect
    public static void computeTransposeKernel(ComputeContext computeContext, Tensor2DF32 input, Tensor2DF32 output, final int M, final int N, final int tm, final int tn) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> transposeKernel(input, output, tm, tn));
    }

    @HatTest
    public void test_hat_tile_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int M = 1024;
        final int N = 512;
        final int tileSize = 128;

        Tensor2DF32 input = Tensor2DF32.create(accelerator, M, N);
        Tensor2DF32 result = Tensor2DF32.create(accelerator, N, M);

        Random r = new Random(19);
        for (int i = 0; i < input.m(); i++) {
            for (int j = 0; j < input.n(); j++) {
                input.array((long) i * input.n() + j, r.nextFloat());
            }
        }

        // Launch kernel
        accelerator.compute( (@Reflect Compute) computeContext ->
                computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));

        // Check results
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                HATAsserts.assertEquals(input.array(i * N + j), result.array(j * M + i), 0.00f);
            }
        }
    }

    @Reflect
    public static void matmulF16(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int num_tiles) {

        // Calculate bidx and bidy using swizzle
        final int bid = TileContext.BIDX();
        final int num_bid_m = TileOp.ceildiv(M, tm);
        final int num_bid_n = TileOp.ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = TileOp.min(num_bid_m - first_bid_m, GROUP_SIZE_M);
        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / group_size_m;

        // Calculate the total number of tiles
        //final int num_tiles = TileOp.numTiles(inputA, 1, TileContext.shape(tm, tk));

        // declare the accumulator using the shapes describes as arguments
        var accumulator = TileOp.zeros(tm, tn);

        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }

        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulF16(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of1D(M * N, tm * tn),
                () -> matmulF16(inputA, inputB, output, tm, tn, tk, M, N, numTiles));
    }

    private static void runSequential(Tensor2DF16 matrixA, Tensor2DF16 matrixB, Tensor2DF32 matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    short a = matrixA.array((long) i * size + k);
                    short b = matrixB.array((long) k * size + j);
                    sum += Float.float16ToFloat(a) * Float.float16ToFloat(b);
                }
                matrixC.array((long) i * size + j, sum);
            }
        }
    }

    @HatTest
    public void test_hat_tile_05() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 1024;

        Tensor2DF16 matrixA = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF16 matrixB = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF32 matrixC = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixSeq = Tensor2DF32.create(accelerator, size, size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int i = 0; i < size * size; i++) {
            matrixA.array(i, Float.floatToFloat16(r.nextFloat()));
            matrixB.array(i, Float.floatToFloat16(r.nextFloat()));
        }

        final int tm = 64;
        final int tn = 64;
        final int tk = 16;
        final int numTiles = (size + tk -1) / tk;
        accelerator.compute( (@Reflect Compute)computeContext -> {
            matmulF16(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size, numTiles);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void matmulSimple(Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int num_tiles) {
        final int bidx = TileContext.BIDX();
        final int bidy = TileContext.BIDY();
        var accumulator = TileOp.zeros(tm, tn);
        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }
        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tileMatmulSimple(ComputeContext computeContext, Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimple(inputA, inputB, output, tm, tn, tk, numTiles));
    }

    @HatTest
    public void test_hat_tile_06() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        // Testing square matrices
        final int size = 1024;
        Tensor2DF32 matrixA = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixB = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixC = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixSeq = Tensor2DF32.create(accelerator, size, size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int j = 0; j < size * size; j++) {
            matrixA.array(j, r.nextFloat());
            matrixB.array(j, r.nextFloat());
        }

        final int tm = 32;
        final int tn = 64;
        final int tk = 64;
        final int numTiles = (size + tk - 1) / tk;
        accelerator.compute( (@Reflect Compute)computeContext -> {
            tileMatmulSimple(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size, numTiles);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void matmulSimpleF16(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int num_tiles) {
        final int bidx = TileContext.BIDX();
        final int bidy = TileContext.BIDY();
        var accumulator = TileOp.zeros(tm, tn);
        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }
        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulSimpleF16(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimpleF16(inputA, inputB, output, tm, tn, tk, numTiles));
    }

    @HatTest
    public void test_hat_tile_07() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        // Testing square matrices
        final int size = 1024;
        Tensor2DF16 matrixA = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF16 matrixB = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF32 matrixC = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixSeq = Tensor2DF32.create(accelerator, size, size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int i = 0; i < size * size; i++) {
            matrixA.array(i, Float.floatToFloat16(r.nextFloat()));
            matrixB.array(i, Float.floatToFloat16(r.nextFloat()));
        }

        final int tm = 32;
        final int tn = 64;
        final int tk = 64;
        final int numTiles = (size + tk - 1) / tk;
        accelerator.compute( (@Reflect Compute)computeContext -> {
            matmulSimpleF16(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size, numTiles);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @HatTest
    public void test_hat_tile_08() {
        final int M = 1024;
        final int N = 64;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        Tensor2DF16 matrixA = Tensor2DF16.create(accelerator, M, N);

        HATAsserts.assertEquals(M, matrixA.m());
        HATAsserts.assertEquals(N, matrixA.n());
    }

    @Reflect
    public static void matmulSimpleF16IRange(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int num_tiles) {
        final int bidx = TileContext.BIDX();
        final int bidy = TileContext.BIDY();
        var accumulator = TileOp.zeros(tm, tn);
        for(int k : TileContext.irange(0, num_tiles)) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }
        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulSimpleF16IRange(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimpleF16IRange(inputA, inputB, output, tm, tn, tk, numTiles));
    }

    @HatTest
    public void test_hat_tile_09() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        // Testing square matrices
        final int size = 1024;
        Tensor2DF16 matrixA = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF16 matrixB = Tensor2DF16.create(accelerator, size, size);
        Tensor2DF32 matrixC = Tensor2DF32.create(accelerator, size, size);
        Tensor2DF32 matrixSeq = Tensor2DF32.create(accelerator, size, size);

        // Initialize matrices (A and B have the same size)
        Random r = new Random(19);
        for (int i = 0; i < size * size; i++) {
            matrixA.array(i, Float.floatToFloat16(r.nextFloat()));
            matrixB.array(i, Float.floatToFloat16(r.nextFloat()));
        }

        final int tm = 32;
        final int tn = 64;
        final int tk = 64;
        final int numTiles = (size + tk - 1) / tk;
        accelerator.compute( (@Reflect Compute)computeContext -> {
            matmulSimpleF16IRange(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size, numTiles);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void partialReduction(TensorF32 input, TensorF32 output, final int tileSize) {

        // Obtain the block-thread ID
        final int pid = TileContext.BIDX();

        // Perform the sum for all blocks of tiles
        var tileA = TileContext.load(input, pid, tileSize);

        // Perform a sum over the tile
        var partial = TileOp.sum(tileA, 0);

        // Store the partial result into global memory
        TileContext.store(output, pid, partial);
    }

    @Reflect
    public static void partialReduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(input.m(), tileSize), () -> partialReduction(input, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_10() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tileSize = 64;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, tileSize);

        // fill input
        Random r = new Random();
        for (int k = 0; k < size; k++) {
            input.array(k, r.nextFloat(1));
        }

        accelerator.compute( (@Reflect Compute)computeContext ->
                partialReduction(computeContext, input, result, tileSize));

        // Check CPU implementation
        float acc = 0.0f;
        for (int k = 0; k < size; k++) {
            acc += input.array(k);
        }

        // Sum-up the partial results
        float accResult = 0.0f;
        for (int k = 0; k < result.m(); k++) {
            accResult += result.array(k);
        }

        HATAsserts.assertEquals(acc, accResult, 0.01f);
    }
}
