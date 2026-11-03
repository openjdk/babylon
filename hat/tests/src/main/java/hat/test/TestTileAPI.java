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
import hat.backend.Backend;
import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF32;

import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.Tile;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Random;

import static hat.TileContext.BIDX;
import static hat.TileContext.BIDY;
import static hat.TileContext.dimMap;
import static hat.TileContext.index;
import static hat.TileContext.load;
import static hat.TileContext.shape;
import static hat.TileContext.store;
import static hat.types.Tile.add;
import static hat.types.Tile.ceildiv;
import static hat.types.Tile.full;
import static hat.types.Tile.min;
import static hat.types.Tile.mma;
import static hat.types.Tile.numTiles;
import static hat.types.Tile.permute;
import static hat.types.Tile.reshape;
import static hat.types.Tile.sum;
import static hat.types.Tile.transpose;
import static hat.types.Tile.zeros;

/**
 * How to run?
 *
 * <p>
 * <code>
 * java @.ffi-cuda-test hat.test.TestTileAPI
 * </code>
 * </p>
 *
 */
public class TestTileAPI {

    private void initRandom(TensorF32 input) {
        Random r = new Random(71);
        for (int i = 0; i < input.m(); i++) {
            input.array(i, r.nextFloat());
        }
    }

    @Reflect
    public static void vectorAddTile(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        final var pid = BIDX();
        var tileA = load(inputA, pid, tileSize);
        var tileB = load(inputB, pid, tileSize);
        var result = add(tileA, tileB);
        store(output, pid, result);
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
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        TensorF32 result = TensorF32.create(accelerator, size);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));
        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));

        // change the tile size
        final int newTileSize = 16;
        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, newTileSize));

        // Alternate the tile size
        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));
        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, newTileSize));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    @HatTest
    public void test_hat_tile_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = Math.powExact(2, 16);
        final int tileSize = 64;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        // Fill data
        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        accelerator.compute((@Reflect Compute) computeContext -> vectorAddTile(computeContext, inputA, inputB, result, tileSize));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    public static final int GROUP_SIZE_M = 8;

    @Reflect
    public static void matmul(Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {

        // Calculate bidx and bidy using swizzle
        final int bid = BIDX();
        final int num_bid_m = ceildiv(M, tm);
        final int num_bid_n = ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / group_size_m;

        // Calculate the total number of tiles
        final int numberOfTiles = numTiles(inputA, 1, shape(tm, tk));

        // declare the accumulator using the shapes describes as arguments
        var accumulator = zeros(tm, tn);

        for (int k = 0; k < numberOfTiles; k++) {
            var tileA = load(inputA, index(bidx, k), shape(tm, tk));
            var tileB = load(inputB, index(k, bidy), shape(tk, tn));
            accumulator = mma(tileA, tileB, accumulator);
        }
        store(output, index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tileMatmul(ComputeContext computeContext, Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {

        var range = NDRange.NDRange2D.of(
                NDRange.Global2D.of(M, N),
                NDRange.Local2D.of(tm, tn),
                NDRange.Tile2D.of(tm, tn),
                NDRange.Warp2D.of(true, false));

        computeContext.dispatchTile(range, () -> matmul(inputA, inputB, output, tm, tn, tk, M, N));
    }

    private static void runSequential(Tensor2DF32 matrixA, Tensor2DF32 matrixB, Tensor2DF32 matrixC, final int size) {
        for (int i = 0; i < matrixA.m(); i++) {
            for (int j = 0; j < matrixB.n(); j++) {
                float sum = 0;
                for (int k = 0; k < matrixA.n(); k++) {
                    float a = matrixA.array((long) i * matrixA.n() + k);
                    float b = matrixB.array((long) k * matrixB.n() + j);
                    sum += a * b;
                }
                matrixC.array((long) i * size + j, sum);
            }
        }
    }

    private void checkResult(Tensor2DF32 expected, Tensor2DF32 actual) {
        HATAsserts.assertEquals(expected.m(), actual.m());
        HATAsserts.assertEquals(expected.n(), actual.n());
        for (int i = 0; i < expected.m(); i++) {
            for (int j = 0; j < expected.n(); j++) {
                HATAsserts.assertEquals(expected.array((long) i * actual.n() + j), actual.array((long) i * actual.n() + j), 0.01f);
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
        accelerator.compute((@Reflect Compute) computeContext -> {
            tileMatmul(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    // This example launches one grid (1, 1, 1).
    @Reflect
    public static void tileReduction(TensorF32 input, TensorF32 output, final int tileSize) {

        // Obtain the number of tiles
        final int numTiles = numTiles(input, 0, tileSize);

        // Initialize a tile
        var acc = full(shape(1), 0.0f);

        // Perform the sum for all blocks of tiles
        for (int i = 0; i < numTiles; i++) {
            // load tile
            var tileA = load(input, i, tileSize);
            // Perform a sum over the tile
            var res = sum(tileA, 0);
            // Store the result into the accumulator
            acc = add(acc, res);
        }

        // Store the final accumulator into global memory
        store(output, 0, acc);
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
        Random r = new Random(71);
        for (int k = 0; k < size; k++) {
            input.array(k, r.nextFloat(1));
        }

        accelerator.compute((@Reflect Compute) computeContext ->
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
        final int bidx = BIDX();
        final int bidy = BIDY();

        // Load the tile with shape tm x tn into memory
        var inputTile = load(inputMatrix, index(bidx, bidy), shape(tm, tn));

        // compute the transpose function.
        var transposedTile = transpose(inputTile);

        // store the resulting transposedTile into global memory.
        // Note that the index used are swapped.
        store(transposedMatrix, index(bidy, bidx), transposedTile);
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
        accelerator.compute((@Reflect Compute) computeContext ->
                computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));

        // Check results
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                HATAsserts.assertEquals(input.array(i * N + j), result.array(j * M + i), 0.00f);
            }
        }
    }

    @Reflect
    public static void matmulF16(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {

        // Calculate bidx and bidy using swizzling
        final int bid = BIDX();
        final int num_bid_m = ceildiv(M, tm);
        final int num_bid_n = ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M);
        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / group_size_m;

        // Calculate the total number of tiles
        final int numberOfTiles = numTiles(inputA, 1, shape(tm, tk));

        // declare the accumulator using the shapes describes as arguments
        var accumulator = zeros(tm, tn);
        for (int k = 0; k < numberOfTiles; k++) {
            var tileA = load(inputA, index(bidx, k), shape(tm, tk));
            var tileB = load(inputB, index(k, bidy), shape(tk, tn));
            accumulator = mma(tileA, tileB, accumulator);
        }

        store(output, index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulF16(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {

        var range = NDRange.NDRange2D.of(
                NDRange.Global2D.of(M, N),
                NDRange.Local2D.of(tm, tn),
                NDRange.Tile2D.of(tm, tn),
                NDRange.Warp2D.of(true, false));

        computeContext.dispatchTile(range, () -> matmulF16(inputA, inputB, output, tm, tn, tk, M, N));
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
        accelerator.compute((@Reflect Compute) computeContext -> {
            matmulF16(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void matmulSimple(Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk) {
        final int bidx = BIDX();
        final int bidy = BIDY();
        var accumulator = zeros(tm, tn);
        final int numberOfTiles = numTiles(inputA, 1, shape(tm, tk));
        for (int k = 0; k < numberOfTiles; k++) {
            var tileA = load(inputA, index(bidx, k), shape(tm, tk));
            var tileB = load(inputB, index(k, bidy), shape(tk, tn));
            accumulator = mma(tileA, tileB, accumulator);
        }
        store(output, index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tileMatmulSimple(ComputeContext computeContext, Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimple(inputA, inputB, output, tm, tn, tk));
    }

    @HatTest
    public void test_hat_tile_06() {
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

        final int tm = 32;
        final int tn = 64;
        final int tk = 64;
        accelerator.compute((@Reflect Compute) computeContext -> {
            tileMatmulSimple(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void matmulSimpleF16(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk) {
        final int bidx = BIDX();
        final int bidy = BIDY();
        var accumulator = zeros(tm, tn);
        final int numberOfTiles = numTiles(inputA, 1, shape(tm, tk));
        for (int k = 0; k < numberOfTiles; k++) {
            var tileA = load(inputA, index(bidx, k), shape(tm, tk));
            var tileB = load(inputB, index(k, bidy), shape(tk, tn));
            accumulator = mma(tileA, tileB, accumulator);
        }
        store(output, index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulSimpleF16(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimpleF16(inputA, inputB, output, tm, tn, tk));
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

        final int tm = 64;
        final int tn = 64;
        final int tk = 64;
        accelerator.compute((@Reflect Compute) computeContext -> {
            matmulSimpleF16(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
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

    /**
     * matmul in FP16 using swizzling and irange.
     */
    @Reflect
    public static void matmulSimpleF16IRangeSwizzling(Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int M, final int N, final int tm, final int tn, final int tk) {
        // Calculate bidx and bidy using swizzle
        final int bid = BIDX();
        final int num_bid_m = ceildiv(M, tm);
        final int num_bid_n = ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / group_size_m;

        var accumulator = zeros(tm, tn);
        final int numberOfTiles = numTiles(inputA, 1, shape(tm, tk));
        for (int k = 0; k < numberOfTiles; k++) {
            var tileA = load(inputA, index(bidx, k), shape(tm, tk));
            var tileB = load(inputB, index(k, bidy), shape(tk, tn));
            accumulator = mma(tileA, tileB, accumulator);
        }
        store(output, index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulSimpleF16IRangeSwizzling(ComputeContext computeContext, Tensor2DF16 inputA, Tensor2DF16 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N) {
        var range = NDRange.NDRange2D.of(
                NDRange.Global2D.of(M, N),
                NDRange.Local2D.of(tm, tn),
                NDRange.Tile2D.of(tm, tn),
                NDRange.Warp2D.of(true, false));
        computeContext.dispatchTile(range, () -> matmulSimpleF16IRangeSwizzling(inputA, inputB, output, M, N, tm, tn, tk));
    }

    // Test matmul in FP16 using swizzling and irange.
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

        final int tm = 64;
        final int tn = 64;
        final int tk = 64;
        accelerator.compute((@Reflect Compute) computeContext ->
                matmulSimpleF16IRangeSwizzling(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size));

        runSequential(matrixA, matrixB, matrixSeq, size);
        checkResult(matrixSeq, matrixC);
    }

    @Reflect
    public static void partialReduction(TensorF32 input, TensorF32 output, final int tileSize) {

        // Obtain the block-thread ID
        final int pid = BIDX();

        // Perform the sum for all blocks of tiles
        var tileA = load(input, pid, tileSize);

        // Perform a sum over the tile
        var partial = sum(tileA, 0);

        // Store the partial result into global memory
        store(output, pid, partial);
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

        final int numTiles = (size + tileSize - 1) / tileSize;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, numTiles);

        // fill input
        Random r = new Random(71);
        for (int k = 0; k < size; k++) {
            input.array(k, r.nextFloat(1));
        }

        accelerator.compute((@Reflect Compute) computeContext ->
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

    @Reflect
    public static void vectorAddTileWithConstant(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        final var pid = BIDX();
        var tileA = load(inputA, pid, tileSize * 4);
        var tileB = load(inputB, pid, tileSize * 4);
        var result = add(tileA, tileB);
        store(output, pid, result);
    }

    @Reflect
    public static void vectorAddTileWithConstant(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.m(), tileSize),
                () -> vectorAddTileWithConstant(inputA, inputB, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_11() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        final int tileSize = 32 / 4;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);
        initRandom(inputA);
        initRandom(inputB);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute((@Reflect Compute) computeContext ->
                vectorAddTileWithConstant(computeContext, inputA, inputB, result, tileSize));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    @Reflect
    public static void reshapeAndPermute01(TensorF32 inputMatrix, TensorF32 output, final int tileSize) {
        final int pid = BIDX();

        // 1D shape if size: tileSize
        var tile = load(inputMatrix, pid, tileSize);

        // Reshape a 1D tile to a 2D tile with shape 2x4
        Tile reshape = reshape(tile, shape(4, 2));
        // To follow this example, let's consider the following input tile:
        // tile = [0, 1, 2, 3, 4, 5, 6, 7]
        // Then, after reshape with shape: [2, 4], we obtain the following tile:
        // 0, 1,
        // 2, 3,
        // 4, 5,
        // 6, 7,

        // Permute rows <-> columns
        Tile permute = permute(reshape, dimMap(1, 0));
        // In this case, the tile with shape [4, 2] is permuted to [1, 0]
        // After permute with 1, 0
        // 0, 2, 4, 6,
        // 1, 3, 5, 7

        // Flatten again to pass the result to a 1D Tensor
        Tile flattened = reshape(permute, shape(8));
        // from tile[1, 0 ] => tile[8]
        // After it gets flatten
        // [ 0, 2, 4, 6, 1, 3, 5, 7 ]

        // Store the final flattened tile into the output tensor buffer
        store(output, pid, flattened);
    }

    @Reflect
    public static void reshapeAndPermute01(ComputeContext computeContext, TensorF32 inputMatrix, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputMatrix.m(), tileSize), () -> reshapeAndPermute01(inputMatrix, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_12() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        // We keep tileSize to 8 for simplification for the output checks
        final int tileSize = 8;
        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 output = TensorF32.create(accelerator, size);
        initRandom(input);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute((@Reflect Compute) computeContext -> reshapeAndPermute01(computeContext, input, output, tileSize));

        // output must be identical to the input
        List<Integer> indicesAfterTransform = List.of(0, 2, 4, 6, 1, 3, 5, 7);
        for (int i = 0; i < size; i += tileSize) {
            for (int j = 0; j < tileSize; j++) {
                HATAsserts.assertEquals(input.array(i + indicesAfterTransform.get(j)), output.array(i + j), 0.01f);
            }
        }
    }

    @Reflect
    public static void reshapeAndPermute02(TensorF32 inputMatrix, TensorF32 output, final int tileSize) {
        final int pid = BIDX();

        // 1D shape if size: tileSize
        var tile = load(inputMatrix, pid, tileSize);

        Tile reshape = reshape(tile, shape(1, 2, 4));
        // tensor reshapes from tile[tileSize] -> tile[1, 2, 4]
        // The HAT runtime checks for compatibility of the new shape.

        // To follow this example, let's consider the following input tile:
        // tile = [0, 1, 2, 3, 4, 5, 6, 7]
        // Then, after reshape with shape: [1, 2, 4], we obtain the following tile:
        // 0, 1, 2, 3,
        // 4, 5, 6, 7,

        Tile permute = permute(reshape, dimMap(2, 0, 1));
        // The permute operation permutes the axes of the input tile.
        // In this case, the tile with shape [1, 2, 4] is permuted to [4, 1, 2]
        // After permute with 4, 1, 2
        // 0, 4,
        // 1, 5,
        // 2, 6,
        // 3, 7

        Tile flattened = reshape(permute, shape(tileSize));
        // Reshape again to flat the tile from 3D to 1D
        // from tile[4, 1, 2] => tile[8]
        // After it gets flatten
        // [ 0, 4, 1, 5, 2, 6, 3, 7 ]
        // Thus, these are the indexes to check for the output

        // Store the final flattened tile into the output tensor buffer
        store(output, pid, flattened);
    }

    @Reflect
    public static void reshapeAndPermute02(ComputeContext computeContext, TensorF32 inputMatrix, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputMatrix.m(), tileSize), () -> reshapeAndPermute02(inputMatrix, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_13() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        // We keep tileSize to 8 for simplification for the output checks
        final int tileSize = 8;
        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 output = TensorF32.create(accelerator, size);
        initRandom(input);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute((@Reflect Compute) computeContext -> reshapeAndPermute02(computeContext, input, output, tileSize));

        // output must be identical to the input
        List<Integer> indicesAfterTransform = List.of(0, 4, 1, 5, 2, 6, 3, 7);
        for (int i = 0; i < size; i += tileSize) {
            for (int j = 0; j < tileSize; j++) {
                HATAsserts.assertEquals(input.array(i + indicesAfterTransform.get(j)), output.array(i + j), 0.01f);
            }
        }
    }
}
