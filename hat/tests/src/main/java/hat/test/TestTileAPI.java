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
import hat.annotations.Kernel;
import hat.annotations.Preformatted;
import hat.backend.Backend;
import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF32;

import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.F16;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static optkl.ifacemapper.MappableIface.*;

/**
 * How to run?
 *
 * <p>
 *     Hello Tile Kernel
 *     <code>
 *         java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_00
 *     </code>
 * </p>
 *
 * <p>p
 *     To run the Vector Addition
 * <code>
 *  java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_01
 * </code>
 * </p>
 *
 * <p>
 *     Matrix Multiplication
 * <code>
 * java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_02
 * </code>
 * </p>
 *
 * <p>
 *     Reduction
 * <code>
 *   java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_03
 * </code>
 * </p>
 *
 * <p>
 *     Transpose Matrix
 * <code>
 *  java @.ffi-opencl-test hat.test.TestTileAPI#test_hat_tile_04
 * </code>
 * </p>
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
        final int bidy = (bid % num_bid_in_group) / num_bid_in_group;

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
    @Reflect
    public static void tileReduction(TensorF32 input, TensorF32 output, final int tileSize) {

        // Obtain the tile-id
        final int pid = TileContext.BIDX();

        // Obtain the number of tiles
        final int numTiles = TileOp.numTiles(input, 0, tileSize);

        // Initialize a tile
        var acc = TileOp.full(TileContext.shape(1), 0.0f);

        // Perform the sum for all blocks of tiles
        for (int i = 0; i < numTiles; i++) {
            // load tile
            var tileA = TileContext.load(input, pid, tileSize);
            // Perform a sum over the tile
            var res = TileOp.sum(tileA, 0);
            // Store the result into the accumulator
            acc = TileOp.add(acc, res);
        }

        // Store the final result into global memory
        TileContext.store(output, 0, acc);
    }

    @Reflect
    public static void tileReduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(input.m(), tileSize),
                () -> tileReduction(input, output, tileSize));
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

        // Load the tile with shape tm x tn into memory (e.g., registers, shared memory, or tensor memory)_
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

        final int M = 2048;
        final int N = 512;
        final int tileSize = 128;

        Tensor2DF32 input = Tensor2DF32.create(accelerator, M, N);
        Tensor2DF32 result = Tensor2DF32.create(accelerator, M, N);

        // Launch kernel
        accelerator.compute( (@Reflect Compute) computeContext ->
                computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));

        // Check results
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                HATAsserts.assertEquals(input.array(i * N + j), result.array(j * N + i), 0.01f);
            }
        }
    }

    @Preformatted("""
            typedef struct Tensor2DF16_s{
                int m;
                int n;
                unsigned char pad$1hAbP[8];
                half array[1];
            }Tensor2DF16_t;

            typedef struct Tensor2DF32_s{
                int m;
                int n;
                unsigned char pad$S9b4s[8];
                float array[1];
            }Tensor2DF32_t;
            """)
    @Kernel("""
            HAT_KERNEL void matmulF16(
                const HAT_GLOBAL_MEM Tensor2DF16_t* __restrict__ inputA,
                const HAT_GLOBAL_MEM Tensor2DF16_t* __restrict__ inputB,
                HAT_GLOBAL_MEM Tensor2DF32_t* __restrict__ output
            ){
                auto tm = 16;
                auto inputA_ = ct::assume_aligned(inputA->array, 16_ic);
                auto inputB_ = ct::assume_aligned(inputB->array, 16_ic);
                auto output_ = ct::assume_aligned(output->array, 16_ic);

                auto M = 1024;
                auto N = 1024;
                auto mSize = ct::assume_divisible(M, 16_ic);
                auto nSize = ct::assume_divisible(N, 16_ic);
                auto kSize = ct::assume_divisible(M, 16_ic);

                auto tn = 64;
                auto tk = 64;
                auto num_tiles = 64;
                int GROUP_SIZE_M = 8;
                int bid = ct::bid().x;
                auto num_bid_m = ct::ceildiv(M, tm);
                auto num_bid_n = ct::ceildiv(N, tn);
                int num_bid_in_group = GROUP_SIZE_M*num_bid_n;
                int group_id = bid/num_bid_in_group;
                int first_bid_m = group_id*GROUP_SIZE_M;
                auto group_size_m = ct::min(num_bid_m-first_bid_m, GROUP_SIZE_M);
                int bidx = first_bid_m+bid%group_size_m;
                int bidy = (bid%num_bid_in_group)/num_bid_in_group;
                auto accumulator = ct::zeros<ct::tile<float, ct::shape<64, 64>>>();
                for(int k = 0; k<num_tiles; k=k+1){
                //for(auto k: ct::irange(0, num_tiles)){
                    auto tileA = ct::partition_view{ct::tensor_span{inputA_, ct::extents{mSize, kSize}},ct::shape{64_ic,16_ic}}.load_masked(bidx, k);
                    auto tileB = ct::partition_view{ct::tensor_span{inputB_, ct::extents{kSize, nSize}},ct::shape{16_ic,64_ic}}.load_masked(k, bidy);
                    accumulator=ct::mma(tileA, tileB, accumulator);
                }
                ct::partition_view{ct::tensor_span{output_, ct::extents{mSize, nSize}},ct::shape{64_ic,64_ic} }.store_masked(accumulator, bidx, bidy);
                return;
            }
            """)
    @Reflect
    public static void matmulF16(@RO Tensor2DF16 inputA,@RO  Tensor2DF16 inputB, @WO Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int num_tiles) {

        // Calculate bidx and bidy using swizzle
        final int bid = TileContext.BIDX();
        final int num_bid_m = TileOp.ceildiv(M, tm);
        final int num_bid_n = TileOp.ceildiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = TileOp.min(num_bid_m - first_bid_m, GROUP_SIZE_M);
        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / num_bid_in_group;

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
    public static void matmulF16(ComputeContext computeContext, @RO Tensor2DF16 inputA, @RO Tensor2DF16 inputB, @WO Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of1D(M * N, tm * tn),
                () -> matmulF16(inputA, inputB, output, tm, tn, tk, M, N, numTiles));
    }

    private static void runSequential(Tensor2DF16 matrixA, Tensor2DF16 matrixB, Tensor2DF32 matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0;
                for (int k = 0; k < size; k++) {
                    F16 a = matrixA.array((long) i * size + k);
                    F16 b = matrixB.array((long) k * size + j);
                    F16 mul = F16.mul(a, b);
                    sum += F16.f16ToFloat(mul);
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
        for (int j = 0; j < size * size; j++) {
            F16 valA = F16.floatToF16(r.nextFloat());
            F16 valB = F16.floatToF16(r.nextFloat());
            matrixA.array(j).value(valA.value());
            matrixB.array(j).value(valB.value());
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
    public static void matmulSimple(Tensor2DF32 inputA, Tensor2DF32 inputB, Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int num_tiles) {
        int bidx = TileContext.BIDX();
        int bidy = TileContext.BIDY();
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
                () -> matmulSimple(inputA, inputB, output, tm, tn, tk, M, N, numTiles));
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

    @Preformatted("""
            typedef struct Tensor2DF16_s{
                int m;
                int n;
                unsigned char pad$t9t1Y[8];
                half array[1];
            }Tensor2DF16_t;

            typedef struct Tensor2DF32_s{
                int m;
                int n;
                unsigned char pad$pIF4y[8];
                float array[1];
            }Tensor2DF32_t;
            """)
    @Kernel("""
            HAT_KERNEL void matmulSimpleF16(
                HAT_GLOBAL_MEM Tensor2DF16_t* __restrict__ inputA,
                HAT_GLOBAL_MEM Tensor2DF16_t* __restrict__ inputB,
                HAT_GLOBAL_MEM Tensor2DF32_t* __restrict__ output
            ){
                auto tm = 32;
                auto inputA_ = ct::assume_aligned(inputA->array, 16_ic);
                auto inputB_ = ct::assume_aligned(inputB->array, 16_ic);
                auto output_ = ct::assume_aligned(output->array, 16_ic);

                auto M = 1024;
                auto N = 1024;
                auto mSize = ct::assume_divisible(M, 16_ic);
                auto nSize = ct::assume_divisible(N, 16_ic);
                auto kSize = ct::assume_divisible(M, 16_ic);

                auto tn = 64;
                auto tk = 64;
                auto num_tiles = 16;
                int bidx = ct::bid().x;
                int bidy = ct::bid().y;
                auto accumulator = ct::zeros<ct::tile<float, ct::shape<32, 64>>>();
                //for(int k = 0; k<num_tiles; k=k+1){
                for(auto k : ct::irange(0, num_tiles)) {
                    auto tileA = ct::partition_view{ct::tensor_span{inputA_, ct::extents{mSize, kSize}},ct::shape{32_ic,64_ic}}.load_masked(bidx, k);
                    auto tileB = ct::partition_view{ct::tensor_span{inputB_, ct::extents{kSize, nSize}},ct::shape{64_ic,64_ic}}.load_masked(k, bidy);
                    accumulator=ct::mma(tileA, tileB, accumulator);
                }
                ct::partition_view{ct::tensor_span{output_, ct::extents{mSize, nSize}},ct::shape{32_ic,64_ic} }.store_masked(accumulator, bidx, bidy);
                return;
            }
            """)
    @Reflect
    public static void matmulSimpleF16(@RO Tensor2DF16 inputA, @RO Tensor2DF16 inputB, @WO Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int num_tiles) {
        int bidx = TileContext.BIDX();
        int bidy = TileContext.BIDY();
        var accumulator = TileOp.zeros(tm, tn);
        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(tm, tk));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(tk, tn));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }
        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void matmulSimpleF16(ComputeContext computeContext, @RO Tensor2DF16 inputA, @RO Tensor2DF16 inputB, @WO Tensor2DF32 output, final int tm, final int tn, final int tk, final int M, final int N, final int numTiles) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmulSimpleF16(inputA, inputB, output, tm, tn, tk, M, N, numTiles));
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
        for (int j = 0; j < size * size; j++) {
            F16 valA = F16.floatToF16(r.nextFloat());
            F16 valB = F16.floatToF16(r.nextFloat());
            matrixA.array(j).value(valA.value());
            matrixB.array(j).value(valB.value());
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
}
