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
import hat.Constant;
import hat.NDRange;
import hat.TileContext;
import hat.TileOp;
import hat.backend.Backend;
import hat.buffer.TensorF32;

import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

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

    @Reflect
    public static void helloTile(@RO TensorF32 inputA, @RO TensorF32 inputB, @WO TensorF32 output, @Constant int tile_size) {
        final var pid = TileContext.BIDX();
        var aTile = TileContext.load(inputA, pid, 16);
        var bTile = TileContext.load(inputB, pid, 16);
        var tileResult = TileOp.add(aTile, bTile);
        TileContext.store(output, pid, tileResult);
    }

    @Reflect
    public static void computeEmptyTile(@RO ComputeContext computeContext, @RO TensorF32 inputA, @RO TensorF32 inputB, @WO TensorF32 output, @Constant int tile_size) {
        computeContext.dispatchTile(NDRange.of1D(inputA.length(), tile_size), () -> helloTile(inputA, inputB, output, tile_size));
    }

    @HatTest
    public void test_hat_tile_00() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;
        final int tile_size = 16;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);

        // Fill data
        Random r = new Random();
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        TensorF32 result = TensorF32.create(accelerator, size);
        accelerator.compute( (@Reflect Compute)computeContext -> computeEmptyTile(computeContext, inputA, inputB, result, tile_size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }

    // ================================================================================================================
    // Expressing Vector Addition
    // ================================================================================================================
    @Reflect
    public static void vectorAddTile(TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tileSize) {
        final var pid = TileContext.BIDX();
        var tileA = TileContext.load(inputA, pid, tileSize);
        var tileB = TileContext.load(inputB, pid, tileSize);
        var result = TileOp.add(tileA, tileB);
        TileContext.store(output, pid, result);
    }

    @Reflect
    public static void myComputeWithTile_vector_add(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.length(), tileSize), () -> vectorAddTile(inputA, inputB, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = Math.powExact(2, 12);
        final int tile_size = 64;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);
        accelerator.compute( (@Reflect Compute)computeContext ->
            myComputeWithTile_vector_add(computeContext, inputA, inputB, result, tile_size));
    }

    // ================================================================================================================
    // Expressing MatMul
    // ================================================================================================================
    public static final int GROUP_SIZE_M = 8;

    @Reflect
    public static void matmul(TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {

        final int GROUP_SIZE_M = 8;
        // Calculate bidx and bidy using swizzle
        final int bid = TileContext.BIDX();
        final int num_bid_m = Math.ceilDiv(M, tm);
        final int num_bid_n = Math.ceilDiv(N, tn);
        final int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        final int group_id = bid / num_bid_in_group;
        final int first_bid_m = group_id * GROUP_SIZE_M;
        final int group_size_m = Math.min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        final int bidx = first_bid_m + (bid % group_size_m);
        final int bidy = (bid % num_bid_in_group) / num_bid_in_group;

        // Calculate the total number of tiles
        final int num_tiles = TileOp.numTiles(inputA, 1, TileContext.shape(tm, tk));

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
    public static void tileMatmul(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmul(inputA, inputB, output, tm, tn, tk, M, N));
    }

    @HatTest
    public void test_hat_tile_02() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = 1024;

        TensorF32 matrixA = TensorF32.create(accelerator, size * size);
        TensorF32 matrixB = TensorF32.create(accelerator, size * size);
        TensorF32 matrixC = TensorF32.create(accelerator, size * size);

        int tm = 64;
        int tn = 64;
        int tk = 64;

        accelerator.compute( (@Reflect Compute)computeContext -> {
            tileMatmul(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });
    }

    // ================================================================================================================
    // Expressing Reductions
    // ================================================================================================================
    @Reflect
    public static void tileReduction(TensorF32 input, TensorF32 output, @Constant int tileSize) {

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
    public static void tileReduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, @Constant int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(input.length(), tileSize),
                () -> tileReduction(input, output, tileSize));
    }

    @HatTest
    public void test_hat_tile_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tileSize = 64;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        accelerator.compute( (@Reflect Compute)computeContext ->
                tileReduction(computeContext, input, result, tileSize));
    }

    // Matrix transpose example
    @Reflect
    public static void transposeKernel(TensorF32 inputMatrix, TensorF32 transposedMatrix, @Constant int tm, @Constant int tn) {
        // In this example we get a 2D block.
        // The block id 0 maps to a row from the input matrix.
        // the block id 1 maps to a column from the input matrix.
        int bidx = TileContext.BIDX();
        int bidy = TileContext.BIDY();

        // Load the tile with shape tm x tn into memory (e.g., registers, shared memory, or tensor memory)_
        var inputTile = TileContext.load(inputMatrix, TileContext.index(bidx, bidy), TileContext.shape(128, 128));

        // compute the transpose function.
        var transposedTile = TileOp.transpose(inputTile);

        // store the resulting transposedTile into global memory.
        // Note that the index used are swapped.
        TileContext.store(transposedMatrix, TileContext.index(bidy, bidx), transposedTile);
    }

    @Reflect
    public static void computeTransposeKernel(ComputeContext computeContext, TensorF32 input, TensorF32 output, @Constant int M, @Constant int N, @Constant int tm, @Constant int tn) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> transposeKernel(input, output, tm, tn));
    }

    @HatTest
    public void test_hat_tile_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int M = 2048;
        final int N = 512;
        final int tileSize = 128;

        TensorF32 input = TensorF32.create(accelerator, M * N);
        TensorF32 result = TensorF32.create(accelerator, M * N);

        accelerator.compute( (@Reflect Compute) computeContext -> computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));
    }
}
