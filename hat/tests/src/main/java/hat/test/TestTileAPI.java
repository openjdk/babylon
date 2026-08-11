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
        var tileResult = TileOp.add(aTile, bTile);  // TODO: we need to infer the shape of the resulting tile based on the operands
        TileContext.store(output, pid, tileResult);
    }

    @Reflect
    public static void computeEmptyTile(@RO ComputeContext computeContext, @RO TensorF32 inputA, @RO TensorF32 inputB, @WO TensorF32 output, @Constant int tile_size) {
        computeContext.dispatchTile(NDRange.of1D(inputA.length(), tile_size), () -> helloTile(inputA, inputB, output, tile_size));
    }

    @Reflect
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
        accelerator.compute( computeContext -> computeEmptyTile(computeContext, inputA, inputB, result, tile_size));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }


    // ================================================================================================================
    // Expressing Vector Addition
    // ================================================================================================================
    @Reflect
    public static void vectorAddTile(TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tile_size) {

        // Program id: get tile-id for 1D
        var pid = TileContext.BIDX();

        var a_tile = TileContext.load(inputA, pid, tile_size);
        var b_tile = TileContext.load(inputB, pid, tile_size);

        // This could be a tensor as well
        // var result = Tensor.add(a_tile, b_tile);
        var result = TileOp.add(a_tile, b_tile);

        TileContext.store(output, pid, result);
    }

    @Reflect
    public static void myComputeWithTile_vector_add(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tile_size) {
        computeContext.dispatchTile(NDRange.of1D(inputA.length(), tile_size),
                () -> vectorAddTile(inputA, inputB, output, tile_size));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_01() {
        // Prototyping vector addition version for tile programming in HAT

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tile_size = 64;

        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        accelerator.compute( computeContext ->
            myComputeWithTile_vector_add(computeContext, inputA, inputB, result, tile_size));
    }

    // ================================================================================================================
    // Expressing MatMul
    // ================================================================================================================
    public static final int GROUP_SIZE_M = 8;

    @Reflect
    public static void matmul(TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {

        // Calculate bidx and bidy using swizzle
        int bid = TileContext.BIDX();
        int num_bid_m = Math.ceilDiv(M, tm);
        int num_bid_n = Math.ceilDiv(N, tn);
        int num_bid_in_group = GROUP_SIZE_M * num_bid_n;    // IDEA: to get the constants in, we can do a pass over to transform this GROUP_SIZE_M (field access) into a Constant into the tree!
                                                            // We can implement a similar idea into the main HAT (thread-kernel mode).
        int group_id = bid / num_bid_in_group;
        int first_bid_m = group_id * GROUP_SIZE_M;
        int group_size_m = Math.min(num_bid_m - first_bid_m, GROUP_SIZE_M);

        int bidx = first_bid_m + (bid % group_size_m);
        int bidy = (bid % num_bid_in_group) / num_bid_in_group;

        // Calculate the total number of tiles
        int num_tiles = TileContext.num_tiles(inputA, 1, TileContext.shape(16,16));

        // Return type should be a TileData
        var accumulator = TileContext.zeros(tm, tk);

        for (int k = 0; k < num_tiles; k++) {
            var tileA = TileContext.load(inputA, TileContext.index(bidx, k), TileContext.shape(16, 16));
            var tileB = TileContext.load(inputB, TileContext.index(k, bidy), TileContext.shape(16, 16));
            accumulator = TileOp.mma(tileA, tileB, accumulator);
        }

        TileContext.store(output, TileContext.index(bidx, bidy), accumulator);
    }

    @Reflect
    public static void tileMatmul(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, @Constant int tm, @Constant int tn, @Constant int tk, @Constant int M, @Constant int N) {
        computeContext.dispatchTile(NDRange.of2D(M, N, tm, tn),
                () -> matmul(inputA, inputB, output, tm, tn, tk, M, N));
    }

    @Reflect
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

        accelerator.compute( computeContext -> {
            tileMatmul(computeContext, matrixA, matrixB, matrixC, tm, tn, tk, size, size);
        });
    }

    // ================================================================================================================
    // Expressing Reductions
    // ================================================================================================================
    @Reflect
    public static void tileReduction(TensorF32 input, TensorF32 output, @Constant int tile_size) {

        // Obtain the tile-id
        int pid = TileContext.BIDX();

        // Obtain the number of tiles
        int numTiles = TileContext.num_tiles(input, 0, TileContext.shape(tile_size));

        // Initialize a tile
        var acc = TileContext.full(TileContext.shape(1), 0);

        // Perform the sum for all blocks of tiles
        for (int i = 0; i < numTiles; i++) {
            // load tile
            var tileA = TileContext.load(input, TileContext.index(pid), TileContext.shape(tile_size));
            // Perform a sum over the tile
            var res = TileContext.sum(tileA, 0);
            // Store the result into the accumulator
            acc = TileOp.add(acc, res);
        }

        // Store the final result into global memory
        TileContext.store(output, TileContext.index(0), acc);
    }

    @Reflect
    public static void tileReduction(ComputeContext computeContext, TensorF32 input, TensorF32 output, @Constant int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(input.length(), tileSize),
                () -> tileReduction(input, output, tileSize));
    }

    @Reflect
    @HatTest
    public void test_hat_tile_03() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int size = Math.powExact(2, 12);
        final int tileSize = 64;

        TensorF32 input = TensorF32.create(accelerator, size);
        TensorF32 result = TensorF32.create(accelerator, size);

        accelerator.compute( computeContext ->
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

    @Reflect
    @HatTest
    public void test_hat_tile_04() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        final int M = 2048;
        final int N = 512;
        final int tileSize = 128;

        TensorF32 input = TensorF32.create(accelerator, M * N);
        TensorF32 result = TensorF32.create(accelerator, M * N);

        accelerator.compute( computeContext -> computeTransposeKernel(computeContext, input, result, M, N, tileSize, tileSize));
    }
}
