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
import hat.buffer.TensorF32;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import hat.types.Tile;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.TileContext.BIDX;
import static hat.TileContext.load;
import static hat.TileContext.store;
import static hat.types.Tile.add;

/**
 * How to test?
 *
 * <p>
 * <code>java @.ffi-cuda-test hat.test.TestTileMultiFunctions</code>
 * </p>
 */
public class TestTileMultiFunctions {

    @Reflect
    public static Tile compute(Tile tileA, Tile tileB) {
        return add(tileA, tileB);
    }

    @Reflect
    public static void multipleKernels(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        final var pid = BIDX();
        var tileA = load(inputA, pid, tileSize);
        var tileB = load(inputB, pid, tileSize);
        var result = compute(tileA, tileB);
        store(output, pid, result);
    }

    @Reflect
    public static void multipleKernels(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.m(), tileSize),
                () -> multipleKernels(inputA, inputB, output, tileSize));
    }

    /**
     * Test:
     *     multipleKernels -> compute
     */
    @HatTest
    public void test_hat_multi_function01() {
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
        accelerator.compute((@Reflect Compute) computeContext ->
                multipleKernels(computeContext, inputA, inputB, result, tileSize));

        for (int i = 0; i < size; i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }


    @Reflect
    public static Tile shared(Tile tileA, Tile tileB) {
        return add(tileA, tileB);
    }

    @Reflect
    public static Tile helperA(Tile tileA, Tile tileB) {
        return shared(tileA, tileB);
    }

    @Reflect
    public static Tile helperB(Tile tileA, Tile tileB) {
        return shared(tileA, tileB);
    }

    @Reflect
    public static void multipleHelpers(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        final var pid = BIDX();
        var tileA = load(inputA, pid, tileSize);
        var tileB = load(inputB, pid, tileSize);
        var r1 = helperA(tileA, tileB);
        var r2 = helperB(r1, tileB);
        store(output, pid, r2);
    }

    @Reflect
    public static void multipleHelpers(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.m(), tileSize),
                () -> multipleHelpers(inputA, inputB, output, tileSize));
    }

    /**
     * Test:
     *     multipleKernels -> helperA -> shared
     *                     -> helperB -> shared
     */
    @HatTest
    public void test_hat_multi_function02() {
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
        accelerator.compute((@Reflect Compute) computeContext ->
                multipleHelpers(computeContext, inputA, inputB, result, tileSize));

        for (int i = 0; i < size; i++) {
            var r1 = inputA.array(i) + inputB.array(i);
            var r2 = r1 + inputB.array(i);
            HATAsserts.assertEquals(r2, result.array(i), 0.01f);
        }
    }
}
