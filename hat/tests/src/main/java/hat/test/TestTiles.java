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
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.TensorF32;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.TileContext.BIDX;
import static hat.TileContext.load;
import static hat.TileContext.store;

import static hat.Accelerator.Compute;
import static hat.TileOp.add;

/**
 * <p>How to run?</p>
 *
 * <p><code>java @.ffi-cuda-test hat.test.TestTiles</code></p>
 */
public class TestTiles {

    @Reflect
    public static void vectorAddTile(TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        store(output, BIDX(), add(load(inputA, BIDX(), tileSize), load(inputB, BIDX(), tileSize)));
    }

    @Reflect
    public static void vectorAddTile(ComputeContext computeContext, TensorF32 inputA, TensorF32 inputB, TensorF32 output, final int tileSize) {
        computeContext.dispatchTile(NDRange.of1D(inputA.m(), tileSize),
                () -> vectorAddTile(inputA, inputB, output, tileSize));
    }

    @HatTest
    public void testSingleLine() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = Math.powExact(2, 16);
        final int tileSize = 512;
        TensorF32 inputA = TensorF32.create(accelerator, size);
        TensorF32 inputB = TensorF32.create(accelerator, size);

        // Fill data
        Random r = new Random(19);
        for (int i = 0; i < size; i++) {
            inputA.array(i, r.nextFloat());
            inputB.array(i, r.nextFloat());
        }

        TensorF32 result = TensorF32.create(accelerator, size);

        // Invoking the kernel multiple times to check the code cache
        accelerator.compute((@Reflect Compute) computeContext ->
                vectorAddTile(computeContext, inputA, inputB, result, tileSize));

        for (int i = 0; i < result.m(); i++) {
            HATAsserts.assertEquals((inputA.array(i) + inputB.array(i)), result.array(i), 0.01f);
        }
    }
}
