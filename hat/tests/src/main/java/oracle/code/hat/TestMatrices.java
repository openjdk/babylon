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
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import oracle.code.hat.annotation.HatTest;
import oracle.code.hat.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

public class TestMatrices {

    @CodeReflection
    public static void matrixMultiplyKernel2D(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {
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
    public static void matrixMultiplyKernel2DLI(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {
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
    public static void matrixMultiplyKernel2DTiling(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {

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
    public static float compute(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, int size, int j) {
        float acc = 0.0f;
        for (int k = 0; k < size; k++) {
            acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + j));
        }
        return acc;
    }

    @CodeReflection
    public static void matrixMultiplyKernel1D(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {
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
    public static void matrixMultiplyKernel1DWithFunctionCalls(@MappableIface.RO KernelContext kc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            for (int j = 0; j < size; j++) {
                float acc = compute(kc, matrixA, matrixB, size, j);
                matrixC.array(kc.x * size + j, acc);
            }
        }
    }

    @CodeReflection
    public static void matrixMultiply1D(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(globalSize));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel1D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    final static int BLOCK_SIZE = 16;

    @CodeReflection
    public static void matrixMultiply1DWithFunctionCalls(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int size) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh1D(size));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel1DWithFunctionCalls(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2D(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh2D(globalSize, globalSize), new LocalMesh2D(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel2D(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DLI(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int globalSize) {
        ComputeRange computeRange = new ComputeRange(new GlobalMesh2D(globalSize, globalSize), new LocalMesh2D(BLOCK_SIZE, BLOCK_SIZE));
        cc.dispatchKernel(computeRange,
                kc -> matrixMultiplyKernel2DLI(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DTiling(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.RW F32Array matrixC, int globalSize) {
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
        System.out.println(accelerator);

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
                TestMatrices.matrixMultiply1D(cc, matrixA, matrixB, matrixC, size));

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
        System.out.println(accelerator);

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
                TestMatrices.matrixMultiply1DWithFunctionCalls(cc, matrixA, matrixB, matrixC, size));

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
        System.out.println(accelerator);

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
                TestMatrices.matrixMultiply2D(cc, matrixA, matrixB, matrixC, size));

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
        System.out.println(accelerator);

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
                TestMatrices.matrixMultiply2DLI(cc, matrixA, matrixB, matrixC, size));

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
        System.out.println(accelerator);

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
                TestMatrices.matrixMultiply2DTiling(cc, matrixA, matrixB, matrixC, size));

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                HatAsserts.assertEquals(resultSeq.array(i * size + j), matrixC.array(i * size + j), 0.01f);
            }
        }
    }
}
