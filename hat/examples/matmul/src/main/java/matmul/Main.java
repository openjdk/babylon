/*
 * Copyright (c) 2024-2025, Oracle and/or its affiliates. All rights reserved.
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
package matmul;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;

import jdk.incubator.code.CodeReflection;

import java.util.Random;

import static hat.ifacemapper.MappableIface.RO;
import static hat.ifacemapper.MappableIface.RW;

/**
 * Canonical example for Matrix Multiply.
 *
 * <p>How to run?</p>
 *
 * <p>For 2D Configuration:
 * <code>
 *     java @hat/run ffi-opencl matmul 2D
 * </code>
 * </p>
 *
 * <p> For 1D Configuration
 *     <code>
 *         java @hat/run ffi-opencl matmul 1D
 *     </code>
 * </p>
 */
public class Main {

    private static final boolean CHECK_RESULT = true;

    private static final int NUM_ITERATIONS = 10;

    /**
     * Naive Matrix Multiplication implemented in 2D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @CodeReflection
    public static void matrixMultiplyKernel2D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            if (kc.y < kc.maxY) {
                float acc = 0;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + kc.y));
                }
                matrixC.array(kc.x * size + kc.y, acc);
            }
        }
    }

    /**
     * Naive Matrix Multiplication implemented in 2D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @CodeReflection
    public static void matrixMultiplyKernel2DLI(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            if (kc.y < kc.maxY) {
                float acc = 0;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.y * size + k) * matrixB.array(k * size + kc.x));
                }
                matrixC.array(kc.y * size + kc.x, acc);
            }
        }
    }

    @CodeReflection
    public static float compute(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, int size, int j) {
        float acc = 0;
        for (int k = 0; k < size; k++) {
            acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + j));
        }
        return acc;
    }

    /**
     * Naive Matrix Multiplication implemented in 1D.
     *
     * @param kc
     * @param matrixA
     * @param matrixB
     * @param matrixC
     * @param size
     */
    @CodeReflection
    public static void matrixMultiplyKernel1D(@RO KernelContext kc, @RO F32Array matrixA, @RO F32Array matrixB, @RW F32Array matrixC, int size) {
        if (kc.x < kc.maxX) {
            for (int j = 0; j < size; j++) {
                float acc = 0;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.x * size + k) * matrixB.array(k * size + j));
                }
                matrixC.array(kc.x * size + j, acc);
            }
        }
    }

    /**
     * 1D Matrix Multiply with function calls passing the kernel context ID. This is just for testing purposes.
     */
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
    public static void matrixMultiply1D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, int size) {
        cc.dispatchKernel(size,
                kc -> matrixMultiplyKernel1D(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply1DWithFunctionCalls(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, int size) {
        cc.dispatchKernel(size,
                kc -> matrixMultiplyKernel1DWithFunctionCalls(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2D(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, int size) {
        cc.dispatchKernel(size, size,
                kc -> matrixMultiplyKernel2D(kc, matrixA, matrixB, matrixC, size)
        );
    }

    @CodeReflection
    public static void matrixMultiply2DLI(@RO ComputeContext cc, @RO F32Array matrixA, @RO F32Array matrixB, @RW  F32Array matrixC, int size) {
        cc.dispatchKernel(size, size,
                kc -> matrixMultiplyKernel2DLI(kc, matrixA, matrixB, matrixC, size)
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

    /**
     * Configuration to use in this example to represent a
     * 1D range or 2D range.
     */
    private enum Configuration {
        _1D,   //
        _1DFC, // 1D with multiple function calls: This is just for testing
        _2D,   //
        _2DLI
    }

    /**
     * Run a 2D version by default.
     * @param args
     *      args: <"1D"|"2D"> for 1D dispatch
     *
     */
    public static void main(String[] args) {
        System.out.println("[INFO] Running Matrix Multiplication: ");

        Configuration configuration = Configuration._2D;
        if (args.length > 0) {
            if (args[0].equals("1D")) {
                configuration = Configuration._1D;
            }
            if (args[0].equals("1DFC")) {
                configuration = Configuration._1DFC;
            }
            if (args[0].equals("2DLI")) {
                configuration = Configuration._2DLI;
            }
        }

        System.out.println("[INFO] NDRangeConfiguration: " + configuration);

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

        // Run Seq for reference
        runSequential(matrixA, matrixB, resultSeq, size);

        for (int it = 0; it < NUM_ITERATIONS; it++) {

            long start = System.nanoTime();
            switch (configuration) {
                case _1D -> accelerator.compute(cc ->
                        Main.matrixMultiply1D(cc, matrixA, matrixB, matrixC, size));
                case _1DFC -> accelerator.compute(cc ->
                        Main.matrixMultiply1DWithFunctionCalls(cc, matrixA, matrixB, matrixC, size));
                case _2D -> accelerator.compute(cc ->
                        Main.matrixMultiply2D(cc, matrixA, matrixB, matrixC, size));
                case _2DLI -> accelerator.compute(cc ->
                        Main.matrixMultiply2DLI(cc, matrixA, matrixB, matrixC, size));
            }

            long end = System.nanoTime();
            System.out.println("Elapsed Time: " + (end - start) + " ns");

            // If check result is ON, then check first and lat iterations
            if (it == 0 || it == (NUM_ITERATIONS - 1) && CHECK_RESULT) {
                // Check result for the first iteration
                boolean isCorrect = true;
                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++) {
                        if (Math.abs(resultSeq.array(i * size + j) - matrixC.array(i * size + j)) > 0.01f) {
                            isCorrect = false;
                            break;
                        }
                    }
                    if (!isCorrect) {
                        break;
                    }
                }

                if (isCorrect) {
                    System.out.println("Result is correct!");
                } else {
                    System.out.println("Result is wrong!");
                }
            }
        }
    }
}
