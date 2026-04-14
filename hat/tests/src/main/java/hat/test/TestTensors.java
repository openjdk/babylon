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
import hat.KernelContext;
import hat.NDRange.Tile2D;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.buffer.F32ArrayPadded;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAssertionError;
import hat.test.exceptions.HATAsserts;
import hat.test.exceptions.HATExpectedPrecisionError;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Reflect;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.NDRange.Global2D;
import static hat.NDRange.Local2D;
import static hat.NDRange.NDRange2D;
import static hat.NDRange.Warp2D;
import static optkl.ifacemapper.MappableIface.RO;
import static optkl.ifacemapper.MappableIface.WO;

/**
 * How to run?
 *
 * <p>
 * <code>
 * HAT=SHOW_CODE java -cp hat/job.jar hat.java test ffi-cuda hat.test.TestTensors
 * HAT=SHOW_CODE java -cp hat/job.jar hat.java test ffi-opencl hat.test.TestTensors
 * </code>
 * </p>
 *
 */
public class TestTensors {

    @Reflect
    public static void mxmTensorsColumnMajor(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int size) {
        final int WMMA_M = 16;
        final int WMMA_N = 16;
        final int WMMA_K = 16;
        int warpM = kc.gix / kc.wrs;
        int warpN = kc.giy;

        final int lda = 1024;
        final int ldb = 1024;
        final int ldc = 1024;

        Tensor tensorA = Tensor.create(Tensor.FIRST, Tensor.shape(16, 16, 16), F16.class, Tensor.ofColumnMajor());
        Tensor tensorB = Tensor.create(Tensor.SECOND, Tensor.shape(16, 16, 16), F16.class, Tensor.ofColumnMajor());
        Tensor acc = Tensor.create(Tensor.ACC, Tensor.shape(16, 16, 16), float.class);

        Tensor.fill(acc, 0.0f);

        for (int i = 0; i < size; i += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = i;

            int bRow = i;
            int bCol = warpN * WMMA_N;

            if (aRow < lda && aCol < lda && bRow < ldb && bCol < ldb) {

                tensorA = Tensor.load(matrixA, aRow, aCol, lda);
                tensorB = Tensor.load(matrixB, bRow, bCol, ldb);

                // acc = tensorA * tensorB + acc
                Tensor.mma(acc, tensorA, tensorB, acc);
            }
        }
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;
        Tensor.store(matrixC, cRow, cCol, acc, ldc, Tensor.ofColumnMajor());
    }

    @Reflect
    public static void mxmTensorsColumnMajor(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int globalSize) {
        // The total number of threads is calculated as follows:
        // [ (size / tile), (size / tile) ]
        // If warpSize > 1, then each dimension using warp operations is multiplied by the value of the warp-size. This is architecture dependent, but the
        // HAT runtime and HAT JIT compiler handle this automatically.

        var ndRange = NDRange2D.of(Global2D.of(globalSize, globalSize),
                Local2D.of(128, 4),
                Tile2D.of(16, 16),
                Warp2D.of(true, false));

        cc.dispatchKernel(ndRange, kc -> mxmTensorsColumnMajor(kc, matrixA, matrixB, matrixC, globalSize));
    }

    @Reflect
    public static void mxmTensorsRowColumnMajor(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int size) {
        final int WMMA_M = 16;
        final int WMMA_N = 16;
        final int WMMA_K = 16;
        int warpM = kc.gix / kc.wrs;
        int warpN = kc.giy;

        final int lda = 1024;
        final int ldb = 1024;
        final int ldc = 1024;

        Tensor tensorA = Tensor.create(Tensor.FIRST, Tensor.shape(16, 16, 16), F16.class, Tensor.ofRowMajor());
        Tensor tensorB = Tensor.create(Tensor.SECOND, Tensor.shape(16, 16, 16), F16.class, Tensor.ofColumnMajor());
        Tensor acc = Tensor.create(Tensor.ACC, Tensor.shape(16, 16, 16), float.class);

        Tensor.fill(acc, 0.0f);

        for (int i = 0; i < size; i += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = i;

            int bRow = i;
            int bCol = warpN * WMMA_N;

            if (aRow < lda && aCol < lda && bRow < ldb && bCol < ldb) {

                tensorA = Tensor.load(matrixA, aRow, aCol, lda);
                tensorB = Tensor.load(matrixB, bRow, bCol, ldb);

                // acc = tensorA * tensorB + acc
                Tensor.mma(acc, tensorA, tensorB, acc);
            }
        }
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;
        Tensor.store(matrixC, cRow, cCol, acc, ldc, Tensor.ofColumnMajor());
    }

    @Reflect
    public static void mxmTensorsRowColumnMajor(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32Array matrixC, int globalSize) {
        // The total number of threads is calculated as follows:
        // [ (size / tile), (size / tile) ]
        // If warpSize > 1, then each dimension using warp operations is multiplied by the value of the warp-size. This is architecture dependent, but the
        // HAT runtime and HAT JIT compiler handle this automatically.

        var ndRange = NDRange2D.of(Global2D.of(globalSize, globalSize),
                Local2D.of(128, 4),
                Tile2D.of(16, 16),
                Warp2D.of(true, false));

        cc.dispatchKernel(ndRange, kc -> mxmTensorsRowColumnMajor(kc, matrixA, matrixB, matrixC, globalSize));
    }

    @Reflect
    public static void mxmTensorsRowMajor(@RO KernelContext kc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32ArrayPadded matrixC, int size) {
        final int WMMA_M = 16;
        final int WMMA_N = 16;
        final int WMMA_K = 16;
        int warpM = kc.gix / kc.wrs;
        int warpN = kc.giy;

        final int lda = 1024;
        final int ldb = 1024;
        final int ldc = 1024;

        Tensor tensorA = Tensor.create(Tensor.FIRST, Tensor.shape(16, 16, 16), F16.class, Tensor.ofRowMajor());
        Tensor tensorB = Tensor.create(Tensor.SECOND, Tensor.shape(16, 16, 16), F16.class, Tensor.ofRowMajor());
        Tensor acc = Tensor.create(Tensor.ACC, Tensor.shape(16, 16, 16), float.class);

        Tensor.fill(acc, 0.0f);

        for (int i = 0; i < size; i += WMMA_K) {
            int aRow = warpM * WMMA_M;
            int aCol = i;

            int bRow = i;
            int bCol = warpN * WMMA_N;

            if (aRow < lda && aCol < lda && bRow < ldb && bCol < ldb) {

                tensorA = Tensor.load(matrixA, aRow, aCol, lda);
                tensorB = Tensor.load(matrixB, bRow, bCol, ldb);

                // acc = tensorA * tensorB + acc
                Tensor.mma(acc, tensorA, tensorB, acc);
            }
        }
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;
        Tensor.store(matrixC, cRow, cCol, acc, ldc, Tensor.ofRowMajor());
    }

    @Reflect
    public static void mxmTensorsRowMajor(@RO ComputeContext cc, @RO F16Array matrixA, @RO F16Array matrixB, @WO F32ArrayPadded matrixC, int globalSize) {
        var ndRange = NDRange2D.of(
                Global2D.of(globalSize, globalSize),
                Local2D.of(128, 4),
                Tile2D.of(16, 16),
                Warp2D.of(true, false));
        cc.dispatchKernel(ndRange, kc -> mxmTensorsRowMajor(kc, matrixA, matrixB, matrixC, globalSize));
    }

    private static void runSequentialColMajor(F16Array matrixA, F16Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    F16 a = matrixA.array((long) k * size + i);
                    F16 b = matrixB.array((long) j * size + k);
                    F16 mul = F16.mul(a, b);
                    sum += F16.f16ToFloat(mul);
                }
                matrixC.array((long) j * size + i, sum);
            }
        }
    }

    private static void runSequentialRowAndColMajor(F16Array matrixA, F16Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    F16 a = matrixA.array((long) i * size + k);
                    F16 b = matrixB.array((long) j * size + k);
                    F16 mul = F16.mul(a, b);
                    sum += F16.f16ToFloat(mul);
                }
                matrixC.array((long) j * size + i, sum);
            }
        }
    }

    private static void runSequentialRowMajor(F16Array matrixA, F16Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
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
    @Reflect
    public void testTensor01() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;

        F16Array matrixAHalf = F16Array.create(accelerator, size * size);
        F16Array matrixBHalf = F16Array.create(accelerator, size * size);
        F32Array matrixC = F32Array.create(accelerator, size * size);
        F32Array resultSequential = F32Array.create(accelerator, size * size);

        Random r = new Random(19);
        for (int j = 0; j < matrixAHalf.length(); j++) {
            matrixAHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixBHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        for (int i = 0; i < 10; i++) {
            accelerator.compute(cc -> mxmTensorsColumnMajor(cc, matrixAHalf, matrixBHalf, matrixC, size));
        }

        runSequentialColMajor(matrixAHalf, matrixBHalf, resultSequential, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final int index = j * size + i;
                float expectedValue = resultSequential.array(index);
                float gotValue = matrixC.array(index);
                try {
                    HATAsserts.assertEquals(expectedValue, gotValue, 0.1f);
                } catch (HATAssertionError e) {
                    throw new HATExpectedPrecisionError("Expected: " + expectedValue + " but got " + gotValue);
                }
            }
        }
    }

    @HatTest
    @Reflect
    public void testTensor02() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;

        F16Array matrixAHalf = F16Array.create(accelerator, size * size);
        F16Array matrixBHalf = F16Array.create(accelerator, size * size);
        F32Array matrixC = F32Array.create(accelerator, size * size);
        F32Array resultSequential = F32Array.create(accelerator, size * size);

        Random r = new Random(19);
        for (int j = 0; j < matrixAHalf.length(); j++) {
            matrixAHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixBHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        for (int i = 0; i < 10; i++) {
            accelerator.compute(cc -> mxmTensorsRowColumnMajor(cc, matrixAHalf, matrixBHalf, matrixC, size));
        }

        runSequentialRowAndColMajor(matrixAHalf, matrixBHalf, resultSequential, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final int index = j * size + i;
                float expectedValue = resultSequential.array(index);
                float gotValue = matrixC.array(index);
                try {
                    HATAsserts.assertEquals(expectedValue, gotValue, 0.1f);
                } catch (HATAssertionError e) {
                    throw new HATExpectedPrecisionError("Expected: " + expectedValue + " but got " + gotValue);
                }
            }
        }
    }

    @HatTest
    @Reflect
    public void testTensor03() {

        // To be able to run tensor-matmul in a row-major layout, we need to add padding.
        // Thus, the result matrix must be of type F32ArrayPadded.

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        final int size = 1024;

        F16Array matrixAHalf = F16Array.create(accelerator, size * size);
        F16Array matrixBHalf = F16Array.create(accelerator, size * size);
        F32ArrayPadded matrixC = F32ArrayPadded.create(accelerator, size * size);
        F32Array resultSequential = F32Array.create(accelerator, size * size);

        Random r = new Random(19);
        for (int j = 0; j < matrixAHalf.length(); j++) {
            matrixAHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
            matrixBHalf.array(j).value(F16.floatToF16(r.nextFloat()).value());
        }

        for (int i = 0; i < 10; i++) {
            accelerator.compute(cc -> mxmTensorsRowMajor(cc, matrixAHalf, matrixBHalf, matrixC, size));
        }

        runSequentialRowMajor(matrixAHalf, matrixBHalf, resultSequential, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final int index = j * size + i;
                float expectedValue = resultSequential.array(index);
                float gotValue = matrixC.array(index);
                try {
                    HATAsserts.assertEquals(expectedValue, gotValue, 0.1f);
                } catch (HATAssertionError e) {
                    throw new HATExpectedPrecisionError("Expected: " + expectedValue + " but got " + gotValue);
                }
            }
        }
    }
}
