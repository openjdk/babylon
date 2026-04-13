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
package tensors;

import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.examples.common.ParseArgs;
import hat.examples.common.ParseArgs.Options;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static hat.NDRange.Global2D;
import static hat.NDRange.Local2D;
import static hat.NDRange.NDRange2D;
import static hat.NDRange.Warp2D;
import static hat.NDRange.of2D;
import static hat.examples.common.StatUtils.dumpStatsToCSVFile;

/**
 * Example to test the Tensor/Tile API vs the Thread API.
 *
 * <p>
 * How to run?
 * <code>java @hat/run ffi-opencl tensors --iterations=10 --verbose --size=2048</code>
 * </p>
 */
public class Main {

    @Reflect
    public static void mxmTensorsCM(@MappableIface.RO KernelContext kc, @MappableIface.RO F16Array matrixA, @MappableIface.RO F16Array matrixB, @MappableIface.WO F32Array matrixC, int size) {
        final int WMMA_M = 16;
        final int WMMA_N = 16;
        final int WMMA_K = 16;
        int warpM = kc.gix / kc.wrs;
        int warpN = kc.giy;

        final int lda = size;
        final int ldb = size;
        final int ldc = size;

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
    public static void mxmTensorsCM(@MappableIface.RO ComputeContext cc, @MappableIface.RO F16Array matrixA, @MappableIface.RO F16Array matrixB, @MappableIface.WO F32Array matrixC, int globalSize) {
        var ndRange = NDRange2D.of(Global2D.of(globalSize, globalSize),
                Local2D.of(128, 4),
                NDRange.Tile2D.of(16, 16),
                Warp2D.of(true, false));
        cc.dispatchKernel(ndRange, kc -> mxmTensorsCM(kc, matrixA, matrixB, matrixC, globalSize));
    }

    @Reflect
    public static void mxmNaiveF32(KernelContext kc, F32Array matrixA, F32Array matrixB, F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.giy < kc.gsy) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    acc += (matrixA.array(kc.giy * size + k) * matrixB.array(k * size + kc.gix));
                }
                matrixC.array(kc.giy * size + kc.gix, acc);
            }
        }
    }

    @Reflect
    public static void mxmNaiveF32(@MappableIface.RO ComputeContext cc, @MappableIface.RO F32Array matrixA, @MappableIface.RO F32Array matrixB, @MappableIface.WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(globalSize, globalSize, 16, 16),
                kc -> mxmNaiveF32(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    @Reflect
    public static void mxmNaiveF16(@MappableIface.RO KernelContext kc, @MappableIface.RO F16Array matrixA, @MappableIface.RO F16Array matrixB, @MappableIface.WO F32Array matrixC, int size) {
        if (kc.gix < kc.gsx) {
            if (kc.giy < kc.gsy) {
                float acc = 0.0f;
                for (int k = 0; k < size; k++) {
                    F16 ha = matrixA.array(kc.giy * size + k);
                    F16 hb = matrixB.array(k * size + kc.gix);
                    F16 hc = F16.mul(ha, hb);
                    float fc = F16.f16ToFloat(hc);
                    acc += fc;
                }
                matrixC.array(kc.giy * size + kc.gix, acc);
            }
        }
    }

    @Reflect
    public static void mxmNaiveF16(@MappableIface.RO ComputeContext cc, @MappableIface.RO F16Array matrixA, @MappableIface.RO F16Array matrixB, @MappableIface.WO F32Array matrixC, int globalSize) {
        cc.dispatchKernel(of2D(globalSize, globalSize, 16, 16),
                kc -> mxmNaiveF16(kc, matrixA, matrixB, matrixC, globalSize)
        );
    }

    private static void runSequential(F32Array matrixA, F32Array matrixB, F32Array matrixC, final int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                float sum = 0.0f;
                for (int k = 0; k < size; k++) {
                    float a = matrixA.array((long) j * size + k);
                    float b = matrixB.array((long) k * size + i);
                    sum += (a * b);
                }
                matrixC.array((long) j * size + i, sum);
            }
        }
    }

    private static void runMultiThreadedWithStreams(F32Array matrixA, F32Array matrixB, F32Array matrixC, int size) {
        IntStream.range(0, size)
                .parallel()
                .forEach(i -> IntStream.range(0, size)
                        .parallel()
                        .forEach(j -> {
                            float sum = 0.0f;
                            for (int k = 0; k < size; k++) {
                                sum += matrixA.array(i * size + k) * matrixB.array(k * size + j);
                            }
                            matrixC.array(i * size + j, sum);
                        }));
    }

    private static boolean checkResult(F32Array reference, F32Array output, int size) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                final float expected = reference.array(i * size + j);
                final float got = output.array(i * size + j);
                if (Math.abs(expected - got) > 0.1f) {
                    IO.println("GOT: " + got + " - but expected: " + expected);
                    return false;
                }
            }
        }
        return true;
    }

    private static void printResult(String version, boolean check) {
        if (check) {
            IO.println("Result-" + version + " is correct!");
        } else {
            IO.println("Result-" + version + " is wrong!");
        }
    }

    static void runBenchmark(Options options) {
        final int size = options.size();
        final int numIterations = options.iterations();

        options.printOptions();

        List<Long> timersJava = new ArrayList<>();
        List<Long> timersParallelStreams = new ArrayList<>();
        List<Long> timersHATNaiveF32 = new ArrayList<>();
        List<Long> timersHATNaiveF16 = new ArrayList<>();
        List<Long> timersHATTensors = new ArrayList<>();

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        F16Array matrixAHalf = F16Array.create(accelerator, size * size);
        F16Array matrixBHalf = F16Array.create(accelerator, size * size);
        F32Array matrixA = F32Array.create(accelerator, size * size);
        F32Array matrixB = F32Array.create(accelerator, size * size);
        F32Array resultNativeF32 = F32Array.create(accelerator, size * size);
        F32Array resultNativeF16 = F32Array.create(accelerator, size * size);
        F32Array resultTensor = F32Array.create(accelerator, size * size);
        F32Array matrixReference = F32Array.create(accelerator, size * size);
        F32Array resultStreams = F32Array.create(accelerator, size * size);

        Random r = new Random(19);
        for (int j = 0; j < matrixAHalf.length(); j++) {
            float a = r.nextFloat();
            float b = r.nextFloat();
            matrixAHalf.array(j).value(F16.floatToF16(a).value());
            matrixA.array(j, a);
            matrixBHalf.array(j).value(F16.floatToF16(b).value());
            matrixB.array(j, b);
        }

        // Java Sequential
        if (!options.skipSequential()) {
            for (int i = 0; i < numIterations; i++) {
                long start = System.nanoTime();
                runSequential(matrixA, matrixB, matrixReference, size);
                long end = System.nanoTime();
                if (options.verbose()) {
                    IO.println("Java Seq Timer: " + (end - start));
                }
                timersJava.add((end - start));
            }
        }

        // Java Parallel Streams
        for (int i = 0; i < numIterations; i++) {
            long start = System.nanoTime();
            runMultiThreadedWithStreams(matrixA, matrixB, resultStreams, size);
            long end = System.nanoTime();
            if (options.verbose()) {
                IO.println("Java Parallel-Stream Timer: " + (end - start));
            }
            timersParallelStreams.add((end - start));
        }

        // HAT Parallel Naive F32
        for (int i = 0; i < numIterations; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) cc -> mxmNaiveF32(cc, matrixA, matrixB, resultNativeF32, size));
            long end = System.nanoTime();
            if (options.verbose()) {
                IO.println("HAT GPU-Naive-F32 Timer: " + (end - start));
            }
            timersHATNaiveF32.add((end - start));
        }

        // HAT Parallel Naive F16
        for (int i = 0; i < numIterations; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) cc -> mxmNaiveF16(cc, matrixAHalf, matrixBHalf, resultNativeF16, size));
            long end = System.nanoTime();
            if (options.verbose()) {
                IO.println("HAT GPU-Naive-F16 Timer: " + (end - start));
            }
            timersHATNaiveF16.add((end - start));
        }

        // HAT Parallel Tensor
        for (int i = 0; i < numIterations; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) cc -> mxmTensorsCM(cc, matrixAHalf, matrixBHalf, resultTensor, size));
            long end = System.nanoTime();
            if (options.verbose()) {
                IO.println("HAT GPU-Tensors Timer: " + (end - start));
            }
            timersHATTensors.add((end - start));
        }

        if (options.checkResult() && !options.skipSequential()) {
            printResult("streams", checkResult(matrixReference, resultStreams, size));
            printResult("HAT-NaiveF32", checkResult(matrixReference, resultNativeF32, size));
            printResult("HAT-NaiveF16", checkResult(matrixReference, resultNativeF16, size));
            printResult("HAT-Tensors", checkResult(matrixReference, resultTensor, size));
        }

        // Write CSV table for all the results
        List<List<Long>> timers = options.skipSequential() ?
                List.of(timersParallelStreams, timersHATNaiveF32, timersHATNaiveF16, timersHATTensors) :
                List.of(timersJava, timersParallelStreams, timersHATNaiveF32, timersHATNaiveF16, timersHATTensors);

        List<String> headers = options.skipSequential() ?
                List.of("Java-streams-fp32-" + size, "HAT-naive-fp32-" + size, "HAT-naive-fp16-" + size, "HAT-tensors-fp16-" + size) :
                List.of("Java-fp32-" + size, "Java-streams-fp32-" + size, "HAT-naive-fp32-" + size, "HAT-naive-fp16-" + size, "HAT-tensors-fp16-" + size);

        final String tableName = "table-tensors-" + size + ".csv";
        dumpStatsToCSVFile(timers, headers, tableName);
    }

    static void main(String[] args) {
        IO.println("Example of Matmul with Tensors");

        final int defaultSize = 1024;
        int numIterations = 100;
        ParseArgs parseArgs = new ParseArgs(args);
        Options options = parseArgs.parseWithDefaults(defaultSize, numIterations);

        // check input size
        if (options.size() % 16 != 0 || options.size() < 128) {
            throw new RuntimeException("Input size must of a multiple of 16, and larger than 128");
        }
        runBenchmark(options);
    }
}
