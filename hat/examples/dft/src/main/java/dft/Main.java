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
package dft;

import dft.Main.ComplexArray.Complex;
import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.HATMath;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.examples.common.ParseArgs;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import optkl.ifacemapper.MappableIface.WO;
import optkl.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

import static hat.examples.common.StatUtils.computeAverage;
import static hat.examples.common.StatUtils.computeSpeedup;
import static hat.examples.common.StatUtils.dumpStatsToCSVFile;
import static hat.examples.common.StatUtils.printCheckResult;

/**
 * How to run?
 *
 * <p>
 * With the OpenCL Backend:
 * <code>
 *     java -cp hat/job.jar hat.java run ffi-opencl dft --size=<size> --iterations=<iterations> --verbose
 * </code>
 * </p>
 *
 * <p>
 * With the CUDA Backend:
 * <code>
 *      java -cp hat/job.jar hat.java run ffi-cuda dft --size=<size> --iterations=<iterations> --verbose
 * </code>
 *
 * <p>
 * Link to DFT: <a href="https://en.wikipedia.org/wiki/Discrete_Fourier_transform">link</a>
 * </p>
 *
 * </p>
 */
public class Main {

    public static final float DELTA = 0.001f;

    // Use a custom data structure for dealing with Array of Complex Numbers
    public interface ComplexArray extends Buffer {
        int length();

        interface Complex extends Struct {
            float real();
            float imag();
            void real(float real);
            void imag(float imag);
        }

        Complex complex(long index);

        Schema<ComplexArray> schema = Schema.of(ComplexArray.class, complex ->
                complex.arrayLen("length")
                        .array("complex",
                                array -> array.fields("real", "imag")));

        static ComplexArray create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }
    }

    @Reflect
    private static void dftKernel(@RW KernelContext kc, @RO ComplexArray input, @WO ComplexArray output) {
        int size = input.length();
        int idx = kc.gix;
        if (idx < kc.gsx) {
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            for (int k = 0; k < size; k++) {
                float angle = -2 * HATMath.PI * ((k * idx) % size) / size;
                Complex complexInput = input.complex(k);
                float cReal = HATMath.native_cosf(angle);
                float cImag = HATMath.native_sinf(angle);
                sumReal += (complexInput.real() * cReal) - (complexInput.imag() * cImag);
                sumImag += (complexInput.real() * cImag) + (complexInput.imag() * cReal);
            }
            Complex complexOutput = output.complex(idx);
            complexOutput.real(sumReal);
            complexOutput.imag(sumImag);
        }
    }

    @Reflect
    private static void dftCompute(@RW ComputeContext cc, @RO ComplexArray input, @WO ComplexArray output) {
        var range = NDRange.of1D(input.length(), 256);
        cc.dispatchKernel(range, kernelContext -> dftKernel(kernelContext, input, output));
    }

    @Reflect
    private static void dftPlainKernel(@RW KernelContext kc, @RO F32Array inReal, @RO F32Array inImag, @WO F32Array outReal, @WO F32Array outImag) {
        int size = inReal.length();
        int idx = kc.gix;
        if (idx < kc.gsx) {
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            for (int k = 0; k < size; k++) {
                float angle = -2 * HATMath.PI * ((idx * k) % size) / size;
                float cReal = HATMath.native_cosf(angle);
                float cImag = HATMath.native_sinf(angle);
                sumReal += (inReal.array(k) * cReal) - (inImag.array(k) * cImag);
                sumImag += (inReal.array(k) * cImag) + (inImag.array(k) * cReal);
            }
            outReal.array(idx, sumReal);
            outImag.array(idx, sumImag);
        }
    }

    @Reflect
    private static void dftPlainCompute(@RW ComputeContext cc, @RO F32Array inReal, @RO F32Array inImag, @WO F32Array outReal, @WO F32Array outImag) {
        var range = NDRange.of1D(inReal.length(), 256);
        cc.dispatchKernel(range, kernelContext -> dftPlainKernel(kernelContext, inReal, inImag, outReal, outImag));
    }

    private static void dftJava(ComplexArray input, ComplexArray output) {
        int size = input.length();
        for (int k = 0; k < size; k++) {
            Complex complexOutput = output.complex(k);
            complexOutput.real(0.0f);
            complexOutput.imag(0.0f);
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            for (int j = 0; j < size; j++) {
                float angle = -2 * HATMath.PI * ((j * k) % size) / size;
                Complex complexInput = input.complex(j);
                float cReal = HATMath.cosf(angle);
                float cImag = HATMath.sinf(angle);
                sumReal += (complexInput.real() * cReal) - (complexInput.imag() * cImag);
                sumImag += (complexInput.real() * cImag) + (complexInput.imag() * cReal);
            }
            complexOutput.real(sumReal);
            complexOutput.imag(sumImag);
        }
    }

    private static void dftJavaStreams(ComplexArray input, ComplexArray output) {
        int size = input.length();
        IntStream.range(0, size).parallel().forEach(idx -> {
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            for (int k = 0; k < size; k++) {
                float angle = -2 * HATMath.PI * ((idx * k) % size) / size;
                Complex complexInput = input.complex(k);
                float cReal = HATMath.cosf(angle);
                float cImag = HATMath.sinf(angle);
                sumReal += (complexInput.real() * cReal) - (complexInput.imag() * cImag);
                sumImag += (complexInput.real() * cImag) + (complexInput.imag() * cReal);
            }
            Complex complexOutput = output.complex(idx);
            complexOutput.real(sumReal);
            complexOutput.imag(sumImag);
        });
    }

    private static boolean checkResult(ComplexArray expected, ComplexArray obtained) {
        for (int i = 0; i < expected.length(); i++) {
            if (Math.abs(expected.complex(i).real() - obtained.complex(i).real()) > DELTA) {
                IO.println(expected.complex(i).real() + " vs " + obtained.complex(i).real());
                return false;
            }
            if (Math.abs(expected.complex(i).imag() - obtained.complex(i).imag()) > DELTA) {
                IO.println(expected.complex(i).imag() + " vs " + obtained.complex(i).imag());
                return false;
            }
        }
        return true;
    }

    // Just for debugging
    private static void printSignal(ComplexArray signal) {
        for (int i = 0; i < signal.length(); i++) {
            IO.println(signal.complex(i).real() + "," + signal.complex(i).imag());
        }
    }

    private static boolean checkResult(ComplexArray outputSeq, F32Array outReal, F32Array outImag) {
        for (int i = 0; i < outputSeq.length(); i++) {
            if (Math.abs(outputSeq.complex(i).real() - outReal.array(i)) > DELTA) {
                IO.println(outputSeq.complex(i).real() + " vs " + outReal.array(i));
                return false;
            }
            if (Math.abs(outputSeq.complex(i).imag() - outImag.array(i)) > DELTA) {
                IO.println(outputSeq.complex(i).imag() + " vs " + outImag.array(i));
                return false;
            }
        }
        return true;
    }

    static void main(String[] args) {
        IO.println("=========================================");
        IO.println("Example: Discrete Fourier Transform (DFT)");
        IO.println("=========================================");

        int size = 32768;
        int iterations = 100;
        ParseArgs parseArgs = new ParseArgs(args);
        ParseArgs.Options options = parseArgs.parseWithDefaults(size, iterations);

        boolean verbose = options.verbose();
        size = options.size();
        iterations = options.iterations();
        boolean skipSequential = options.skipSequential();
        IO.println("Input Size     = " + size);
        IO.println("Num Iterations = " + iterations);

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);
        ComplexArray input = ComplexArray.create(accelerator, size);
        ComplexArray outputSeq = ComplexArray.create(accelerator, size);
        ComplexArray.create(accelerator, size);
        ComplexArray outputStreams = ComplexArray.create(accelerator, size);
        ComplexArray outputHAT = ComplexArray.create(accelerator, size);

        F32Array inReal = F32Array.create(accelerator, size);
        F32Array inImag = F32Array.create(accelerator, size);
        F32Array outReal = F32Array.create(accelerator, size);
        F32Array outImag = F32Array.create(accelerator, size);

        // Initialize
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            Complex c = input.complex(i);
            c.real(r.nextFloat());
            c.imag(r.nextFloat());
            inReal.array(i, c.real());
            inImag.array(i, c.imag());
        }

        List<Long> timersJavaDFT = new ArrayList<>();
        List<Long> timersStreams = new ArrayList<>();
        List<Long> timersDFTHat = new ArrayList<>();
        List<Long> timersDFTHatFlatten = new ArrayList<>();

        // Run Java sequential version, DFT
        if (!skipSequential) {
            for (int i = 0; i < iterations; i++) {
                long start = System.nanoTime();
                dftJava(input, outputSeq);
                long end = System.nanoTime();
                timersJavaDFT.add((end - start));
                if (verbose) {
                    IO.println("[Timer] Java DFT: " + (end - start));
                }
            }
        }

        // Run Java Parallel Stream Version of the DFT
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            dftJavaStreams(input, outputStreams);
            long end = System.nanoTime();
            timersStreams.add((end-start));
            if (verbose) {
                IO.println("[Timer] Parallel Stream: " + (end-start));
            }
        }

        // HAT: Initial version (DFT)
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) computeContext -> dftCompute(computeContext, input, outputHAT));
            long end = System.nanoTime();
            timersDFTHat.add((end-start));
            if (verbose) {
                IO.println("[Timer] HAT with CDS: " + (end-start));
            }
        }

        // HAT: DFT using plain arrays instead of a custom data structure
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) computeContext -> dftPlainCompute(computeContext, inReal, inImag, outReal, outImag));
            long end = System.nanoTime();
            timersDFTHatFlatten.add((end - start));
            if (verbose) {
                IO.println("[Timer] HAT-Plain: " + (end - start));
            }
        }

        // Check results
        ComplexArray baseline = skipSequential ? outputStreams : outputSeq;
        boolean isStreamCorrect = checkResult(baseline, outputStreams);
        boolean isHATCorrect = checkResult(baseline, outputHAT);
        boolean isHATPlainCorrect = checkResult(baseline, outReal, outImag);
        printCheckResult(isStreamCorrect, "Java-Stream");
        printCheckResult(isHATCorrect, "HAT-Naive");
        printCheckResult(isHATPlainCorrect, "HAT-Plain");

        // Print Performance Metrics
        final int skip = iterations / 2;
        double averageJavaTimer = computeAverage(timersJavaDFT, skip);
        double averageJavaStreamTimer = computeAverage(timersStreams, skip);
        double averageHATTimers = computeAverage(timersDFTHat, skip);
        double averageHATTimersFlatten = computeAverage(timersDFTHatFlatten, skip);

        IO.println("\nAverage elapsed time:");
        IO.println("Average Java-Seq DFT           : " + averageJavaTimer);
        IO.println("Average Java-Streams DFT       : " + averageJavaStreamTimer);
        IO.println("Average HAT DFT                : " + averageHATTimers);
        IO.println("Average HAT DFT (Flatten)      : " + averageHATTimersFlatten);

        if (!skipSequential) {
            IO.println("\nSpeedups vs Java:");
            IO.println("Java / Java Parallel Stream    : " + computeSpeedup(averageJavaTimer, averageJavaStreamTimer) + "x");
            IO.println("Java / HAT                     : " + computeSpeedup(averageJavaTimer, averageHATTimers) + "x");
            IO.println("Java / HAT-Flatten             : " + computeSpeedup(averageJavaTimer, averageHATTimersFlatten) + "x");
        }

        IO.println("\nSpeedups vs Java Parallel Streams:");
        IO.println("Java / HAT                     : " + computeSpeedup(averageJavaStreamTimer, averageHATTimers) + "x");
        IO.println("Java / HAT-Flatten             : " + computeSpeedup(averageJavaStreamTimer, averageHATTimersFlatten) + "x");

        IO.println("\nSpeedups vs HAT-DFT:");
        IO.println("HAT / HAT-Flatten              : " + computeSpeedup(averageHATTimers, averageHATTimersFlatten) + "x");

        // Write CSV table with all results
        List<List<Long>> timers = List.of(timersJavaDFT, timersStreams, timersDFTHat, timersDFTHatFlatten);
        List<String> header = List.of("Java-fp32-" + size, "Streams-fp32-" + size, "HAT-UDT-fp32-" + size, "HAT-Plain-fp32-" + size);
        String fileName = "table-results-dft-" + size + ".csv";
        if (skipSequential) {
            timers = List.of(timersStreams, timersDFTHat, timersDFTHatFlatten);
            header = List.of("Streams-fp32-" + size, "HAT-UDT-fp32-" + size, "HAT-Plain-fp32-" + size);
        }
        dumpStatsToCSVFile(timers, header, fileName);
    }
}
