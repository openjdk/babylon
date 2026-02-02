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
package fft;

import fft.Main.ArrayComplex.Complex;
import hat.Accelerator;
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.HATMath;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
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
 *     java -cp hat/job.jar hat.java run ffi-opencl fft
 * </code>
 * </p>
 *
 * <p>
 * With the CUDA Backend:
 * <code>
 *      java -cp hat/job.jar hat.java run ffi-cuda fft
 * </code>
 *
 * <p>
 * Link to DFT: <a href="https://en.wikipedia.org/wiki/Discrete_Fourier_transform">link</a>
 * </p>
 *
 * </p>
 */
public class Main {

    public static final int ITERATIONS = 10;

    // Use a custom data structure for dealing with Array of Complex Numbers
    public interface ArrayComplex extends Buffer {
        int length();

        interface Complex extends Struct {
            float real();
            float imag();
            void real(float real);
            void imag(float imag);
        }

        Complex complex(long index);

        Schema<ArrayComplex> schema = Schema.of(ArrayComplex.class, complex -> {
            complex.arrayLen("length")
                    .array("complex", array -> array.fields("real", "imag"));
        });

        static ArrayComplex create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }
    }

    @Reflect
    private static void dftKernel(@RW KernelContext kc, @RO ArrayComplex input, @WO ArrayComplex output) {
        int size = input.length();
        if (kc.gix < kc.gsx) {
            for (int i = 0; i < size; i++) {
                float sumReal = 0.0f;
                float sumImag = 0.0f;
                for (int k = 0; k < size; k++) {
                    float angle = 2 * HATMath.PI * k * i / size;
                    Complex complexInput = input.complex(k);
                    sumReal += complexInput.real() * HATMath.cos(angle) + complexInput.imag() * HATMath.sin(angle);
                    sumImag += -complexInput.real() * HATMath.sin(angle) + complexInput.imag() * HATMath.cos(angle);
                }
                Complex complexOutput = output.complex(i);
                complexOutput.real(sumReal);
                complexOutput.imag(sumImag);
            }
        }
    }

    @Reflect
    private static void dftCompute(@RW ComputeContext cc, @RO ArrayComplex input, @WO ArrayComplex output) {
        var range = NDRange.of1D(input.length(), 256);
        cc.dispatchKernel(range, kernelContext -> dftKernel(kernelContext, input, output));
    }

    private static void dftJava(ArrayComplex input, ArrayComplex output) {
        int size = input.length();
        for (int j = 0; j < size; j++) {
            for (int i = 0; i < size; i++) {
                float sumReal = 0.0f;
                float sumImag = 0.0f;
                for (int k = 0; k < size; k++) {
                    float angle = 2 * HATMath.PI * k * i / size;
                    Complex complexInput = input.complex(k);
                    sumReal += complexInput.real() * HATMath.cos(angle) + complexInput.imag() * HATMath.sin(angle);
                    sumImag += -complexInput.real() * HATMath.sin(angle) + complexInput.imag() * HATMath.cos(angle);
                }
                Complex complexOutput = output.complex(i);
                complexOutput.real(sumReal);
                complexOutput.imag(sumImag);
            }
        }
    }

    private static void dftJavaStreams(ArrayComplex input, ArrayComplex output) {
        int size = input.length();
        IntStream.range(0, size).parallel().forEach(j -> {
            for (int i = 0; i < size; i++) {
                float sumReal = 0.0f;
                float sumImag = 0.0f;
                for (int k = 0; k < size; k++) {
                    float angle = 2 * HATMath.PI * k * i / size;
                    Complex complexInput = input.complex(k);
                    sumReal += complexInput.real() * HATMath.cos(angle) + complexInput.imag() * HATMath.sin(angle);
                    sumImag += -complexInput.real() * HATMath.sin(angle) + complexInput.imag() * HATMath.cos(angle);
                }
                Complex complexOutput = output.complex(i);
                complexOutput.real(sumReal);
                complexOutput.imag(sumImag);
            }
        });
    }


    private static boolean checkResult(ArrayComplex outputSeq, ArrayComplex outputHAT) {
        boolean isResultCorrect = true;
        for (int i = 0; i < outputSeq.length(); i++) {
            if (Math.abs(outputSeq.complex(i).real() - outputHAT.complex(i).real()) > 0.001f) {
                return false;
            }
            if (Math.abs(outputSeq.complex(i).imag() - outputHAT.complex(i).imag()) > 0.001f) {
                return false;
            }
        }
        return isResultCorrect;
    }

    static void main(String[] args) {
        IO.println("Example: Fast Fourier Transform (FFT)");

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        boolean verbose = false;
        int size = 4096;
        // process parameters
        for (String arg : args) {
            if (arg.equals("--verbose")) {
                verbose = true;
                IO.println("Verbose mode on? " + verbose);
            } else if (arg.startsWith("--size=")) {
                String number = arg.split("=")[1];
                try {
                    size = Integer.parseInt(number);
                } catch (NumberFormatException _) {
                }
            }
        }
        IO.println("Input Size = " + size);

        // Let's first compute the DFT (Discrete Fourier Transform)

        ArrayComplex input = ArrayComplex.create(accelerator, size);
        ArrayComplex outputSeq = ArrayComplex.create(accelerator, size);
        ArrayComplex outputStreams = ArrayComplex.create(accelerator, size);
        ArrayComplex outputHAT = ArrayComplex.create(accelerator, size);

        // Initialize
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            Complex c = input.complex(i);
            c.real(r.nextFloat());
            c.imag(r.nextFloat());
        }

        List<Long> timersJava = new ArrayList<>();
        List<Long> timersStreams = new ArrayList<>();
        List<Long> timersDFTHat = new ArrayList<>();

        // Run Java sequential version
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            dftJava(input, outputSeq);
            long end = System.nanoTime();
            timersJava.add((end-start));
            if (verbose) {
                IO.println("[Timer] seq: " + (end-start));
            }
        }

        // Run Java sequential version
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            dftJavaStreams(input, outputStreams);
            long end = System.nanoTime();
            timersStreams.add((end-start));
            if (verbose) {
                IO.println("[Timer] seq: " + (end-start));
            }
        }

        // HAT: Initial version (DFT)
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            accelerator.compute((@Reflect Compute) computeContext -> dftCompute(computeContext, input, outputHAT));
            long end = System.nanoTime();
            timersDFTHat.add((end-start));
            if (verbose) {
                IO.println("[Timer] hat: " + (end-start));
            }
        }

        // Check results
        boolean isStreamCorrect = checkResult(outputSeq, outputStreams);
        boolean isHATCorrect = checkResult(outputSeq, outputHAT);
        printCheckResult(isStreamCorrect, "Java-Stream");
        printCheckResult(isHATCorrect, "HAT-Naive");

        // Print Performance Metrics
        final int skip = ITERATIONS / 2;
        double averageJavaTimer = computeAverage(timersJava, skip);
        double averageJavaStreamTimer = computeAverage(timersStreams, skip);
        double averageHATTimers = computeAverage(timersDFTHat, skip);

        IO.println("\nAverage elapsed time:");
        IO.println("Average Java-Seq     DFT       : " + averageJavaTimer);
        IO.println("Average Java-Streams DFT       : " + averageJavaStreamTimer);
        IO.println("Average HAT DFT                : " + averageHATTimers);

        IO.println("\nSpeedups vs Java:");
        IO.println("Java / Java Parallel Stream       : " + computeSpeedup(averageJavaTimer, averageJavaStreamTimer) + "x");
        IO.println("Java / HAT                        : " + computeSpeedup(averageJavaTimer, averageHATTimers) + "x");

        // Write CSV table with all results
        dumpStatsToCSVFile(
                List.of(
                    timersJava,
                    timersStreams,
                    timersDFTHat),
                List.of("Java-fp32",
                        "Streams-fp32",
                        "HAT-Naive-fp32"),
                "table-results-fft-" + size + ".csv");
    }

}
