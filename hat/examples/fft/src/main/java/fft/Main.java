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

import fft.Main.ComplexArray.Complex;
import hat.Accelerator;
import hat.HATMath;
import hat.backend.Backend;
import hat.examples.common.ParseArgs;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static hat.examples.common.StatUtils.computeAverage;
import static hat.examples.common.StatUtils.dumpStatsToCSVFile;

/**
 * How to run?
 *
 * <p>
 * With the OpenCL Backend:
 * <code>
 *     java -cp hat/job.jar hat.java run ffi-opencl fft --size=<size> --iterations=<iterations> -verbose
 * </code>
 * </p>
 *
 * <p>
 * With the CUDA Backend:
 * <code>
 *      java -cp hat/job.jar hat.java run ffi-cuda fft --size=<size> --iterations=<iterations> --verbose
 * </code>
 *
 * <p>
 * Link to FFT: <a href="https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm"">link</a>
 * </p>
 *
 * </p>
 */
public class Main {

    public static final int ITERATIONS = 100;

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

        Schema<ComplexArray> schema = Schema.of(ComplexArray.class, complex -> {
            complex.arrayLen("length")
                    .array("complex", array -> array.fields("real", "imag"));
        });

        static ComplexArray create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }
    }

    /**
     * Initial version of FFT in Java.
     *
     * Trying to map the algorithm explain from Wikipedia into Java. It might need some refining.
     *
     * <p>
     * Link: <a href="https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm">Cooley–Tukey_FFT_algorithm</a>
     * </p>
     *
     * @param input
     * @param output
     */
    private static void fftJava(ComplexArray input, ComplexArray output) {
        final int size = input.length();

        // We need to add +1 to avoid the use of input.length() as index
        int maxValue = Integer.numberOfLeadingZeros(size) + 1;

        // Reverse indexes
        for (int i = 0; i < size; i++) {
            int reverse = Integer.reverse(i) >>> maxValue;
            if (i < reverse) {
                // swap i <-> reverse
                // We can use the output since we do not need to compute in-place
                output.complex(i).real(input.complex(reverse).real());
                output.complex(i).imag(input.complex(reverse).imag());

                output.complex(reverse).real(input.complex(i).real());
                output.complex(reverse).imag(input.complex(i).imag());
            } else {
                // if not, copy directly from the input
                output.complex(i).real(input.complex(i).real());
                output.complex(i).imag(input.complex(i).imag());
            }
        }

        // m <- 2^s
        for (int s = 2; s <= size; s *= 2) {
            for (int k = 0; k < size; k += s) {
                // ωm <- exp(−2πi/m)
                float angle = 2 * HATMath.PI * k / s;
                float wmReal = HATMath.cos(angle);
                float wmImag = -HATMath.sin(angle);

                // w <- 1
                float wReal = 1.0f;
                float wImag = 0.0f;

                for (int j = 0; j < s/2; j++) {
                    // t <- ω A[k + j + m/2]
                    Complex A = output.complex(j + k + s/2);
                    float tReal = (wReal * A.real()) - (wImag * A.imag());
                    float tImag = (wReal * A.imag()) + (wImag * A.real());

                    // u <- A[k + j]
                    Complex u = output.complex(j + k);

                    // A[k + j] <- u + t
                    output.complex(j + k).real(u.real() + tReal);
                    output.complex(j + k).imag(u.imag() + tImag);

                    // A[k + j + m/2] <- u – t
                    output.complex(j + k + s/2).real(u.real() - tReal);
                    output.complex(j + k + s/2).imag(u.imag() - tImag);

                    // update w
                    // ω <- ω ωm
                    wReal = (wReal * wmReal) - (wImag * wmImag);
                    wImag  = (wReal * wmImag) + (wImag * wmReal);
                }
            }
        }
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

    static void main(String[] args) {
        IO.println("=====================================");
        IO.println("Example: Fast Fourier Transform (FFT)");
        IO.println("=====================================");

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);

        final int defaultSize = 512;
        int iterations = ITERATIONS;
        ParseArgs parseArgs = new ParseArgs(args);
        ParseArgs.Options options = parseArgs.parseWithDefaults(defaultSize, iterations);

        boolean verbose = options.verbose();
        int size = options.size();
        iterations = options.iterations();

        IO.println("Input Size     = " + size);
        IO.println("Num Iterations = " + iterations);
        IO.println("Sequence Length = " + size);

        // Let's first compute the DFT (Discrete Fourier Transform)
        ComplexArray input = ComplexArray.create(accelerator, size);
        ComplexArray outputSeq = ComplexArray.create(accelerator, size);

        // Initialize
        Random r = new Random(71);
        for (int i = 0; i < size; i++) {
            Complex c = input.complex(i);
            c.real(r.nextFloat());
            c.imag(r.nextFloat());
        }

        List<Long> timersJavaFFT = new ArrayList<>();

        // Run Java sequential version: FFT
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            fftJava(input, outputSeq);
            long end = System.nanoTime();
            timersJavaFFT.add((end-start));
            if (verbose) {
                IO.println("[Timer] Java FFT : " + (end-start));
            }
        }

        // Print Performance Metrics
        final int skip = iterations / 2;
        double averageJavaTimer = computeAverage(timersJavaFFT, skip);

        IO.println("\nAverage elapsed time:");
        IO.println("Average Java-Seq DFT           : " + averageJavaTimer);

        // Write CSV table with all results
        dumpStatsToCSVFile(
                List.of(timersJavaFFT),
                List.of("Java-FFT-SEQ"),
                "table-results-fft-" + size + ".csv");
    }
}
