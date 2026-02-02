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
import hat.HATMath;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.test.TestDFT.ArrayComplex.Complex;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import optkl.ifacemapper.MappableIface.WO;
import optkl.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public class TestDFT {

    public interface ArrayComplex extends Buffer {
        int length();

        interface Complex extends Struct {
            float real();
            float imag();
            void real(float real);
            void imag(float imag);
        }

        Complex complex(long index);

        Schema<ArrayComplex> schema = Schema.of(ArrayComplex.class,
                complex ->
                        complex.arrayLen("length")
                                .array("complex",
                                        array -> array.fields("real", "imag")));

        static ArrayComplex create(Accelerator accelerator, int length) {
            return schema.allocate(accelerator, length);
        }
    }

    @Reflect
    private static void dftKernel(@RW KernelContext kc,
                                  @RO ArrayComplex input,
                                  @WO ArrayComplex output) {
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

    @Reflect
    private static void dftCompute(@RW ComputeContext cc,
                                   @RO ArrayComplex input,
                                   @WO ArrayComplex output) {
        var range = NDRange.of1D(input.length(), 128);
        cc.dispatchKernel(range, kernelContext -> dftKernel(kernelContext, input, output));
    }

    @HatTest
    public void testDFTWithOwnDS() {

        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);
        final int size = 256;

        ArrayComplex input = ArrayComplex.create(accelerator, size);
        ArrayComplex outputSeq = ArrayComplex.create(accelerator, size);
        ArrayComplex outputHAT = ArrayComplex.create(accelerator, size);

        accelerator.compute((@Reflect Compute) computeContext -> dftCompute(computeContext, input, outputHAT));

        dftJava(input, outputSeq);

        for (int i = 0; i < outputSeq.length(); i++) {
            HATAsserts.assertEquals(outputSeq.complex(i).real(), outputHAT.complex(i).real(), 0.001f);
            HATAsserts.assertEquals(outputSeq.complex(i).imag(), outputHAT.complex(i).imag(), 0.001f);
        }
    }

}
