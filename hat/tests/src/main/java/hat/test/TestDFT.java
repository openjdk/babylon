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
import hat.device.DeviceSchema;
import hat.device.NonMappableIface;
import hat.test.annotation.HatTest;
import hat.test.exceptions.HATAsserts;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface.RO;
import optkl.ifacemapper.MappableIface.RW;
import optkl.ifacemapper.MappableIface.WO;
import optkl.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

import static hat.test.TestDFT.ArrayComplex.Complex;
import static hat.test.TestDFT.ArrayComplex.create;

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
            return BoundSchema.of(accelerator, schema, length).allocate();
        }
    }

    @Reflect
    private static void dftKernel(@RW KernelContext kc,
                                  @RO ArrayComplex input,
                                  @WO ArrayComplex output) {
        int size = input.length();
        int idx = kc.gix;
        if (idx < kc.gsx) {
            float sumReal = 0.0f;
            float sumImag = 0.0f;
            for (int k = 0; k < size; k++) {
                float angle = -2 * HATMath.PI * ((k * idx) % size) / size;
                Complex complexInput = input.complex(k);
                float cReal = HATMath.cosf(angle);
                float cImag = HATMath.sinf(angle);
                sumReal += (complexInput.real() * cReal) - (complexInput.imag() * cImag);
                sumImag += (complexInput.real() * cImag) + (complexInput.imag() * cReal);
            }
            Complex complexOutput = output.complex(idx);
            complexOutput.real(sumReal);
            complexOutput.imag(sumImag);
        }
    }

    private static void dftJava(ArrayComplex input, ArrayComplex output) {
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
        final int size = 8192;
        ArrayComplex input = create(accelerator, size);
        ArrayComplex outputSeq = create(accelerator, size);
        ArrayComplex outputHAT = create(accelerator, size);
        accelerator.compute((@Reflect Compute) computeContext -> dftCompute(computeContext, input, outputHAT));
        dftJava(input, outputSeq);
        for (int i = 0; i < outputSeq.length(); i++) {
            HATAsserts.assertEquals(outputSeq.complex(i).real(), outputHAT.complex(i).real(), 0.001f);
            HATAsserts.assertEquals(outputSeq.complex(i).imag(), outputHAT.complex(i).imag(), 0.001f);
        }
    }

    // Simple test to check User Data Structures in Private Memory
    public interface ArrayComplexPrivate extends NonMappableIface {
        interface PrivateComplex extends Struct {
            float real();
            float imag();
            void real(float real);
            void imag(float imag);
        }
        PrivateComplex complex(long index);
        DeviceSchema<ArrayComplexPrivate> schema =
                DeviceSchema.of(ArrayComplexPrivate.class,
                        complex ->
                                complex.withArray("complex", 128)
                                        .withDeps(PrivateComplex.class,
                                                c2 -> c2.withFields("real", "imag")));
        static ArrayComplexPrivate createPrivate() {
            return null;
        }
    }

    @Reflect
    private static void testPrivateDS(@RW KernelContext kc,
                                      @RO ArrayComplex input,
                                      @WO ArrayComplex output) {
        int idx = kc.gix;
        ArrayComplexPrivate priv = ArrayComplexPrivate.createPrivate();
        ArrayComplexPrivate.PrivateComplex complex = priv.complex(0);
        complex.real(1.0f);
        complex.imag(2.0f);
        Complex complexOutput = output.complex(idx);
        complexOutput.real(complex.real());
        complexOutput.imag(complex.imag());
    }


    @Reflect
    private static void complexNumbersInPrivate(@RW ComputeContext cc,
                                   @RO ArrayComplex input,
                                   @WO ArrayComplex output) {
        var range = NDRange.of1D(input.length(), 128);
        cc.dispatchKernel(range, kernelContext -> testPrivateDS(kernelContext, input, output));
    }

    @HatTest
    public void complexNumbersInPrivate() {
        var lookup = MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);
        final int size = 8192;
        ArrayComplex input = create(accelerator, size);
        ArrayComplex outputSeq = create(accelerator, size);
        ArrayComplex outputHAT = create(accelerator, size);
        accelerator.compute((@Reflect Compute) computeContext -> complexNumbersInPrivate(computeContext, input, outputHAT));
        for (int i = 0; i < outputSeq.length(); i++) {
            HATAsserts.assertEquals(1.0f, outputHAT.complex(i).real(), 0.001f);
            HATAsserts.assertEquals(2.0f, outputHAT.complex(i).imag(), 0.001f);
        }
    }

}
