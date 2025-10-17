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
package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.MappableIface.RO;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HatAsserts;

import java.lang.invoke.MethodHandles;
import java.util.Random;

import static hat.ifacemapper.MappableIface.*;

public class TestBlackscholes {

    @CodeReflection
    public static void blackScholesKernel(@RO KernelContext kc, @WO F32Array call, @WO F32Array put,
                                          @RO F32Array sArray, @RO F32Array xArray, @RO F32Array tArray,
                                          float r, float v) {
        if (kc.x < kc.gsx) {
            float S = sArray.array(kc.x);
            float X = xArray.array(kc.x);
            float T = tArray.array(kc.x);
            float expNegRt = (float) Math.exp(-r * T);
            float d1 = (float) ((Math.log(S / X) + (r + v * v * .5f) * T) / (v * Math.sqrt(T)));
            float d2 = (float) (d1 - v * Math.sqrt(T));
            float cnd1 = CND(d1);
            float cnd2 = CND(d2);
            float value = S * cnd1 - expNegRt * X * cnd2;
            call.array(kc.x, value);
            put.array(kc.x, expNegRt * X * (1 - cnd2) - S * (1 - cnd1));
        }
    }

    @CodeReflection
    public static float CND(float input) {
        float x = input;
        if (input < 0f) {
            x = -input;
        }
        float term = 1f / (1f + (0.2316419f * x));
        float term_pow2 = term * term;
        float term_pow3 = term_pow2 * term;
        float term_pow4 = term_pow2 * term_pow2;
        float term_pow5 = term_pow2 * term_pow3;
        float part1 = (1f / (float)Math.sqrt(2f * 3.1415926535f)) * (float)Math.exp((-x * x) * 0.5f);
        float part2 = (0.31938153f * term) +
                (-0.356563782f * term_pow2) +
                (1.781477937f * term_pow3) +
                (-1.821255978f * term_pow4) +
                (1.330274429f * term_pow5);
        if (input >= 0f) {
            return 1f - part1 * part2;
        }
        return part1 * part2;
    }

    @CodeReflection
    public static void blackScholes(@MappableIface.RO ComputeContext cc, @WO F32Array call, @WO F32Array put, @MappableIface.RO F32Array S, @MappableIface.RO F32Array X, @MappableIface.RO F32Array T, float r, float v) {
        cc.dispatchKernel(call.length(),
                kc -> blackScholesKernel(kc, call, put, S, X, T, r, v)
        );
    }

    static F32Array floatArray(Accelerator accelerator, int size, float low, float high) {
        F32Array array = F32Array.create(accelerator, size);
        Random random = new Random();
        for (int i = 0; i <size; i++) {
            array.array(i, random.nextFloat() * (high - low) + low);
        }
        return array;
    }

    public static void blackScholesKernelSeq(F32Array call, F32Array put, F32Array sArray, F32Array xArray, F32Array tArray, float r, float v) {
        for (int i = 0; i <call.length() ; i++) {
            float S = sArray.array(i);
            float X = xArray.array(i);
            float T = tArray.array(i);
            float expNegRt = (float) Math.exp(-r * T);
            float d1 = (float) ((Math.log(S / X) + (r + v * v * .5f) * T) / (v * Math.sqrt(T)));
            float d2 = (float) (d1 - v * Math.sqrt(T));
            float cnd1 = CND(d1);
            float cnd2 = CND(d2);
            float value = S * cnd1 - expNegRt * X * cnd2;
            call.array(i, value);
            put.array(i, expNegRt * X * (1 - cnd2) - S * (1 - cnd1));
        }
    }

    @HatTest
    public void testBlackscholes() {
        int size = 1024;
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var call = F32Array.create(accelerator, size);
        var put = F32Array.create(accelerator, size);
        for (int i = 0; i < put.length(); i++) {
            put.array(i, i);
            call.array(i, i);
        }

        var S = floatArray(accelerator, size,1f, 100f);
        var X = floatArray(accelerator, size,1f, 100f);
        var T = floatArray(accelerator,size, 0.25f, 10f);
        float r = 0.02f;
        float v = 0.30f;

        accelerator.compute(cc -> blackScholes(cc, call, put, S, X, T, r, v));

        var seqCall = F32Array.create(accelerator, size);
        var seqPut = F32Array.create(accelerator, size);
        for (int i = 0; i < seqCall.length(); i++) {
            seqCall.array(i, i);
            seqPut.array(i, i);
        }

        blackScholesKernelSeq(seqCall, seqPut, S, X, T, r, v);

        for (int i = 0; i < call.length(); i++) {
            HatAsserts.assertEquals(seqCall.array(i), call.array(i), 0.01f);
            HatAsserts.assertEquals(seqPut.array(i), put.array(i), 0.01f);
        }
    }
}
