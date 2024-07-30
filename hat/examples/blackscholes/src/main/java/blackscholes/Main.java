/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package blackscholes;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.F32Array;
import java.util.Random;

import java.lang.runtime.CodeReflection;

public class Main {
    static Random rand;

    @CodeReflection
    public static void blackScholesKernel(KernelContext kc, F32Array call, F32Array put, F32Array sArray, F32Array xArray, F32Array tArray, float r, float v) {
        if (kc.x<kc.maxX){
            float S = sArray.array(kc.x);
            float X = xArray.array(kc.x);
            float T = tArray.array(kc.x);
            float expNegRt = (float) Math.exp(-r * T);
            float d1 = (float) ((Math.log(S / X) + (r + v * v * .5f) * T) / (v * Math.sqrt(T)));
            float d2 = (float) (d1 - v * Math.sqrt(T));
            float value = (float) (S * CND(d1) - X * expNegRt * CND(d2));
            call.array(kc.x, value);
            put.array(kc.x, value + expNegRt - S);
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
        } else {
            return part1 * part2;
        }
    }

    @CodeReflection
    public static void blackScholes(ComputeContext cc, F32Array call, F32Array put, F32Array S, F32Array X, F32Array T, float r, float v) {
        cc.dispatchKernel(call.length(),
                kc -> blackScholesKernel(kc, call, put, S, X, T, r, v)
        );
    }

    static F32Array floatArray(float low, float high, Accelerator accelerator) {
        F32Array array = F32Array.create(accelerator, 50);
        for (int i = 0; i < array.length(); i++) {
            array.array(i, rand.nextFloat() * (high - low) + low);
        }
        return array;
    }

    public static void main(String[] args) {
        rand = new Random();
        var lookup = java.lang.invoke.MethodHandles.lookup();
        var accelerator = new Accelerator(lookup, Backend.FIRST);//new JavaMultiThreadedBackend());
        var call = F32Array.create(accelerator, 50);
        for (int i = 0; i < call.length(); i++) {
            call.array(i, i);
        }

        var put = F32Array.create(accelerator, 50);
        for (int i = 0; i < put.length(); i++) {
            put.array(i, i);
        }

        var S = floatArray(5, 30, accelerator);
        var X = floatArray(1, 100, accelerator);
        var T = floatArray(0.25f, 10, accelerator);
        float r = 0.02f;
        float v = 0.30f;

        accelerator.compute(
                cc -> Main.blackScholes(cc, call, put, S, X, T, r, v)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < call.length(); i++) {
            System.out.println("S=" + S.array(i) + "\t X=" + X.array(i) + "\t T=" + T.array(i) + "\t call option price = " + call.array(i) + "\t\t put option price = " + put.array(i));
        }
    }
}
