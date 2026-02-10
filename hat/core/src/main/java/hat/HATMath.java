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
package hat;

import hat.types.F16;

public class HATMath {

    public static final float PI = (float) Math.PI;

    // Binary Operations
    public static float max(float a, float b) {
        return Math.max(a, b);
    }

    public static F16 max(F16 a, F16 b) {
        float fa = F16.f16ToFloat(a);
        float fb = F16.f16ToFloat(b);
        return F16.floatToF16(Math.max(fa, fb));
    }

    public static float min(float a, float b) {
        return Math.min(a, b);
    }

    public static F16 min(F16 a, F16 b) {
        float fa = F16.f16ToFloat(a);
        float fb = F16.f16ToFloat(b);
        return F16.floatToF16(Math.min(fa, fb));
    }

    // Unary operations
    public static float cosf(float a) {
        return (float) Math.cos(a);
    }

    public static double cos(double a) {
        return Math.cos(a);
    }

    public static float native_cosf(float a) {
        return (float) Math.cos(a);
    }

    public static float expf(float a) {
        return (float) Math.exp(a);
    }

    public static double exp(double a) {
        return Math.exp(a);
    }

    public static F16 exp(F16 a) {
        float fa = F16.f16ToFloat(a);
        return F16.floatToF16((float) Math.exp(fa));
    }

    public static float native_exp(float a) {
        return (float) Math.exp(a);
    }


    public static float sinf(float a) {
        return (float) Math.sin(a);
    }

    public static double sin(double a) {
        return Math.sin(a);
    }

    public static float native_sinf(float a) {
        return (float) Math.sin(a);
    }

    public static float sqrtf(float a) {
        return (float) Math.sqrt(a);
    }
    public static double sqrt(double a) {
        return Math.sqrt(a);
    }

    public static float tanf(float a) {
        return (float) Math.tan(a);
    }

    public static double tan(double a) {
        return Math.tan(a);
    }

    public static float native_tanf(float a) {
        return (float) Math.tan(a);
    }

    private HATMath() {}
}
