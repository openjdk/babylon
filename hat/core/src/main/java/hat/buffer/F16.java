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
package hat.buffer;

// Interface for Floating Point numbers of 16-bits
// Values are stored in a short format.
public interface F16 extends Buffer.Struct {
    String HAT_MAPPING_TYPE = "half";

    short value();
    void value(short value);

    /**
     * Intrinsic for the HAT compiler to create a new half.
     */
    static F16 of(float value) {
        return new F16() {
            @Override
            public short value() {
                return Float.floatToFloat16(value);
            }

            @Override
            public void value(short value) {
            }
        };
    }

    static F16 of(short value) {
        return new F16() {
            @Override
            public short value() {
                return value;
            }

            @Override
            public void value(short value) {
            }
        };
    }

    /**
     * Built-in that can be in HAT Kernel Java code to transform a float into a {@link F16} value.
     */
    static F16 floatToF16(float value) {
        return of(value);
    }

    /**
     * Built-in that can be used in Kernel Code to transform an {@link F16} into a float.
     *
     * @param value {@link F16}
     * @return float
     */
    static float f16ToFloat(F16 value) {
        return Float.float16ToFloat(value.value());
    }

    static F16 add(F16 ha, F16 hb) {
        return F16.of(f16ToFloat(ha) + f16ToFloat(hb));
    }

    static float add(float ha, F16 hb) {
        return ha + f16ToFloat(hb);
    }

    static F16 sub(F16 ha, F16 hb) {
        return F16.of(f16ToFloat(ha) - f16ToFloat(hb));
    }

    static F16 mul(F16 ha, F16 hb) {
        return F16.of(f16ToFloat(ha) * f16ToFloat(hb));
    }

    static F16 div(F16 ha, F16 hb) {
        return F16.of(f16ToFloat(ha) / f16ToFloat(hb));
    }

    default F16 add(F16 ha) {
        return F16.add(this, ha);
    }

    default F16 sub(F16 ha) {
        return F16.sub(this, ha);
    }

    default F16 mul(F16 ha) {
        return F16.mul(this, ha);
    }

    default F16 div(F16 ha) {
        return F16.div(this, ha);
    }
}