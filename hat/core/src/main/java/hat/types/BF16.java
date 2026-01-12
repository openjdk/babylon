/*
 * Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.types;

public interface BF16 extends HAType {

    short value();
    void value(short value);

    static BF16 of(float value) {
        return new BF16() {
            @Override
            public short value() {
                int bits = Float.floatToRawIntBits(value);
                short sign_bit = (short)((bits & 0x8000_0000) >> 16);

                // For round to nearest even, determining whether or not to
                // round up (in magnitude) is a function of the least
                // significant bit (LSB), the next bit position (the round
                // position), and the sticky bit (whether there are any
                // nonzero bits in the exact result to the right of the round
                // digit). An increment occurs in three cases:
                //
                // LSB  Round Sticky
                // 0    1     1
                // 1    1     0
                // 1    1     1
                // See "Computer Arithmetic Algorithms," Koren, Table 4.9

                int lsb    = bits & 0x1_0000;
                int round  = bits & 0x0_8000;
                int sticky = bits & 0x0_7FFF;
                if (round != 0 && ((lsb | sticky) != 0 )) {
                    bits += 0x1_0000;
                }
                return (short) (((bits >> 16 ) | sign_bit) & 0xffff);
            }

            @Override
            public void value(short value) {
            }
        };
    }

    static BF16 of(short value) {
        return new BF16() {
            @Override
            public short value() {
                return value;
            }

            @Override
            public void value(short value) {
            }
        };
    }

    static BF16 float2bfloat16(float value) {
        return of(value);
    }

    static float bfloat162float(BF16 value) {
        return Float.intBitsToFloat(value.value() << 16);
    }

    static BF16 add(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) + bfloat162float(hb));
    }

    static BF16 add(float f32, BF16 hb) {
        return BF16.of(f32 + bfloat162float(hb));
    }

    static BF16 sub(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) - bfloat162float(hb));
    }

    static BF16 sub(float f32, BF16 hb) {
        return BF16.of(f32 - bfloat162float(hb));
    }

    static BF16 sub(BF16 hb, float f32) {
        return BF16.of(bfloat162float(hb) - f32);
    }

    static BF16 mul(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) * bfloat162float(hb));
    }

    static BF16 mul(float f32, BF16 hb) {
        return BF16.of(f32 * bfloat162float(hb));
    }

    static BF16 div(BF16 ha, BF16 hb) {
        return BF16.of(bfloat162float(ha) / bfloat162float(hb));
    }

    static BF16 div(float f32, BF16 hb) {
        return BF16.of(f32 / bfloat162float(hb));
    }

    static BF16 add(BF16 hb, float f32) {
        return BF16.of(bfloat162float(hb) / f32);
    }

    default BF16 add(BF16 ha) {
        return BF16.add(this, ha);
    }

    default BF16 sub(BF16 ha) {
        return BF16.sub(this, ha);
    }

    default BF16 mul(BF16 ha) {
        return BF16.mul(this, ha);
    }

    default BF16 div(BF16 ha) {
        return BF16.div(this, ha);
    }

}
