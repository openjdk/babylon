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

import hat.Accelerator;
import hat.ifacemapper.Schema;

public interface F16Array extends Buffer {
    int length();

    // Interface for Floating Point numbers of 16-bits
    // Values are stored in a short format.
    interface F16 extends Struct {
        String HAT_MAPPING_TYPE = "half";

        short value();
        void value(short value);

        // Intrinsic for the HAT compiler to create a
        // new half
        String F16_INSTANCE_OF = "of";
        static F16 of(float value) {
            return new F16() {
                @Override
                public short value() {
                    return floatToF16(value);
                }

                @Override
                public void value(short value) {
                }
            };
        }

        static short floatToF16(float value) {
            return Float.floatToFloat16(value);
        }

        static float f16ToFloat(short value) {
            return Float.float16ToFloat(value);
        }

        static F16 add(F16 ha, F16 hb) {
            return F16.of(f16ToFloat(ha.value()) + f16ToFloat(hb.value()));
        }

        static F16 sub(F16 ha, F16 hb) {
            return F16.of(f16ToFloat(ha.value()) - f16ToFloat(hb.value()));
        }

        static F16 mul(F16 ha, F16 hb) {
            return F16.of(f16ToFloat(ha.value()) * f16ToFloat(hb.value()));
        }

        static F16 div(F16 ha, F16 hb) {
            return F16.of(f16ToFloat(ha.value()) / f16ToFloat(hb.value()));
        }
    }

    F16 array(long index);

    Schema<F16Array> schema = Schema.of(F16Array.class, f16array ->
            f16array.arrayLen("length")
                    .array("array",
                            half -> half.fields("value")));

    static F16Array create(Accelerator accelerator, int length){
        return schema.allocate(accelerator, length);
    }
}
