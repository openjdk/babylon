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
package hat.buffer;

import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;
import optkl.util.carriers.ArenaAndLookupCarrier;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface Tensor2DF16 extends Buffer, Half {

    int m();
    int n();
    void array(long index, short value);
    short array(long index);

    long ARRAY_HEADER_OFFSET = JAVA_INT.byteSize() * 128;

    Schema<Tensor2DF16> schema = Schema.of(Tensor2DF16.class, tensor ->
            tensor.arrayLen("m", "n")
                    .pad(120)
                    .array("array"));

    static Tensor2DF16 create(ArenaAndLookupCarrier cc, int m, int n) {
        return BoundSchema.of(cc ,schema, m, n).allocate();
    }

    default Tensor2DF16 copyFrom(float[] floats) {
        MemorySegment.copy(floats, 0, MappableIface.getMemorySegment(this), JAVA_FLOAT, ARRAY_HEADER_OFFSET, m() * n());
        return this;
    }

    default Tensor2DF16 copyTo(float[] floats) {
        MemorySegment.copy(MappableIface.getMemorySegment(this), JAVA_FLOAT, ARRAY_HEADER_OFFSET, floats, 0, m() * n());
        return this;
    }

    default float[] arrayView() {
        float[] arr = new float[this.m() * this.n()];
        this.copyTo(arr);
        return arr;
    }
}
