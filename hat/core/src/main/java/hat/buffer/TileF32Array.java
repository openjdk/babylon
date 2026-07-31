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

import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;
import optkl.util.carriers.ArenaAndLookupCarrier;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface TileF32Array extends Buffer {
    @Reflect
    default void schema() {
        array(length());
    }

    int length();

    float array(long idx);

    void array(long idx, float f);

    long ARRAY_OFFSET = JAVA_INT.byteSize();

    Schema<TileF32Array> schema = Schema.of(TileF32Array.class, $ -> $
            .arrayLen("length").pad(12).array("array"));

    static TileF32Array create(ArenaAndLookupCarrier cc, int length) {
        return BoundSchema.of(cc ,schema, length).allocate();
    }

    default TileF32Array copyFrom(float[] floats) {
        MemorySegment.copy(floats, 0, MappableIface.getMemorySegment(this), JAVA_FLOAT, ARRAY_OFFSET, length());
        return this;
    }

    static TileF32Array createFrom(ArenaAndLookupCarrier cc, float[] arr) {
        return create(cc, arr.length).copyFrom(arr);
    }

    default TileF32Array copyTo(float[] floats) {
        MemorySegment.copy(MappableIface.getMemorySegment(this), JAVA_FLOAT, ARRAY_OFFSET, floats, 0, length());
        return this;
    }

    default float[] arrayView() {
        float[] arr = new float[this.length()];
        this.copyTo(arr);
        return arr;
    }
}
