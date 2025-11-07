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

import hat.Accelerator;
import hat.ifacemapper.Schema;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public interface F32ArrayPadded extends Buffer {
    int length();
    float array(long idx);
    void array(long idx, float f);

    int ARRAY_OFFSET = 16;

    Schema<F32ArrayPadded> schema = Schema.of(F32ArrayPadded.class, $ -> $
            .arrayLen("length").pad(ARRAY_OFFSET-4).array("array"));

    static F32ArrayPadded create(Accelerator accelerator, int length){
        return schema.allocate(accelerator, length);
    }

    default F32ArrayPadded copyFrom(float[] floats) {
        MemorySegment.copy(floats, 0, Buffer.getMemorySegment(this), JAVA_FLOAT, ARRAY_OFFSET, length());
        return this;
    }

    static F32ArrayPadded createFrom(Accelerator accelerator, float[] arr){
        return create( accelerator, arr.length).copyFrom(arr);
    }

    default F32ArrayPadded copyTo(float[] floats) {
        MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_FLOAT, ARRAY_OFFSET, floats, 0, length());
        return this;
    }

    default Float4.MutableImpl[] float4ArrayView() {
        return null;
    }

    default Float4.MutableImpl float4View(int index) {
        return  null;
    }

    default void storeFloat4View(Float4 v, int index) {
    }

    default Float2.MutableImpl float2View(int index) {
        return  null;
    }

    default void storeFloat2View(Float2 v, int index) {
    }

}
