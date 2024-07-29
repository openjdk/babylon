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
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface F32Array extends Buffer {
    int length();
    @BoundBy("length")
    float array(long idx);
    void array(long idx, float f);

    Schema<F32Array> schema = Schema.of(F32Array.class, s32Array->s32Array
            .arrayLen("length").array("array"));

    static F32Array create(Accelerator accelerator, int length){
        return schema.allocate(accelerator, length);
    }
    default F32Array copyFrom(float[] floats) {
        MemorySegment.copy(floats, 0, Buffer.getMemorySegment(this), JAVA_FLOAT, 4, length());
        return this;
    }
    static F32Array createFrom(Accelerator accelerator, float[] arr){
        return create( accelerator, arr.length).copyFrom(arr);
    }


    default F32Array copyTo(float[] floats) {
        MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_FLOAT, 4, floats, 0, length());
        return this;
    }

}
