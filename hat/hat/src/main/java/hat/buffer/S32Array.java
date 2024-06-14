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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface S32Array extends Array1D {
    StructLayout layout  = Array1D.getLayout(S32Array.class, JAVA_INT);
    static S32Array create(BufferAllocator bufferAllocator, int length) {
        return Array1D.create(bufferAllocator, S32Array.class,layout, length);
    }

    static S32Array create(BufferAllocator bufferAllocator, int[] source) {
        return create(bufferAllocator, source.length).copyfrom(source);
    }

    int array(long idx);

    void array(long idx, int f);

    default S32Array copyfrom(int[] floats) {
        MemorySegment.copy(floats, 0, Buffer.getMemorySegment(this), JAVA_INT, 4, length());
        return this;
    }

    default S32Array copyTo(int[] floats) {
        MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_INT, 4, floats, 0, length());
        return this;
    }
}
