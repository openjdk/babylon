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
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface Array2D extends Array {
    static <T extends Array2D> StructLayout getLayout(Class<T> iface, MemoryLayout memoryLayout) {
        return MemoryLayout.structLayout(
                JAVA_INT.withName("width"),
                JAVA_INT.withName("height"),
                MemoryLayout.sequenceLayout(0, memoryLayout).withName("array")
        ).withName(iface.getSimpleName());
    }

    static <T extends Array2D> T create(BufferAllocator bufferAllocator, Class<T> clazz, StructLayout structLayout,int width, int height) {

        T buffer = bufferAllocator.allocate(SegmentMapper.ofIncomplete(MethodHandles.lookup(), clazz, structLayout, (long) width * height));
        MemorySegment segment = Buffer.getMemorySegment(buffer);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("width")), width);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("height")), height);
        return buffer;
    }

    int width();

    int height();

    default int size() {
        return width() * height();
    }
}
