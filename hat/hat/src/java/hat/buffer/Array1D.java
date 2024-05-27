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

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface Array1D extends Array {
    static <T extends Array1D> StructLayout layout(Class<T> clazz, MemoryLayout memoryLayout, int length) {
        return MemoryLayout.structLayout(
                JAVA_INT.withName("length"),
                MemoryLayout.sequenceLayout(length, memoryLayout).withName("array")
        ).withName(clazz.getSimpleName());
    }

    static <T extends Array1D> T create(Accelerator accelerator, Class<T> clazz, int length, MemoryLayout memoryLayout) {
        StructLayout structLayout = Array1D.layout(clazz, memoryLayout, length);
        T buffer = SegmentMapper.of(accelerator.lookup, clazz, structLayout).allocate(accelerator.backend.arena());
        MemorySegment segment = buffer.memorySegment();
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("length")), length);
        return buffer;
    }


    int length();


}
