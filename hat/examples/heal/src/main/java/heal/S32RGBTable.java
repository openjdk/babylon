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
package heal;

import hat.Accelerator;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.buffer.Table;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface S32RGBTable extends Table<S32RGBTable.RGB> {

    interface RGB {
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("r"),
                JAVA_INT.withName("g"),
                JAVA_INT.withName("b")
        ).withName("RGB");
        int r();

        int g();

        int b();

        void r(int r);
        void g(int g);

        void b(int b);
    }
    StructLayout layout = MemoryLayout.structLayout(  JAVA_INT.withName("length"),

            MemoryLayout.sequenceLayout(0, S32RGBTable.RGB.layout).withName("rgb")).withName(S32XYTable.class.getSimpleName());

    static S32RGBTable create(BufferAllocator bufferAllocator, int length) {
        S32RGBTable table = bufferAllocator.allocate(
                SegmentMapper.ofIncomplete(MethodHandles.lookup(), S32RGBTable.class,layout,length));
        Buffer.setLength(table,length);
        return table;
    }


    RGB rgb(long idx);

}
