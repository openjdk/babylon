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
package life;

import hat.Accelerator;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface CellGrid extends Buffer {
    byte ALIVE = (byte) 0xff;
    byte DEAD = 0x00;
    /*
     * struct CellGrid{
     *     int width;
     *     int height;
     *     byte[width*height*2] cellArray;
     *  }
     */
    int width();

    int height();

    byte cell(long idx);

    void cell(long idx, byte b);

    Schema<CellGrid> schema = Schema.of(CellGrid.class, lifeData -> lifeData
            .arrayLen("width", "height").stride(2).array("cell")
    );

    static CellGrid create(Accelerator accelerator, int width, int height) {
        return BoundSchema.of(accelerator, schema, width, height).allocate();
    }

    ValueLayout valueLayout = JAVA_BYTE;
    long headerOffset = JAVA_INT.byteSize() * 2;

    default void copySliceTo(byte[] bytes, int to) {
        long offset = headerOffset + (long) to * valueLayout.byteSize();
        MemorySegment.copy(MappableIface.getMemorySegment(this), valueLayout, offset, bytes, 0, width() * height());

    }

   // default int wxh() {
      //  return width() * height();
   // }
}
