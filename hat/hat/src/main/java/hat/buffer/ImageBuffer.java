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

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferUShort;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public interface ImageBuffer extends IncompleteBuffer {
    static StructLayout createLayout(Class iface, ValueLayout valueLayout) {
        return MemoryLayout.structLayout(
                JAVA_INT.withName("width"),
                JAVA_INT.withName("height"),
                JAVA_INT.withName("elementsPerPixel"),
                JAVA_INT.withName("bufferedImageType"),
                MemoryLayout.sequenceLayout(0, valueLayout).withName("data")
        ).withName(iface.getSimpleName());
    }
    /* BufferedImage types
                 TYPE_INT_RGB, TYPE_INT_ARGB, TYPE_INT_ARGB_PRE, TYPE_INT_BGR,
                  TYPE_3BYTE_BGR, TYPE_4BYTE_ABGR, TYPE_4BYTE_ABGR_PRE, TYPE_BYTE_GRAY,
                   TYPE_BYTE_BINARY, TYPE_BYTE_INDEXED, TYPE_USHORT_GRAY,
                    TYPE_USHORT_565_RGB, TYPE_USHORT_555_RGB, TYPE_CUSTOM

     */
    static <T extends ImageBuffer> T create(Accelerator accelerator, Class<T> iface,StructLayout structLayout, int width, int height, int bufferedImageType, int elementsPerPixel) {
        T rgba = SegmentMapper.ofIncomplete(accelerator.lookup, iface, structLayout, width * height * elementsPerPixel).allocate(accelerator.backend.arena());
        MemorySegment segment = Buffer.getMemorySegment(rgba);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("width")), width);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("height")), height);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("elementsPerPixel")), elementsPerPixel);
        segment.set(JAVA_INT, structLayout.byteOffset(MemoryLayout.PathElement.groupElement("bufferedImageType")), bufferedImageType);
        return rgba;
    }

    int elementsPerPixel();

    // void elementsPerPixel(int elementsPerPixel);

    int bufferedImageType();

    // void bufferedImageType(int bufferedImageType);
    @SuppressWarnings("unchecked")
    default <T extends ImageBuffer> T syncToRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
        switch (dataBuffer) {
            case DataBufferUShort arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_SHORT, 16L, arr.getData(), 0, arr.getData().length);
            case DataBufferInt arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_INT, 16L, arr.getData(), 0, arr.getData().length);
            case DataBufferByte arr ->
                    MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_BYTE, 16L, arr.getData(), 0, arr.getData().length);
            default -> throw new IllegalStateException("Unexpected value: " + dataBuffer);
        }
        return (T) this;
    }

    default <T extends ImageBuffer> T syncToRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncToRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }

    @SuppressWarnings("unchecked")
    default <T extends ImageBuffer> T syncFromRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
        switch (dataBuffer) {
            case DataBufferInt arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_INT, 16L, arr.getData().length);
            case DataBufferByte arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_BYTE, 16L, arr.getData().length);
            case DataBufferUShort arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_SHORT, 16L, arr.getData().length);
            default -> throw new IllegalStateException("Unexpected value: " + dataBuffer);
        }
        return (T) this;
    }

    default <T extends ImageBuffer> T syncFromRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncFromRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }

    int width();

    int height();

    default int size() {
        return width() * height();
    }
}
