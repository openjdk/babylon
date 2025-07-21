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

import java.awt.image.BufferedImage;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.DataBufferUShort;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public interface ImageIfaceBuffer<T extends ImageIfaceBuffer<?>> extends Buffer {
    @SuppressWarnings("unchecked")
    default T syncFromRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
        switch (dataBuffer) {
            case DataBufferInt arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_INT, 16L, arr.getData().length);
            case DataBufferByte arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_BYTE, 16L, arr.getData().length);
            case DataBufferUShort arr ->
                    MemorySegment.copy(arr.getData(), 0, Buffer.getMemorySegment(this), JAVA_SHORT, 16L, arr.getData().length);
            default -> throw new IllegalStateException("Unexpected value: " + dataBuffer);
        }
        return (T)this;
    }

    default T syncFromRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncFromRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }

    @SuppressWarnings("unchecked")
    default T syncToRasterDataBuffer(DataBuffer dataBuffer) { // int[], byte[], short[]
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

    default T syncToRaster(BufferedImage bufferedImage) { // int[], byte[], short[]
        return syncToRasterDataBuffer(bufferedImage.getRaster().getDataBuffer());
    }
}
