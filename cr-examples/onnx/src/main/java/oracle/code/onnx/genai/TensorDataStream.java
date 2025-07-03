/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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
package oracle.code.onnx.genai;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.util.stream.LongStream;
import oracle.code.onnx.Tensor;

public final class TensorDataStream {


    private final Arena arena;
    private final MemorySegment data;
    private long offset;

    public TensorDataStream(Arena arena, String dataFilePath) throws IOException {
        this.arena = arena;
        try (var dataFile = new RandomAccessFile(dataFilePath, "r")) {
            this.data = dataFile.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, dataFile.length(), arena);
        }
    }

    public TensorDataStream(Arena arena, MemorySegment data) {
        this.arena = arena;
        this.data = data;
    }

    public <T> Tensor<T> nextTensor(Tensor.ElementType type, long... shape) {
        long size = type.bitSize() * LongStream.of(shape).reduce(1l, (a, b) -> a * b) / 8l;
        Tensor<T> tensor = new Tensor<>(arena, data.asSlice(offset, size), type, shape);
        offset += size;
        return tensor;
    }
}
