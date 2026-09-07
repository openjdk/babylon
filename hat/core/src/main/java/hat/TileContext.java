/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package hat;

import hat.buffer.Tensor2DF16;
import hat.buffer.Tensor2DF32;
import hat.buffer.TensorF32;
import hat.types.Tile;

import java.util.stream.IntStream;

/**
 * Interface to provide an execution-context and memory operations for the
 * implementation of a Tile Programming Model in HAT.
 *
 * <p>A Tile programming model expresses kernels (compute methods to be offloaded
 * and accelerated on specialized hardware such as GPUs) in terms of multidimensional
 * arrays, called {@link Tile tiles}, rather than individual accelerator threads.
 * </p>
 *
 * <p>Each tile-program instance obtains its identifier in the thread-block dispatch
 * by calling the methods {@link #BIDX()}, {@link #BIDY()}, and {@link #BIDZ()}.
 * Based on these positions, developers can load tiles, operate on tiles and store
 * the resulting tiles to tensors.</p>
 *
 * <p>For example, the following code-snippet shows a tile version of a vector addition:</p>
 * <p>
 * {@snippet :
 * final int pid = TileContext.BIDX();
 * Tile a = TileContext.load(inputTensorA, pid, tileSize);
 * Tile b = TileContext.load(inputTensorB, pid, tileSize);
 * Tile result = TileOp.add(a, b);
 * TileContext.store(output, pid, result);
 *}
 *
 * @apiNote The block-index, load and store operations are implemented as tile-kernel
 * intrinsics. Invocation to these methods are replaced by the HAT Transformer during
 * Just-In-Time Compilation and lowered to the corresponding low-level operations
 * during the code generation.
 *
 * @apiNote Currently, there is no Java implementation for the following block-index,
 * load and store operations. However, this is planned for future iterations of the
 * Tile Programming Model implementation within HAT.
 *
 */
public interface TileContext {

    static int BIDX() {
        return 0;
    }

    static int BIDY() {
        return 0;
    }

    static int BIDZ() {
        return 0;
    }

    static Tile load(TensorF32 buffer, int pid, int tileSize) {
        return null;
    }

    static Tile load(Tensor2DF32 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        return null;
    }

    static Tile load(Tensor2DF16 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        return null;
    }

    static Tile load(TensorF32 buffer, TileIndex2D tileIndex2D, Shape shape) {
        return null;
    }

    static void store(TensorF32 buffer, int pid, Tile result) {

    }

    static void store(TensorF32 buffer, TileIndex2D tileIndex, Tile result) {

    }

    static void store(Tensor2DF32 buffer, TileIndex2D tileIndex, Tile result) {

    }

    static TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    static Shape shape(int tm) {
        return new Shape(tm, 1, 1);
    }

    static Shape shape(int tm, int tn) {
        return new Shape(tm, tn, 1);
    }

    static Shape shape(int tm, int tn, int tk) {
        return new Shape(tm, tn, tk);
    }

    static int[] irange(int startIndex, int endIndex) {
        return IntStream.range(0, (endIndex - startIndex)).map(i -> startIndex + i).toArray();
    }
}
