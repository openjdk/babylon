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
 * Based on these ids, developers can load tiles, operate on tiles and store
 * the resulting tiles to tensors in parallel.</p>
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
 * during the code generation (e.g., CUDA Tile C++).
 *
 * @apiNote Currently, there is no Java implementation for the following block-index,
 * load and store operations. However, this is planned for future iterations of the
 * Tile Programming Model implementation within HAT.
 *
 */
public interface TileContext {

    /**
     * Returns the thead-block index of the current tile program along the first dimension.
     *
     * @return the block index along the first dimension.
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static int BIDX() {
        throw  new UnsupportedOperationException("TileContext.BIDX() is a compiler intrinsic.");
    }

    /**
     * Returns the thead-block index of the current tile program along the second dimension.
     *
     * @return the block index along the second dimension.
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static int BIDY() {
        throw new UnsupportedOperationException("TileContext.BIDY() is a compiler intrinsic.");
    }

    /**
     * Returns the thead-block index of the current tile program along the third dimension.
     *
     * @return the block index along the third dimension.
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static int BIDZ() {
        throw new UnsupportedOperationException("TileContext.BIDZ() is a compiler intrinsic.");
    }

    /**
     * Loads one dimensional tile from a single precision tensor {@link TensorF32}.
     *
     * @param buffer: tensor from which to load the tile
     * @param pid: index of the tile within the input tensor
     * @param tileSize: indicates the number of elements in the tile.
     *                For better performance, use a multiple of 16.
     * @return the loaded {@link Tile}
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static Tile load(TensorF32 buffer, int pid, int tileSize) {
        throw  new UnsupportedOperationException("TileContext.load() is a compiler intrinsic.");
    }

    /**
     * Loads a tile from a two-dimensional single precision tensor {@link TensorF32}.
     *
     * @param buffer: tensor from which to load the tile
     * @param tileIndex2D: the two-dimension coordinates of the tile
     * @param tileSize: indicates the shape (size for each dimension) of the tile to load.
     *                For better performance, use a multiple of 16 for each dimension.
     * @return the loaded {@link Tile}
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static Tile load(Tensor2DF32 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        throw  new UnsupportedOperationException("TileContext.load() is a compiler intrinsic.");
    }

    /**
     * Loads a tile from a two-dimensional single precision tensor {@link hat.buffer.TensorF16}.
     *
     * @param buffer: tensor from which to load the tile
     * @param tileIndex2D: the two-dimension coordinates of the tile
     * @param tileSize: indicates the shape (size for each dimension) of the tile to load.
     *                For better performance, use a multiple of 16 for each dimension.
     * @return the loaded {@link Tile}
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static Tile load(Tensor2DF16 buffer, TileIndex2D tileIndex2D, Shape tileSize) {
        throw new UnsupportedOperationException("TileContext.load() is a compiler intrinsic.");
    }

    /**
     * Stores a tile in single-precision (float) into a single precision tensor.
     *
     * @param buffer the destination tensor.
     * @param pid the thread-block id
     * @param result the resulting tile to store.
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static void store(TensorF32 buffer, int pid, Tile result) {
        throw  new UnsupportedOperationException("TileContext.store() is a compiler intrinsic.");
    }

    /**
     * Stores a two-dimensional in single precision (float) into a two-dimensional tensor.
     *
     * @param buffer the destination tensor
     * @param tileIndex the two-dimensional thread-block index in which store the tile
     * @param result the tile to store
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static void store(Tensor2DF32 buffer, TileIndex2D tileIndex, Tile result) {
        throw  new UnsupportedOperationException("TileContext.store() is a compiler intrinsic.");
    }

    /**
     * Creates a two-dimensional thread-index.
     *
     * @param bidx the thread-block index for the first dimension.
     * @param bidy the thread-block index for the second dimension.
     * @return result two-dimension thread-block index.
     * @throws UnsupportedOperationException if the method is invoked as ordinary Java code
     */
    static TileIndex2D index(int bidx, int bidy) {
        return new TileIndex2D(bidx, bidy);
    }

    /**
     * Creates a one-dimensional tile shape.
     *
     * @param tm the extent of the first dimension
     * @return the shape {@code (tm, 1, 1)}
     */
    static Shape shape(int tm) {
        return new Shape(tm, 1, 1);
    }

    /**
     * Creates a two-dimensional tile shape.
     *
     * @param tm the extent of the first dimension
     * @param tn the extent of the second dimension
     * @return the shape {@code (tm, tn, 1)}
     */
    static Shape shape(int tm, int tn) {
        return new Shape(tm, tn, 1);
    }

    /**
     * Creates a three-dimensional tile shape.
     *
     * @param tm the extent of the first dimension
     * @param tn the extent of the second dimension
     * @param tk the extent of the third dimension
     * @return the shape {@code (tm, tn, tk)}
     */
    static Shape shape(int tm, int tn, int tk) {
        return new Shape(tm, tn, tk);
    }

    /**
     * Returns a range of size (endIndex - startIndex).
     *
     * @param startIndex the inclusive lower bound
     * @param endIndex the exclusive upper bound
     * @return the integers in the half-open range {@code [starIndex, endIndex)}
     */
    static int[] irange(int startIndex, int endIndex) {
        return IntStream.range(startIndex, endIndex).toArray();
    }

    /**
     * Returns a range of size endIndex.
     *
     * @param endIndex the exclusive upper bound
     * @return the integers in the half-open range {@code [0, endIndex)}
     */
    static int[] irange(int endIndex) {
        return irange(0, endIndex);
    }
}
