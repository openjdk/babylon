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
package hat.types;

import hat.DimMap;
import hat.Shape;
import hat.buffer.Tensor;
import jdk.incubator.code.CodeType;
import optkl.IfaceValue;

/**
 * Data type to represent views over tensors (tiles) within the compute kernels.
 *
 * <p>Provides arithmetic, reduction, shape-transformation and construction operations
 * for the {@link Tile} Programming Model implementation.</p>
 *
 * <p>Binary operations involving two tiles are performed element-wise.
 * Operations that involve a tile and a scalar values apply the scalar
 * to every element of the tile. Compatible tile shapes may be broadcasted
 * by the compiler (or HAT Transformer).</p>
 *
 * <p>All methods in this class are compiler intrinsics. Calls made from a HAT
 * Tile kernel are replaced by the HAT Tile Transformer and lowered to the
 * backend-specific operations.</p>
 *
 * @apiNote These methods have no ordinary Java implementation and throws an
 * {@link UnsupportedOperationException} exception. Support for a Java CPU
 * execution is planned for future versions of the Tile programming model
 * implementation within HAT.
 */
public record Tile() implements IfaceValue {

    /**
     * Add two tiles.
     * @param aTile tile A
     * @param bTile tile B
     *
     * @return {@link Tile}
     */
    public static Tile add(Tile aTile, Tile bTile) {
        throw new UnsupportedOperationException("Tile.add is a compiler intrinsic.");
    }

    /**
     * Add a tile with a scalar value.
     *
     * @param aTile tile
     * @param val scalar value
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile add(Tile aTile, T val) {
        throw new UnsupportedOperationException("Tile.add is a compiler intrinsic.");
    }

    /**
     * Add a scalar value with a tile.
     *
     * @param val scalar value
     * @param tile tile
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile add(T val, Tile tile) {
        throw new UnsupportedOperationException("Tile.add is a compiler intrinsic.");
    }

    /**
     * Subtract two tiles
     *
     * @param aTile tile A
     * @param bTile tile B
     * @return {@link Tile}
     */
    public static Tile sub(Tile aTile, Tile bTile) {
        throw new UnsupportedOperationException("Tile.sub is a compiler intrinsic.");
    }

    /**
     * Subtract a scalar value from a tile
     *
     * @param tile tile
     * @param val scalar value
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile sub(Tile tile, T val) {
        throw new UnsupportedOperationException("Tile.sub is a compiler intrinsic.");
    }

    /**
     * Subtract a tile from a scalar value. The value is broadcasted to the tile size.
     *
     * @param val value
     * @param tile tile
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile sub(T val, Tile tile) {
        throw new UnsupportedOperationException("Tile.sub is a compiler intrinsic.");
    }

    /**
     * Multiply two tiles
     *
     * @param aTile tile A
     * @param bTile tile B
     * @return {@link Tile}
     */
    public static Tile mul(Tile aTile, Tile bTile) {
        throw new UnsupportedOperationException("Tile.mul is a compiler intrinsic.");
    }

    /**
     * Multiply a tile by a scalar value
     *
     * @param tile tile
     * @param val scalar value
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile mul(Tile tile, T val) {
        throw new UnsupportedOperationException("Tile.mul is a compiler intrinsic.");
    }

    /**
     * Multiply a value by a tile.
     *
     * @param tile tile
     * @param val scalar value
     * @return {@link Tile}
     * @param <T extends Number>
     */
    public static <T extends Number> Tile mul(T val, Tile tile) {
        throw new UnsupportedOperationException("Tile.mul is a compiler intrinsic.");
    }

    /**
     * true division aTile / bTile
     *
     * @param aTile
     * @param bTile
     * @return
     */
    public static Tile truediv(Tile aTile, Tile bTile) {
        throw new UnsupportedOperationException("Tile.truediv is a compiler intrinsic.");
    }

    public static <T extends Number> Tile truediv(Tile aTile, T val) {
        throw new UnsupportedOperationException("Tile.truediv is a compiler intrinsic.");
    }

    public static <T extends Number> Tile truediv(T val, Tile aTile) {
        throw new UnsupportedOperationException("Tile.truediv is a compiler intrinsic.");
    }

    /**
     * Ceil division (aTile / bTile)
     *
     * @param aTile
     * @param bTile
     * @return
     */
    public static Tile ceildiv(Tile aTile, Tile bTile) {
        throw new UnsupportedOperationException("Tile.ceildiv is a compiler intrinsic.");
    }

    public static <T extends Number> Tile ceildiv(Tile aTile, T val) {
        throw new UnsupportedOperationException("Tile.ceildiv is a compiler intrinsic.");
    }

    public static <T extends Number> Tile ceildiv(T val, Tile aTile) {
        throw new UnsupportedOperationException("Tile.ceildiv is a compiler intrinsic.");
    }

    public static int ceildiv(int a, int b) {
        throw new UnsupportedOperationException("Tile.ceildiv is a compiler intrinsic.");
    }

    /**
     * Transpose the two axes of the input tile. If the input tile is 2-dimensional,
     * then the axes are transposed. If the input tile is n-dimensional (n > 2),
     * then this operation only transposes the first two axes.
     *
     * @param tile
     * @return {@link Tile}
     */
    public static Tile transpose(Tile tile) {
        throw new UnsupportedOperationException("Tile.transpose is a compiler intrinsic.");
    }

    /**
     * Reshape an input tile with another shape.
     *
     * @param aTile
     * @param aShape
     *
     * @return {@link Tile}
     */
    public static Tile reshape(Tile aTile, Shape aShape) {
        throw new UnsupportedOperationException("Tile.reshape is a compiler intrinsic.");
    }

    public static Tile sum(Tile aTile, int axis) {
        throw new UnsupportedOperationException("Tile.sum is a compiler intrinsic.");
    }

    public static Tile min(Tile aTile, int axis) {
        throw new UnsupportedOperationException("Tile.min is a compiler intrinsic.");
    }

    public static int min(final int a, final int b) {
        throw new UnsupportedOperationException("Tile.min is a compiler intrinsic.");
    }

    public static Tile max(Tile tile, int axis) {
        throw new UnsupportedOperationException("Tile.max is a compiler intrinsic.");
    }

    /**
     * Computes the number of tiles for an input tensor using the axis (second parameter)
     * with the tile size.
     * @param input Inpute tensor from global memory.
     * @param axis dimension to compute
     * @param tileSize input tile size. This is equivalent to shape(value).
     * @return {@link Tile}
     */
    public static int numTiles(Tensor input, int axis, int tileSize) {
        throw new UnsupportedOperationException("Tile.numTiles is a compiler intrinsic.");
    }

    /**
     * Computes the number of tiles for an input tensor using the axis (second parameter)
     * with the specified shape.
     * @param input Inpute tensor from global memory.
     * @param axis dimension to compute
     * @param shape a sequence of int values to specify the shape of the tile
     * @return {@link Tile}
     */
    public static int numTiles(Tensor input, int axis, Shape shape) {
        throw new UnsupportedOperationException("Tile.numTiles is a compiler intrinsic.");
    }

    /**
     * Creates and initializes a tile with a specific shape to a given float (F32) value.
     * @param shape input shape
     * @param value init value
     * @return {@link Tile}
     */
    public static Tile full(Shape shape, float value) {
        throw new UnsupportedOperationException("Tile.full is a compiler intrinsic.");
    }

    /**
     * Perform a Matrix-Multiple-Accumulate operation (a * b + c)
     * @param tileA tile A
     * @param tileB tile B
     * @param accumulator accumulator
     * @return {@link Tile}
     */
    public static Tile mma(Tile tileA, Tile tileB, Tile accumulator) {
        throw new UnsupportedOperationException("Tile.mma is a compiler intrinsic.");
    }

    /**
     * It creates a new tile and set the initial value to zeros.
     * By default, the tensor type is {@link hat.DType#TENSOR_F32_TYPE} for 1D and
     * {@link hat.DType#TENSOR_2D_F32_TYPE} for a 2D tile.
     *
     * @param shape
     * @return {@link Tile}
     */
    public static Tile zeros(Shape shape) {
        throw new UnsupportedOperationException("Tile.zeros is a compiler intrinsic.");
    }

    /**
     * Creates a 1D-tile with sequence elements starting with zero.
     * <p>
     * <code>
     * var tile = Tile.arange(4); // [0, 1, 2, 3]
     * </code>
     * </p>
     * <p>
     * DType for this expression is Integer (JavaType.INT)
     *
     * @param size
     * @return {@link Tile}
     */
    public static Tile arange(final int size) {
        throw new UnsupportedOperationException("Tile.arange is a compiler intrinsic.");
    }

    /**
     * Creates a 1D-tile with sequence elements starting with zero with a specific dType.
     * <p>
     * <code>
     * var tile = Tile.arange(4, DType.FLOAT); // [0.0f, 1.0f, 2.0f, 3.0f]
     * </code>
     * </p>
     * <p>
     *
     * @param size Tile size
     * @param dType dType
     * @return {@link Tile}
     */
    public static Tile arange(final int size, CodeType dType) {
        throw new UnsupportedOperationException("Tile.arange is a compiler intrinsic.");
    }

    /**
     * Creates a 1D-tile with sequence of elements starting with "start",
     * in increments of 1 (step set to one).
     * <p>
     * <code>
     * var tile = Tile.arange(4, DType.INT, 2); // [2, 3, 4, 5]
     * </code>
     * </p>
     *
     * @param size tile size
     * @param dType Type of the Tensor. Use the class {@link hat.DType} to obtain the type
     * @param start start value
     * @return {@link Tile}
     */
    public static Tile arange(final int size, CodeType dType, final int start) {
        throw new UnsupportedOperationException("Tile.arange is a compiler intrinsic.");
    }


    /**
     * Creates a 1D-tile with sequence of elements starting with "start",
     * in increments of "step".
     * <p>
     * <code>
     * var tile = Tile.arange(4, DType.INT, 2, 2); // [2, 4, 6, 8]
     * </code>
     * </p>
     *
     * @param size tile size
     * @param dType Type of the Tensor. Use the class {@link hat.DType} to obtain the type
     * @param start start value
     * @param step step value
     * @return {@link Tile}
     */
    public static Tile arange(final int size, CodeType dType, final int start, final int step) {
        throw new UnsupportedOperationException("Tile.arange is a compiler intrinsic.");
    }

    /**
     * Arbitrary reordering of the axis of the input tensor according to the {@link DimMap} object.
     *
     * @param tile
     * @param dimMap
     *
     * {@snippet :
     *  // if inputTile is of shape:
     *  // [0, 1, 2, 3,
     *  // 4, 5, 6, 7]
     *  // Then, after the following permute:
     *  // 0, 4,
     *  // 1, 5,
     *  // 2, 6,
     *  // 3, 7
     *  var tile = Tile.permute(inputTile, TileContext.dimension(2, 0, 1));
     * }
     *
     * @return {@link Tile}
     */
    public static Tile permute(Tile tile, DimMap dimMap) {
        throw new UnsupportedOperationException("Tile.permute is a compiler intrinsic.");
    }
}
