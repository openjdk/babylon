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

import hat.buffer.F16Array;
import hat.buffer.F32Array;
import hat.buffer.F32ArrayPadded;
import optkl.IfaceValue;

/**
 * Describes a logical tensor fragment used by a HAT tensor-aware backend.
 *
 * <p>A {@code Tensor} is a small, typed view of an input matrix. The API is intentionally backend-neutral:
 * CUDA backends may lower these Tensor operations to NVIDIA WMMA tensor-core primitives, while other backends,
 * such as OpenCL, may lower them to ordinary memory and arithmetic operations without exposing hardware
 * tensor cores directly.
 * </p>
 *
 * <p>The Java methods in the Tensor class primarily describe the intent to the HAT compiler pipeline. Thus,
 * some method have little or non-Java implementation. The functionality is supplied by the HAT backend.</p>
 *
 * @param shape Represents the logic tensor shape, expressed as {@code m}, {@code n}, {@code k} dimensions.
 * @param klass Java {@code Class} of each element of the tensor.
 * @param tensorAccess Memory layout used when lading from or storing to backing arrays, or {@code null} when the
 *                     backend default layout is used.
 */
public record Tensor(Shape shape, Class<?> klass, Access tensorAccess) implements IfaceValue {

    /**
     * Describes the logical dimension of a tensor
     *
     * @param x first tensor dimension
     * @param y second tensor dimension
     * @param z third tensor dimension
     */
    public record Shape(int x, int y, int z) {
    }

    /**
     * Marker interface to define the memory layout accessors.
     */
    public interface Access { }

    /**
     * Column-major access layout.
     */
    public record ColumMajor() implements Access {
    }

    /**
     * Row-major access layout
     */
    public record RowMajor() implements Access {
    }


    /**
     * Column-major access layout.
     */
    public static ColumMajor ofColumnMajor() {
        return new ColumMajor();
    }

    /**
     * Row-major access layout
     */
    public static RowMajor ofRowMajor() {
        return new RowMajor();
    }

    /**
     * Creates a tensor shape descriptor. The three dimensions are interpreted by the backend according to the tensor
     * operation being lowered. For Matrix-Multiply-Accumulate (MMA) correspond to the {@code m}, {@code n}, and {@code k}
     * dimensions of the tile.
     *
     * @param dim1 first dimension
     * @param dim2 second dimension
     * @param dim3 third dimension
     * @return a tensor shape descriptor
     */
    public static Shape shape(int dim1, int dim2, int dim3) {
        return new Shape(dim1, dim2, dim3);
    }


    /**
     * Creates a Tensor with a specific shape and class. Currently, the only class supported is {@code F16.class}
     * and {@code float} In the future, new types may appear such as {@code FP8} and/or {@code double}. Tensors
     * created using this method are meant to be used as accumulators. To create a tensor from a view of input data,
     * the method {@link Tensor#loadF16} must be used instead.
     *
     * @param shape the shape descriptor
     * @param klass the tensor type (e.g., {@code float.class}
     * @return a tensor with the backend default access layout
     */
    public static Tensor create(Shape shape, Class<?> klass) {
        return new Tensor(shape, klass, null);
    }

    /**
     * Creates a Tensor whose content is initialized to zero by the HAT Backend. This operation is intended
     * to be used for tensor accumulators, similar to the {@link Tensor#create} method.
     *
     * @param shape the shape descriptor.
     * @param klass the tensor type (e.g., {@code float.class}.
     * @return a zero-initialized Tensor.
     */
    public static Tensor zeros(Shape shape, Class<?> klass) {
        return new Tensor(shape, klass, null);
    }

    /**
     * Fills a tensor with a specific value.
     *
     * @param tensor tensor to fill.
     * @param value scalar value to be assigned to each element of the tensor.
     */
    public static void fill(Tensor tensor, float value) {
    }

    /**
     * Performs Matrix-Multiply-Accumulate (MMA) operation on the Tensors.
     *
     * <p>The operation represents {@code result = tensorA * tensorB + acc}. HAT CUDA backends
     * may lower this oeration to an explicit mma operation, while a non-WMMA backend may implement
     * the equivalent functionality using loop-tiling.
     * </p>
     *
     * @param tensorA tensor that represents the first matrix.
     * @param tensorB tensor that represents the second matrix.
     * @param acc tensor accumulator
     * @return result result of the MMA operation.
     */
    public static Tensor mma(Tensor tensorA, Tensor tensorB, Tensor acc) {
        // This is used as a marker for the HAT Backend. When supporting the CPU,
        // we will need to insert the content as well acc = add(dot(tensorA, tensorB), acc);
        return new Tensor(acc.shape(), acc.klass, acc.tensorAccess);
    }

    /**
     * Loads a tensor of type {@code F16} from an input matrix using the default row-major access layout.
     *
     * <p>The indices {@code rowIndex}, and {@code j} identify the starting matrix location for the tensor,
     * and the {@code ldd} represents the leading dimension used by the backend when loading the data
     * from the input matrix.
     * </p>
     *
     * @param matrix input matrix
     * @param rowIndex row index
     * @param columnIndex column index
     * @param ldd leading dimension
     * @param shape shape descriptor
     * @return a tensor in {@link F16} type using the default memory access layout (row major).
     */
    public static Tensor loadF16(F16Array matrix, int rowIndex, int columnIndex, int ldd, Shape shape) {
        // select Row Major as default
        return new Tensor(shape, F16.class, ofRowMajor());
    }

    /**
     * Loads a tensor of type {@code F16} from an input matrix using the specific memory layout accessor.
     *
     * <p>The indices {@code rowIndex}, and {@code j} identify the starting matrix location for the tensor,
     * and the {@code ldd} represents the leading dimension used by the backend when loading the data
     * from the input matrix.
     * </p>
     *
     * @param matrix input matrix
     * @param rowIndex row index
     * @param columnIndex column index
     * @param ldd leading dimension
     * @param shape shape descriptor
     * @param tensorAccess memory layout accessor for loading the data from the input matrix into the tensor
     * @return a tensor in {@link F16}.
     */
    public static Tensor loadF16(F16Array matrix, int rowIndex, int columnIndex, int ldd, Shape shape, final Access tensorAccess) {
        return new Tensor(shape, F16.class, tensorAccess);
    }

    /**
     * Stores a tensor into an output matrix in single-precision ({@link F32Array}) using the specific memory layout.
     *
     * @param matrix destination matrix
     * @param rowIndex row index from the output matrix
     * @param columnIndex column index from the output matrix
     * @param resultTensor result tensor to store
     * @param ldd leading dimension
     * @param tensorAccess memory layout accessor
     */
    public static void store(F32Array matrix, int rowIndex, int columnIndex, Tensor resultTensor, int ldd, Access tensorAccess) {
    }

    /**
     * Stores a tensor into an output matrix in single-precision ({@link F32Array}) using the backend default accessor (row-major).
     *
     * @param matrix destination matrix
     * @param rowIndex row index from the output matrix
     * @param columnIndex column index from the output matrix
     * @param resultTensor resulting tensor to store
     * @param ldd leading dimension
     */
    public static void store(F32Array matrix, int rowIndex, int columnIndex, Tensor resultTensor, int ldd) {
    }

    /**
     * Stores a tensor into an output matrix in single-precision with padding ({@link F32ArrayPadded}) using the backend
     * default accessor (row-major).
     *
     * @param matrix destination matrix
     * @param rowIndex row index from the output matrix
     * @param columnIndex column index from the output matrix
     * @param resultTensor resulting tensor to store
     * @param ldd leading dimension
     */
    public static void store(F32ArrayPadded matrix, int rowIndex, int columnIndex, Tensor resultTensor, int ldd) {
    }


    /**
     * Stores a tensor into an output matrix in single-precision with padding ({@link F32ArrayPadded}) using the specific memory layout.
     *
     * @param matrix destination matrix
     * @param rowIndex row index from the output matrix
     * @param columnIndex column index from the output matrix
     * @param resultTensor resulting tensor to store
     * @param ldd leading dimension
     */
    public static void store(F32ArrayPadded matrix, int rowIndex, int columnIndex, Tensor resultTensor, int ldd, Access tensorAccess) {
    }
}
