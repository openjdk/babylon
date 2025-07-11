/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package oracle.code.onnx;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

    /*
class DataType(enum.IntEnum):
    """Enum for the data types of ONNX tensors, defined in ``onnx.TensorProto``."""

    # NOTE: Naming: It is tempting to use shorter and more modern names like f32, i64,
    # but we should stick to the names used in the ONNX spec for consistency.
    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22
    FLOAT4E2M1 = 23
     */

public class Tensor<T> extends OnnxNumber {

    public static final long[] SCALAR_SHAPE = new long[0];

    public static Tensor<Boolean> ofScalar(boolean b) {
        return ofScalar(Arena.ofAuto(), b);
    }

    public static Tensor<Boolean> ofScalar(Arena arena, boolean b) {
        return new Tensor(arena, arena.allocateFrom(ValueLayout.JAVA_BYTE, b ? (byte)1 : 0), ElementType.BOOL, SCALAR_SHAPE);
    }

    public static Tensor<Byte> ofScalar(byte b) {
        return ofShape(SCALAR_SHAPE, b);
    }

    public static Tensor<Byte> ofScalar(Arena arena, byte b) {
        return ofShape(arena, SCALAR_SHAPE, b);
    }

    public static Tensor<Long> ofScalar(long l) {
        return ofShape(SCALAR_SHAPE, l);
    }

    public static Tensor<Long> ofScalar(Arena arena, long l) {
        return ofShape(arena, SCALAR_SHAPE, l);
    }

    public static Tensor<Float> ofScalar(float f) {
        return ofShape(SCALAR_SHAPE, f);
    }

    public static Tensor<Float> ofScalar(Arena arena, float f) {
        return ofShape(arena, SCALAR_SHAPE, f);
    }


    public static Tensor<Byte> ofFlat(byte... values) {
        return ofShape(new long[]{values.length}, values);
    }

    public static Tensor<Byte> ofFlat(Arena arena, byte... values) {
        return ofShape(arena, new long[]{values.length}, values);
    }

    public static Tensor<Long> ofFlat(long... values) {
        return ofShape(new long[]{values.length}, values);
    }

    public static Tensor<Long> ofFlat(Arena arena, long... values) {
        return ofShape(arena, new long[]{values.length}, values);
    }

    public static Tensor<Float> ofFlat(float... values) {
        return ofShape(new long[]{values.length}, values);
    }

    public static Tensor<Float> ofFlat(Arena arena, float... values) {
        return ofShape(arena, new long[]{values.length}, values);
    }

    public static Tensor<Byte> ofShape(long[] shape, byte... values) {
        return ofShape(Arena.ofAuto(), shape, values);
    }

    public static Tensor<Byte> ofShape(Arena arena, long[] shape, byte... values) {
        return new Tensor(arena, arena.allocateFrom(ValueLayout.JAVA_BYTE, values), ElementType.UINT8, shape);
    }

    public static Tensor<Long> ofShape(long[] shape, long... values) {
        return ofShape(Arena.ofAuto(), shape, values);
    }

    public static Tensor<Long> ofShape(Arena arena, long[] shape, long... values) {
        return new Tensor(arena, arena.allocateFrom(ValueLayout.JAVA_LONG, values), ElementType.INT64, shape);
    }

    public static Tensor<Float> ofShape(long[] shape, float... values) {
        return ofShape(Arena.ofAuto(), shape, values);
    }

    public static Tensor<Float> ofShape(Arena arena, long[] shape, float... values) {
        return new Tensor(arena, arena.allocateFrom(ValueLayout.JAVA_FLOAT, values), ElementType.FLOAT, shape);
    }

    public static <T> Tensor<T> ofShape(long[] shape, byte[] rawData, ElementType elementType) {
        return ofShape(Arena.ofAuto(), shape, rawData, elementType);
    }

    public static <T> Tensor<T> ofShape(Arena arena, long[] shape, byte[] rawData, ElementType elementType) {
        return new Tensor(arena, arena.allocateFrom(ValueLayout.JAVA_BYTE, rawData), elementType, shape);
    }

    // Mandatory reference to dataAddr to avoid its garbage colletion
    private final MemorySegment dataAddr;
    final MemorySegment tensorAddr;

    public Tensor(Arena arena, MemorySegment dataAddr, ElementType type, long[] shape) {
        this(dataAddr, OnnxRuntime.getInstance().createTensor(arena, dataAddr, type, shape));
    }

    Tensor(MemorySegment dataAddr, MemorySegment tensorAddr) {
        this.dataAddr = dataAddr;
        this.tensorAddr = tensorAddr;
    }

    public ElementType elementType() {
        return OnnxRuntime.getInstance().tensorElementType(tensorAddr);
    }

    public long[] shape() {
        return OnnxRuntime.getInstance().tensorShape(tensorAddr);
    }

    public MemorySegment data() {
        return dataAddr;
    }

    public enum ElementType {
        FLOAT(1, float.class),
        UINT8(2, byte.class),
        INT8(3, byte.class),
        UINT16(4, short.class),
        INT16(5, short.class),
        INT32(6, int.class),
        INT64(7, long.class),
        STRING(8, String.class),
        BOOL(9, boolean.class),
        FLOAT16(10, Object.class),
        DOUBLE(11, double.class),
        UINT32(12, int.class),
        UINT64(13, long.class),
        COMPLEX64(14, Object.class),
        COMPLEX128(15, Object.class),
        BFLOAT16(16, Object.class),
        FLOAT8E4M3FN(17, Object.class),
        FLOAT8E4M3FNUZ(18, Object.class),
        FLOAT8E5M2(19, Object.class),
        FLOAT8E5M2FNUZ(20, Object.class),
        UINT4(21, Object.class),
        INT4(22, Object.class),
        FLOAT4E2M1(23, Object.class);

        final int id;
        final Class<?> type;

        ElementType(int id, Class<?> type) {
            this.id = id;
            this.type = type;
        }

        public Class<?> type() {
            return type;
        }

        public String onnxName() {
            return name().toLowerCase();
        }

        public int bitSize() {
            return switch (this) {
                case INT4, UINT4, FLOAT4E2M1 -> 4;
                case UINT8, INT8, BOOL, FLOAT8E4M3FN, FLOAT8E4M3FNUZ, FLOAT8E5M2, FLOAT8E5M2FNUZ -> 8;
                case UINT16, INT16, FLOAT16, BFLOAT16 -> 16;
                case UINT32, INT32, FLOAT -> 32;
                case UINT64, INT64, DOUBLE, COMPLEX64 -> 64;
                case COMPLEX128 -> 128;
                case STRING -> -1;
            };
        }

        public static ElementType fromOnnxName(String name) {
            return ElementType.valueOf(name.toUpperCase());
        }

        public static ElementType fromOnnxId(int id) {
            return values()[id - 1];
        }
    }
}
