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

package oracle.code.onnx.ir;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.type.TypeElementFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public abstract sealed class OnnxType implements TypeElement {

    public static final TypeElementFactory FACTORY = new TypeElementFactory() {
        @Override
        public OnnxType constructType(ExternalizedTypeElement tree) {
            switch (tree.identifier()) {
                case TypeVariable.NAME: {
                    if (tree.arguments().size() < 2) {
                        throw new IllegalArgumentException();
                    }

                    ExternalizedTypeElement typeVariable = tree.arguments().getFirst();
                    if (!typeVariable.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }
                    List<OnnxType> types = new ArrayList<>();
                    for (int i = 1; i < tree.arguments().size(); i++) {
                        types.add(constructType(tree.arguments().get(i)));
                    }
                    return new TypeVariable(typeVariable.identifier(), types);
                }
                case OptionalType.NAME: {
                    if (tree.arguments().size() != 1) {
                        throw new IllegalArgumentException();
                    }

                    return new OptionalType(constructType(tree.arguments().getFirst()));
                }
                case "seq":
                case SequenceType.NAME: {
                    if (tree.arguments().size() != 1) {
                        throw new IllegalArgumentException();
                    }

                    return new SequenceType(constructType(tree.arguments().getFirst()));
                }
                case MapType.NAME: {
                    if (tree.arguments().size() != 2) {
                        throw new IllegalArgumentException();
                    }

                    return new MapType(constructType(
                            tree.arguments().get(0)), constructType(tree.arguments().get(1)));
                }
                case TensorType.NAME: {
                    if (tree.arguments().size() != 1) {
                        throw new IllegalArgumentException();
                    }

                    // @@@ Shape encoding
                    return new TensorType(
                            (OnnxElementType) constructType(tree.arguments().getFirst()),
                            List.of());
                }
                case Float16Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float16Type();
                }
                case "float":
                case Float32Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float32Type();
                }
                case "double":
                case Float64Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float64Type();
                }
                case BFloat16Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new BFloat16Type();
                }
                case Float8e4m3fnType.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float8e4m3fnType();
                }
                case Float8e5m2Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float8e5m2Type();
                }
                case Float8e4m3fnuzType.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float8e4m3fnuzType();
                }
                case Float8e5m2fnuzType.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float8e5m2fnuzType();
                }
                case Float4e2m1Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Float4e2m1Type();
                }
                case Int4Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Int4Type();
                }
                case Int8Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Int8Type();
                }
                case Int16Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Int16Type();
                }
                case Int32Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Int32Type();
                }
                case Int64Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Int64Type();
                }
                case UInt4Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new UInt4Type();
                }
                case UInt8Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new UInt8Type();
                }
                case UInt16Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new UInt16Type();
                }
                case UInt32Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new UInt32Type();
                }
                case UInt64Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new UInt64Type();
                }
                case Complex64Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Complex64Type();
                }
                case Complex128Type.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new Complex128Type();
                }
                case BoolType.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new BoolType();
                }
                case StringType.NAME: {
                    if (!tree.arguments().isEmpty()) {
                        throw new IllegalArgumentException();
                    }

                    return new StringType();
                }
            }
            return null;
        }
    };


    public static final class TypeVariable extends OnnxType {
        static final String NAME = "variable";

        final String name;
        final List<OnnxType> types;

        public TypeVariable(String name, List<OnnxType> types) {
            this.name = name;
            this.types = List.copyOf(types);
        }

        public String name() {
            return name;
        }

        public List<OnnxType> types() {
            return types;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof TypeVariable that)) return false;
            return Objects.equals(name, that.name) && Objects.equals(types, that.types);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, types);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            List<ExternalizedTypeElement> children = new ArrayList<>();
            children.add(new ExternalizedTypeElement(name, List.of()));
            for (OnnxType type : types) {
                children.add(type.externalize());
            }
            return new ExternalizedTypeElement(NAME, children);
        }
    }


    public static final class OptionalType extends OnnxType {
        static final String NAME = "optional";

        final OnnxType eType;

        public OptionalType(OnnxType eType) {
            this.eType = eType;
        }

        public OnnxType eType() {
            return eType;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            OptionalType that = (OptionalType) o;
            return Objects.equals(eType, that.eType);
        }

        @Override
        public int hashCode() {
            return Objects.hash(eType);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(eType.externalize()));
        }
    }

    public static final class SequenceType extends OnnxType {
        static final String NAME = "sequence";

        final OnnxType eType;

        public SequenceType(OnnxType eType) {
            this.eType = eType;
        }

        public OnnxType eType() {
            return eType;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            SequenceType that = (SequenceType) o;
            return Objects.equals(eType, that.eType);
        }

        @Override
        public int hashCode() {
            return Objects.hash(eType);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(eType.externalize()));
        }
    }

    public static final class MapType extends OnnxType {
        static final String NAME = "map";

        final OnnxType keyType;
        final OnnxType valueType;

        public MapType(OnnxType keyType, OnnxType valueType) {
            this.keyType = keyType;
            this.valueType = valueType;
        }

        public OnnxType keyType() {
            return keyType;
        }

        public OnnxType valueType() {
            return valueType;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            MapType that = (MapType) o;
            return Objects.equals(keyType, that.keyType) && Objects.equals(valueType, that.valueType);
        }

        @Override
        public int hashCode() {
            return Objects.hash(keyType, valueType);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of(keyType.externalize(), valueType.externalize()));
        }
    }

    public static final class TensorType extends OnnxType {
        static final String NAME = "tensor";

        final OnnxElementType eType;
        // A tensor can be defined as a pair of sequences/lists (V, S) where S is the shape of the tensor
        // (a list of non-negative integers) and V is a list of values with length equal to the product
        // of the dimensions in S
        // If S has length 0, V must have length 1, since the empty product is defined to be 1.
        // In this case, the tensor represents a scalar.
        // S can contain dimensions of value 0. If any dimensions are 0, V must have length 0.
        // If S has length 1, V has length equal to the single dimension in S.
        // In this case, the tensor represents a vector.
        // A tensor representing a vector of length 1 has shape [1], while a tensor representing
        // a scalar has shape []. They both have a single element, but scalars are not vectors of length 1.
        //
        // Inputs and outputs of a model (top-level graph) are required to have a shape, indicating
        // the rank of inputs and outputs, even though the exact dimensions need not be specified.
        //
        // null value indicates any shape
        // empty list indicates scalar
        // Each list element is either an integer representing the size of the dimension
        // or a string representing a dimension variable e.g. [100, 100] or [N,M]
        final List<Object> shape;

        public TensorType(OnnxElementType eType) {
            this(eType, null);
        }

        public TensorType(OnnxElementType eType, List<Object> shape) {
            this.eType = eType;
            this.shape = shape != null ? List.copyOf(shape) : null;
        }

        public OnnxElementType eType() {
            return eType;
        }

        public List<Object> shape() {
            return shape;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TensorType that = (TensorType) o;
            return Objects.equals(eType, that.eType) && Objects.equals(shape, that.shape);
        }

        @Override
        public int hashCode() {
            return Objects.hash(eType, shape);
        }

        @Override
        public ExternalizedTypeElement externalize() {
            List<ExternalizedTypeElement> args = new ArrayList<>();
            if (shape != null) {
                for (Object i : shape) {
                    args.add(new ExternalizedTypeElement("x" + i, List.of()));
                }
            }
            args.add(eType.externalize());
            return new ExternalizedTypeElement(NAME, args);
        }
    }


    public static abstract sealed class OnnxElementType extends OnnxType {
        public abstract int id();
    }

    public static final class Float16Type extends OnnxElementType {
        static final String NAME = "float16";

        Float16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 10;
        }
    }

    public static final class Float32Type extends OnnxElementType {
        // float32
        static final String NAME = "float32";

        Float32Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 1;
        }
    }

    public static final class Float64Type extends OnnxElementType {
        // float64
        static final String NAME = "float64";

        Float64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 11;
        }
    }

    public static final class BFloat16Type extends OnnxElementType {
        static final String NAME = "bfloat16";

        BFloat16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 16;
        }
    }

    public static final class Float8e4m3fnType extends OnnxElementType {
        static final String NAME = "float8e4m3fn";

        Float8e4m3fnType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 17;
        }
    }

    public static final class Float8e5m2Type extends OnnxElementType {
        static final String NAME = "float8e5m2";

        Float8e5m2Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 19;
        }
    }

    public static final class Float8e4m3fnuzType extends OnnxElementType {
        static final String NAME = "float8e4m3fnuz";

        Float8e4m3fnuzType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 18;
        }
    }

    public static final class Float8e5m2fnuzType extends OnnxElementType {
        static final String NAME = "float8e5m2fnuz";

        Float8e5m2fnuzType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 20;
        }
    }

    public static final class Float4e2m1Type extends OnnxElementType {
        static final String NAME = "float4e2m1";

        Float4e2m1Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 23;
        }
    }

    public static final class Int4Type extends OnnxElementType {
        static final String NAME = "int4";

        Int4Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 22;
        }
    }

    public static final class Int8Type extends OnnxElementType {
        static final String NAME = "int8";

        Int8Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 3;
        }
    }

    public static final class Int16Type extends OnnxElementType {
        static final String NAME = "int16";

        Int16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 5;
        }
    }

    public static final class Int32Type extends OnnxElementType {
        static final String NAME = "int32";

        Int32Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 6;
        }
    }

    public static final class Int64Type extends OnnxElementType {
        static final String NAME = "int64";

        Int64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 7;
        }
    }

    public static final class UInt4Type extends OnnxElementType {
        static final String NAME = "uint4";

        UInt4Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 21;
        }
    }

    public static final class UInt8Type extends OnnxElementType {
        static final String NAME = "uint8";

        UInt8Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 2;
        }
    }

    public static final class UInt16Type extends OnnxElementType {
        static final String NAME = "uint16";

        UInt16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 4;
        }
    }

    public static final class UInt32Type extends OnnxElementType {
        static final String NAME = "uint32";

        UInt32Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 12;
        }
    }

    public static final class UInt64Type extends OnnxElementType {
        static final String NAME = "uint64";

        UInt64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 13;
        }
    }

    public static final class Complex64Type extends OnnxElementType {
        static final String NAME = "complex64";

        Complex64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 14;
        }
    }

    public static final class Complex128Type extends OnnxElementType {
        static final String NAME = "complex128";

        Complex128Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 15;
        }
    }

    public static final class BoolType extends OnnxElementType {
        static final String NAME = "bool";

        BoolType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 9;
        }
    }

    public static final class StringType extends OnnxElementType {
        static final String NAME = "string";

        StringType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }

        @Override
        public int id() {
            return 8;
        }
    }

    public static final Int4Type INT4 = new Int4Type();
    public static final Int8Type INT8 = new Int8Type();
    public static final Int16Type INT16 = new Int16Type();
    public static final Int32Type INT32 = new Int32Type();
    public static final Int64Type INT64 = new Int64Type();

    public static final UInt4Type UINT4 = new UInt4Type();
    public static final UInt8Type UINT8 = new UInt8Type();
    public static final UInt16Type UINT16 = new UInt16Type();
    public static final UInt32Type UINT32 = new UInt32Type();
    public static final UInt64Type UINT64 = new UInt64Type();

    public static final Float16Type FLOAT16 = new Float16Type();
    public static final Float32Type FLOAT32 = new Float32Type();
    public static final Float64Type FLOAT64 = new Float64Type();

    public static final BFloat16Type BFLOAT16 = new BFloat16Type();

    public static final Float4e2m1Type FLOAT4E2M1 = new Float4e2m1Type();
    public static final Float8e5m2Type FLOAT8E5M2 = new Float8e5m2Type();
    public static final Float8e4m3fnType FLOAT8E4M3FN = new Float8e4m3fnType();
    public static final Float8e4m3fnuzType FLOAT8E4M3FNUZ = new Float8e4m3fnuzType();
    public static final Float8e5m2fnuzType FLOAT8E5M2FNUZ = new Float8e5m2fnuzType();

    public static final Complex64Type COMPLEX64 = new Complex64Type();
    public static final Complex128Type COMPLEX128 = new Complex128Type();

    public static final StringType STRING = new StringType();
    public static final BoolType BOOL = new BoolType();

    public static final TensorType TENSOR_INT4 = new TensorType(INT4);
    public static final TensorType TENSOR_INT8 = new TensorType(INT8);
    public static final TensorType TENSOR_INT16 = new TensorType(INT16);
    public static final TensorType TENSOR_INT32 = new TensorType(INT32);
    public static final TensorType TENSOR_INT64 = new TensorType(INT64);

    public static final TensorType TENSOR_UINT4 = new TensorType(UINT4);
    public static final TensorType TENSOR_UINT8 = new TensorType(UINT8);
    public static final TensorType TENSOR_UINT16 = new TensorType(UINT16);
    public static final TensorType TENSOR_UINT32 = new TensorType(UINT32);
    public static final TensorType TENSOR_UINT64 = new TensorType(UINT64);

    public static final TensorType TENSOR_FLOAT16 = new TensorType(FLOAT16);
    public static final TensorType TENSOR_FLOAT32 = new TensorType(FLOAT32);
    public static final TensorType TENSOR_FLOAT64 = new TensorType(FLOAT64);

    public static final TensorType TENSOR_BFLOAT16 = new TensorType(BFLOAT16);

    public static final TensorType TENSOR_FLOAT4E2M1 = new TensorType(FLOAT4E2M1);
    public static final TensorType TENSOR_FLOAT8E5M2 = new TensorType(FLOAT8E5M2);
    public static final TensorType TENSOR_FLOAT8E4M3FN = new TensorType(FLOAT8E4M3FN);
    public static final TensorType TENSOR_FLOAT8E4M3FNUZ = new TensorType(FLOAT8E4M3FNUZ);
    public static final TensorType TENSOR_FLOAT8E5M2FNUZ = new TensorType(FLOAT8E5M2FNUZ);

    public static final TensorType TENSOR_COMPLEX64 = new TensorType(COMPLEX64);
    public static final TensorType TENSOR_COMPLEX128 = new TensorType(COMPLEX128);

    public static final TensorType TENSOR_STRING = new TensorType(STRING);
    public static final TensorType TENSOR_BOOL = new TensorType(BOOL);


    public static Int4Type int4() { return INT4; }
    public static Int8Type int8() { return INT8; }
    public static Int16Type int16() { return INT16; }
    public static Int32Type int32() { return INT32; }
    public static Int64Type int64() { return INT64; }

    public static Float16Type float16() { return FLOAT16; }
    public static Float32Type float32() { return FLOAT32; }
    public static Float64Type float64() { return FLOAT64; }

    public static UInt4Type uint4() { return UINT4; }
    public static UInt8Type uint8() { return UINT8; }
    public static UInt16Type uint16() { return UINT16; }
    public static UInt32Type uint32() { return UINT32; }
    public static UInt64Type uint64() { return UINT64; }

    public static Complex64Type complex64() { return COMPLEX64; }
    public static Complex128Type complex128() { return COMPLEX128; }

    public static BFloat16Type bfloat16() { return BFLOAT16; }

    public static Float4e2m1Type float4e2m1() { return FLOAT4E2M1; }
    public static Float8e5m2Type float8e5m2() { return FLOAT8E5M2; }
    public static Float8e4m3fnType float8e4m3fn() { return FLOAT8E4M3FN; }
    public static Float8e4m3fnuzType float8e4m3fnuz() { return FLOAT8E4M3FNUZ; }
    public static Float8e5m2fnuzType float8e5m2fnuz() { return FLOAT8E5M2FNUZ; }

    public static StringType string() { return STRING; }
    public static BoolType bool() { return BOOL; }

    public static TensorType tensor(OnnxElementType e) {
        TensorType tt = switch (e) {
            case Int4Type t -> OnnxType.TENSOR_INT4;
            case Int8Type t -> OnnxType.TENSOR_INT8;
            case Int16Type t -> OnnxType.TENSOR_INT16;
            case Int32Type t -> OnnxType.TENSOR_INT32;
            case Int64Type t -> OnnxType.TENSOR_INT64;

            case UInt4Type t -> OnnxType.TENSOR_UINT4;
            case UInt8Type t -> OnnxType.TENSOR_UINT8;
            case UInt16Type t -> OnnxType.TENSOR_UINT16;
            case UInt32Type t -> OnnxType.TENSOR_UINT32;
            case UInt64Type t -> OnnxType.TENSOR_UINT64;

            case Float16Type t -> OnnxType.TENSOR_FLOAT16;
            case Float32Type t -> OnnxType.TENSOR_FLOAT32;
            case Float64Type t -> OnnxType.TENSOR_FLOAT64;

            case BFloat16Type t -> OnnxType.TENSOR_BFLOAT16;

            case Float4e2m1Type t -> OnnxType.TENSOR_FLOAT4E2M1;
            case Float8e5m2Type t -> OnnxType.TENSOR_FLOAT8E5M2;
            case Float8e4m3fnType t -> OnnxType.TENSOR_FLOAT8E4M3FN;
            case Float8e4m3fnuzType t -> OnnxType.TENSOR_FLOAT8E4M3FNUZ;
            case Float8e5m2fnuzType float8e5m2fnuzType -> OnnxType.TENSOR_FLOAT8E5M2FNUZ;

            case Complex64Type t -> OnnxType.TENSOR_COMPLEX64;
            case Complex128Type t -> OnnxType.TENSOR_COMPLEX128;

            case StringType t -> OnnxType.TENSOR_STRING;
            case BoolType t -> OnnxType.TENSOR_BOOL;
        };

        assert tt.eType.equals(e);
        return tt;
    }

    public static OptionalType optional(OnnxType e) {
        return new OptionalType(e);
    }

    public static SequenceType seq(OnnxType e) {
        return new SequenceType(e);
    }

    public static MapType map(OnnxType k, OnnxType v) {
        return new MapType(k, v);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}