package oracle.code.onnx.ir;

import jdk.incubator.code.TypeElement;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public abstract sealed class OnnxType implements TypeElement {

    public static final class OptionalType extends OnnxType {
        static final String NAME = "optional";

        final TypeElement eType;

        public OptionalType(TypeElement eType) {
            this.eType = eType;
        }

        public TypeElement eType() {
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
        static final String NAME = "seq";

        final TypeElement eType;

        public SequenceType(TypeElement eType) {
            this.eType = eType;
        }

        public TypeElement eType() {
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

        final TypeElement keyType;
        final TypeElement valueType;

        public MapType(TypeElement keyType, TypeElement valueType) {
            this.keyType = keyType;
            this.valueType = valueType;
        }

        public TypeElement keyType() {
            return keyType;
        }

        public TypeElement valueType() {
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

        final TypeElement eType;
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

        // When size == 0, tensor represents a scalar whose contents hold one element
        final int size;

        public TensorType(TypeElement eType, List<Integer> shape) {
            this.eType = eType;
            this.shape = List.copyOf(shape);
            int s = 1;
            for (Integer i : shape) {
                s *= i;
            }
            this.size = s;
        }

        public TypeElement eType() {
            return eType;
        }

        public List<Object> shape() {
            return shape;
        }

        public int size() {
            return size;
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
            for (Object i : shape) {
                args.add(new ExternalizedTypeElement("x" + i, List.of()));
            }
            args.add(eType.externalize());
            return new ExternalizedTypeElement(NAME, args);
        }
    }


    public static abstract sealed class OnnxElementType extends OnnxType {
    }

    public static final class Float16Type extends OnnxElementType {
        static final String NAME = "float16";

        public Float16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class FloatType extends OnnxElementType {
        // float32
        static final String NAME = "float";

        public FloatType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class DoubleType extends OnnxElementType {
        // float64
        static final String NAME = "double";

        public DoubleType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class BFloat16Type extends OnnxElementType {
        static final String NAME = "bfloat16";

        public BFloat16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Float8e4m3fnType extends OnnxElementType {
        static final String NAME = "float8e4m3fn";

        public Float8e4m3fnType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Float8e5m2Type extends OnnxElementType {
        static final String NAME = "float8e5m2";

        public Float8e5m2Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Float8e4m3fnuzType extends OnnxElementType {
        static final String NAME = "float8e4m3fnuz";

        public Float8e4m3fnuzType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Float8e5m2fnuzType extends OnnxElementType {
        static final String NAME = "float8e5m2fnuz";

        public Float8e5m2fnuzType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Float4e2m1Type extends OnnxElementType {
        static final String NAME = "float4e2m1";

        public Float4e2m1Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Int4Type extends OnnxElementType {
        static final String NAME = "int4";

        public Int4Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Int8Type extends OnnxElementType {
        static final String NAME = "int8";

        public Int8Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Int16Type extends OnnxElementType {
        static final String NAME = "int16";

        public Int16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Int32Type extends OnnxElementType {
        static final String NAME = "int32";

        public Int32Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Int64Type extends OnnxElementType {
        static final String NAME = "int64";

        public Int64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Uint4Type extends OnnxElementType {
        static final String NAME = "uint4";

        public Uint4Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Uint8Type extends OnnxElementType {
        static final String NAME = "uint8";

        public Uint8Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Uint16Type extends OnnxElementType {
        static final String NAME = "uint16";

        public Uint16Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Uint32Type extends OnnxElementType {
        static final String NAME = "uint32";

        public Uint32Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Uint64Type extends OnnxElementType {
        static final String NAME = "uint64";

        public Uint64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Complex64Type extends OnnxElementType {
        static final String NAME = "complex64";

        public Complex64Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class Complex128Type extends OnnxElementType {
        static final String NAME = "complex128";

        public Complex128Type() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class BoolType extends OnnxElementType {
        static final String NAME = "bool";

        public BoolType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }

    public static final class StringType extends OnnxElementType {
        static final String NAME = "string";

        public StringType() {
        }

        @Override
        public ExternalizedTypeElement externalize() {
            return new ExternalizedTypeElement(NAME, List.of());
        }
    }


    @Override
    public String toString() {
        return externalize().toString();
    }
}