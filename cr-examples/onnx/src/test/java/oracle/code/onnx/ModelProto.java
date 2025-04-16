package oracle.code.onnx;

import java.io.RandomAccessFile;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public record ModelProto (
        @f(1) Long ir_version,
        @f(2) String producer_name,
        @f(3) String producer_version,
        @f(4) String domain,
        @f(5) Long model_version,
        @f(6) String doc_string,
        @f(7) GraphProto graph,
        @f(8) List<OperatorSetIdProto> opset_import,
        @f(14) List<StringStringEntryProto> metadata_props,
        @f(20) List<TrainingInfoProto> training_info,
        @f(25) List<FunctionProto> functions) {

    public record Attribute (
            @f(1) String name,
            @f(2) Float f,
            @f(3) Long i,
            @f(4) byte[] s,
            @f(5) TensorProto t,
            @f(6) GraphProto g,
            @f(7) List<float[]> floats,
            @f(8) List<long[]> ints,
            @f(9) List<byte[]> strings,
            @f(10) List<TensorProto> tensors,
            @f(11) List<GraphProto> graphs,
            @f(13) String doc_string,
            @f(14) TypeProto tp,
            @f(15) List<TypeProto> type_protos,
            @f(20) Long type,
            @f(21) String ref_attr_name,
            @f(22) SparseTensorProto sparse_tensor,
            @f(23) List<SparseTensorProto> sparse_tensors) {
    }

    public record ValueInfoProto (
            @f(1) String name,
            @f(2) TypeProto type,
            @f(3) String doc_string,
            @f(4) List<StringStringEntryProto> metadata_props) {
    }

    public record NodeProto (
            @f(1) List<String> input,
            @f(2) List<String> output,
            @f(3) String name,
            @f(4) String op_type,
            @f(5) List<Attribute> attribute,
            @f(6) String doc_string,
            @f(7) String domain,
            @f(8) String overload,
            @f(9) List<StringStringEntryProto> metadata_props) {
    }

    public record TrainingInfoProto (
            @f(1) GraphProto initialization,
            @f(2) GraphProto algorithm,
            @f(3) List<StringStringEntryProto> initialization_binding,
            @f(4) List<StringStringEntryProto> update_binding) {
    }

    public record StringStringEntryProto (
            @f(1) String key,
            @f(2) String value) {
    }

    public record TensorAnnotation (
            @f(1) String tensor_name,
            @f(2) List<StringStringEntryProto> quant_parameter_tensor_names) {
    }

    public record GraphProto (
            @f(1) List<NodeProto> node,
            @f(2) String name,
            @f(5) List<TensorProto> initializer,
            @f(10) String doc_string,
            @f(11) List<ValueInfoProto> input,
            @f(12) List<ValueInfoProto> output,
            @f(13) List<ValueInfoProto> value_info,
            @f(14) List<TensorAnnotation> quantization_annotation,
            @f(15) List<SparseTensorProto> sparse_initializer,
            @f(16) List<StringStringEntryProto> metadata_props) {
    }

    public record TensorProto (
            @f(1) List<long[]> dims,
            @f(2) Long data_type,
            @f(3) Segment segment,
            @f(4) List<float[]> float_data,
            @f(5) List<long[]> int32_data,
            @f(6) List<byte[]> string_data,
            @f(7) List<long[]> int64_data,
            @f(8) String name,
            @f(9) byte[] raw_data,
            @f(10) List<double[]> double_data,
            @f(11) List<long[]> uint64_data,
            @f(12) String doc_string,
            @f(13) List<StringStringEntryProto> external_data,
            @f(14) Long data_location,
            @f(16) List<StringStringEntryProto> metadata_props) {

        public record Segment (
                @f(1) Long begin,
                @f(2) Long end) {
        }
    }

    public record SparseTensorProto (
            @f(1) TensorProto values,
            @f(2) TensorProto indices,
            @f(3) List<long[]> dims) {
    }

    public record TensorShapeProto (
            @f(1) List<Dimension> dim) {

        public record Dimension (
                @f(1) Long dim_value,
                @f(2) String dim_param,
                @f(3) String denotation) {
        }
    }

    public record TypeProto (
            @f(1) Tensor tensor_type,
            @f(4) Sequence sequence_type,
            @f(5) Map map_type,
            @f(6) String denotation,
            @f(8) SparseTensor sparse_tensor_type,
            @f(9) Optional optional_type) {

        public record Tensor (
                @f(1) Long elem_type,
                @f(2) TensorShapeProto shape) {
        }

        public record Sequence (
                @f(1) TypeProto elem_type) {
        }

        public record Map (
                @f(1) Long key_type,
                @f(2) TypeProto value_type) {
        }

        public record Optional (
                @f(1) TypeProto elem_type) {
        }

        public record SparseTensor (
                @f(1) Long elem_type,
                @f(2) TensorShapeProto shape) {
        }
    }

    public record OperatorSetIdProto (
            @f(1) String domain,
            @f(2) Long version) {
    }

    public record FunctionProto (
            @f(1) String name,
            @f(4) List<String> input,
            @f(5) List<String> output,
            @f(6) List<String> attribute,
            @f(7) List<NodeProto> node,
            @f(8) String doc_string,
            @f(9) List<OperatorSetIdProto> opset_import,
            @f(10) String domain,
            @f(11) List<Attribute> attribute_proto,
            @f(12) List<ValueInfoProto> value_info,
            @f(13) String overload,
            @f(14) List<StringStringEntryProto> metadata_props) {
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.RECORD_COMPONENT)
    @interface f {
        int value();
    }

    private static long decodeVarint(ByteBuffer data) {
        int i, shift = 0;
        long value = 0;
        do {
            value |= ((i = data.get()) & 0x7f) << shift;
            shift += 7;
        } while ((i & 0x80) != 0);
        return value;
    }

    private static long[] readPackedLongs(ByteBuffer data) {
        var ret = new long[(int)(decodeVarint(data)/4)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = decodeVarint(data);
        }
        return ret;
    }

    private static float[] readPackedFloats(ByteBuffer data) {
        var ret = new float[(int)(decodeVarint(data)/4)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getFloat();
        }
        return ret;
    }

    private static double[] readPackedDoubles(ByteBuffer data) {
        var ret = new double[(int)(decodeVarint(data)/8)];
        for (int i = 0; i < ret.length; i++) {
            ret[i] = data.getDouble();
        }
        return ret;
    }

    private static byte[] readBytes(ByteBuffer data) {
        var bytes = new byte[(int)decodeVarint(data)];
        data.get(bytes);
        return bytes;
    }

    private static Object readData(Class<?> baseType, boolean packed, ByteBuffer bb) {
        if (baseType == Long.class) {
            return decodeVarint(bb);
        } else if (baseType == long[].class) {
            return packed ? readPackedLongs(bb) : new long[]{decodeVarint(bb)};
        } else if (baseType == Float.class) {
            return bb.getFloat();
        } else if (baseType == float[].class) {
            return packed ? readPackedFloats(bb) : new float[] {bb.getFloat()};
        } else if (baseType == Double.class) {
            return bb.getDouble();
        } else if (baseType == double[].class) {
            return packed ? readPackedDoubles(bb) : new double[] {bb.getDouble()};
        } else if (baseType == byte[].class) {
            return readBytes(bb);
        } else if (baseType == String.class) {
            return new String(readBytes(bb));
        } else {
            var size = decodeVarint(bb);
            int limit = bb.limit();
            var data = readFrom((Class<Record>)baseType, bb.limit(bb.position() + (int)size));
            bb.limit(limit);
            return data;
        }
    }

    private static int getRecordFieldIndex(RecordComponent[] rcs, int fieldIndex) {
        for (int i = 0; i < rcs.length; i++) {
            if (rcs[i].getAnnotation(f.class).value() == fieldIndex) {
                return i;
            }
        }
        throw new IllegalArgumentException("Field index " + fieldIndex + " not found in " + rcs[0].getDeclaringRecord());
    }

    private static <T> T readFrom(Class<T> type, ByteBuffer bb) {
        Object[] fieldsData = new Object[type.getRecordComponents().length];
        while (bb.remaining() > 0) {
            long tag = decodeVarint(bb);
            RecordComponent[] rcs = type.getRecordComponents();
            int rfi = getRecordFieldIndex(rcs, (int)tag >> 3);
            boolean packed = (tag & 7) == 2;
            RecordComponent rc = rcs[rfi];
            Class<?> rcType = rc.getType();
            if (rcType == List.class) {
                List list;
                if (fieldsData[rfi] instanceof List l) {
                    list = l;
                } else {
                    list = new ArrayList();
                    fieldsData[rfi] = list;
                }
                Class baseType = (Class)((ParameterizedType)rc.getGenericType()).getActualTypeArguments()[0];
                list.add(readData(baseType, packed, bb));
            } else {
                fieldsData[rfi] = readData(rcType, packed, bb);
            }
        }
        try {
            return (T)type.getDeclaredConstructors()[0].newInstance(fieldsData);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    private static void print(int indent, String name, Object value) throws ReflectiveOperationException {
        if (value == null) return;
        System.out.print("  ".repeat(indent) + name);
        switch (value) {
            case List l -> {
                System.out.println(name.matches(".*[shxz]") ? "es:" : "s:");
                for (var el : l) print(indent + 1, "- " + name, el);
            }
            case Record r -> {
                System.out.println(":");
                for (var rc : r.getClass().getRecordComponents()) {
                    print(indent + 2, rc.getName(), rc.getAccessor().invoke(r));
                }
            }
            case byte[] a ->
                System.out.println(a.length > PRINT_LIMIT ? ": [] # data too big to print" : ": " + Arrays.toString(a));
            case long[] a ->
                System.out.println(a.length > PRINT_LIMIT ? ": [] # data too big to print" : ": " + Arrays.toString(a));
            case float[] a ->
                System.out.println(a.length > PRINT_LIMIT ? ": [] # data too big to print" : ": " + Arrays.toString(a));
            case double[] a ->
                System.out.println(a.length > PRINT_LIMIT ? ": [] # data too big to print" : ": " + Arrays.toString(a));
            case String s ->
                System.out.println(": \"" + s + "\"");
            default ->
                System.out.println(": " + value);
        }
    }

    static final int PRINT_LIMIT = 1000;

    public void print() {
        try {
            print(0, "model", this);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public static ModelProto readFrom(ByteBuffer onnxProtoModel) {
        return readFrom(ModelProto.class, onnxProtoModel);
    }

    public static void main(String... args) throws Exception {
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                ModelProto model = readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()).order(ByteOrder.LITTLE_ENDIAN));
                model.print();
            }
        }
    }
}
