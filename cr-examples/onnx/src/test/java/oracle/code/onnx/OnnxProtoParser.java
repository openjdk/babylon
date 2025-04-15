package oracle.code.onnx;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static oracle.code.onnx.OnnxProtoPrinter.decodeVarint;

public final class OnnxProtoParser {

    record Attribute(
            String name,
            Float f,
            Long i,
            byte[] s,
            TensorProto t,
            GraphProto g,
            List<float[]> floats,
            List<long[]> ints,
            List<byte[]> strings,
            List<TensorProto> tensors,
            List<GraphProto> graphs,
            String doc_string,
            TypeProto tp,
            List<TypeProto> type_protos,
            Long type,
            String ref_attr_name,
            SparseTensorProto sparse_tensor,
            List<SparseTensorProto> sparse_tensors) {}

    record ValueInfoProto(
            String name,
            TypeProto type,
            String doc_string,
            List<StringStringEntryProto> metadata_props) {}

    record NodeProto(
            List<String> input,
            List<String> output,
            String name,
            String op_type,
            List<Attribute> attribute,
            String doc_string,
            String domain,
            String overload,
            List<StringStringEntryProto> metadata_props) {}

    record TrainingInfoProto(
            GraphProto initialization,
            GraphProto algorithm,
            List<StringStringEntryProto> initialization_binding,
            List<StringStringEntryProto> update_binding) {}

    record ModelProto(
            Long ir_version,
            String producer_name,
            String producer_version,
            String domain,
            Long model_version,
            String doc_string,
            GraphProto graph,
            List<OperatorSetIdProto> opset_import,
            List<StringStringEntryProto> metadata_props,
            List<TrainingInfoProto> training_info,
            List<FunctionProto> functions) {}

    record StringStringEntryProto(
            String key,
            String value) {}

    record TensorAnnotation(
            String tensor_name,
            List<StringStringEntryProto> quant_parameter_tensor_names) {}

    record GraphProto(
            List<NodeProto> node,
            String name,
            List<TensorProto> initializer,
            String doc_string,
            List<ValueInfoProto> input,
            List<ValueInfoProto> output,
            List<ValueInfoProto> value_info,
            List<TensorAnnotation> quantization_annotation,
            List<SparseTensorProto> sparse_initializer,
            List<StringStringEntryProto> metadata_props) {}

    record TensorProto(
            List<long[]> dims,
            Long data_type,
            Segment segment,
            List<float[]> float_data,
            List<long[]> int32_data,
            List<byte[]> string_data,
            List<long[]> int64_data,
            String name,
            byte[] raw_data,
            List<double[]> double_data,
            List<long[]> uint64_data,
            String doc_string,
            List<StringStringEntryProto> external_data,
            Long data_location,
            List<StringStringEntryProto> metadata_props) {}

    record Segment(
            Long begin,
            Long end) {}

    record SparseTensorProto(
            TensorProto values,
            TensorProto indices,
            List<long[]> dims) {}

    record TensorShapeProto(
            List<Dimension> dim) {}

    record Dimension(
            Long dim_value,
            String dim_param,
            String denotation) {}

    record TypeProto(
            Tensor tensor_type,
            Sequence sequence_type,
            Map map_type,
            String denotation,
            SparseTensor sparse_tensor_type,
            Optional optional_type) {}

    record Tensor(
            Long elem_type,
            TensorShapeProto shape) {}

    record Sequence(
            TypeProto elem_type) {}

    record Map(
            Long key_type,
            TypeProto value_type) {}

    record Optional(
            TypeProto elem_type) {

    }

    record SparseTensor(
            Long elem_type,
            TensorShapeProto shape) {}

    record OperatorSetIdProto(
            String domain,
            Long version) {}

    record FunctionProto(
            String name,
            List<String> input,
            List<String> output,
            List<String> attribute,
            List<NodeProto> node,
            String doc_string,
            List<OperatorSetIdProto> opset_import,
            String domain,
            List<Attribute> attribute_proto,
            List<ValueInfoProto> value_info,
            String overload,
            List<StringStringEntryProto> metadata_props) {}




    private record Node(OnnxProtoPrinter type, Object[] fieldsData) {
        Node(OnnxProtoPrinter type) {
            this(type, new Object[type.fields[type.fields.length - 1].fieldNum() + 1]);
        }

        void set(int fieldIndex, int compactIndex, Object data) {
            if (type.fields[fieldIndex].repeated()) {
                if (fieldsData[compactIndex] instanceof List l) {
                    l.add(data);
                } else {
                    var l = new ArrayList();
                    l.add(data);
                    fieldsData[compactIndex] = l;
                }
            } else {
                fieldsData[compactIndex] = data;
            }
        }
    }

    private static Object parse(OnnxProtoPrinter type, ByteBuffer data) throws Exception {
        Node n = new Node(type);
        while (data.remaining() > 0) {
            long tag = decodeVarint(data);
            int fIndex = ((int)tag >> 3) - 1;
            var f = type.fields[fIndex];
            boolean packed = (tag & 7) == 2;
            n.set(fIndex, f.fieldNum(), switch (f.type()) {
                case INT -> decodeVarint(data);
                case INTS -> packed ? readPackedLongs(data) : new long[]{decodeVarint(data)};
                case FLOAT -> data.getFloat();
                case FLOATS -> packed ? readPackedFloats(data) : new float[] {data.getFloat()};
                case DOUBLE -> data.getDouble();
                case DOUBLES -> packed ? readPackedDoubles(data) : new double[] {data.getDouble()};
                case BYTES -> readBytes(data);
                case STRING -> new String(readBytes(data));
                default -> {
                    var size = decodeVarint(data);
                    int limit = data.limit();
                    Object child = parse(f.type(), data.limit(data.position() + (int)size));
                    data.limit(limit);
                    yield child;
                }
            });
        }
        var constructor = Stream.of(OnnxProtoParser.class.getDeclaredClasses()).filter(cls -> cls.getName().endsWith("$" + n.type().name())).findFirst().orElseThrow().getDeclaredConstructors()[0];
        return constructor.newInstance(n.fieldsData());
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

    public static void main(String... args) throws Exception {
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                Object model = parse(OnnxProtoPrinter.ModelProto, in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()).order(ByteOrder.LITTLE_ENDIAN));
                System.out.println(model);
            }
        }
    }
}
