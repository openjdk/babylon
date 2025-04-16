package oracle.code.onnx;

import java.io.RandomAccessFile;
import java.lang.reflect.Constructor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.stream.Stream;

import static oracle.code.onnx.OnnxProtoPrinter.decodeVarint;

public final class OnnxProtoParser {

    public record Attribute(
            Void _0,
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
            List<SparseTensorProto> sparse_tensors) {
    }

    public record ValueInfoProto(
            String name,
            TypeProto type,
            String doc_string,
            List<StringStringEntryProto> metadata_props) {
    }

    public record NodeProto(
            List<String> input,
            List<String> output,
            String name,
            String op_type,
            List<Attribute> attribute,
            String doc_string,
            String domain,
            String overload,
            List<StringStringEntryProto> metadata_props) {
    }

    public record TrainingInfoProto(
            GraphProto initialization,
            GraphProto algorithm,
            List<StringStringEntryProto> initialization_binding,
            List<StringStringEntryProto> update_binding) {
    }

    public record ModelProto(
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
            List<FunctionProto> functions) {
    }

    public record StringStringEntryProto(
            String key,
            String value) {
    }

    public record TensorAnnotation(
            String tensor_name,
            List<StringStringEntryProto> quant_parameter_tensor_names) {
    }

    public record GraphProto(
            List<NodeProto> node,
            String name,
            List<TensorProto> initializer,
            String doc_string,
            List<ValueInfoProto> input,
            List<ValueInfoProto> output,
            List<ValueInfoProto> value_info,
            List<TensorAnnotation> quantization_annotation,
            List<SparseTensorProto> sparse_initializer,
            List<StringStringEntryProto> metadata_props) {
    }

    public record TensorProto(
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
            List<StringStringEntryProto> metadata_props) {
    }

    public record Segment(
            Long begin,
            Long end) {
    }

    public record SparseTensorProto(
            TensorProto values,
            TensorProto indices,
            List<long[]> dims) {
    }

    public record TensorShapeProto(
            List<Dimension> dim) {
    }

    public record Dimension(
            Long dim_value,
            String dim_param,
            String denotation) {
    }

    public record TypeProto(
            Tensor tensor_type,
            Sequence sequence_type,
            Map map_type,
            String denotation,
            SparseTensor sparse_tensor_type,
            Optional optional_type) {
    }

    public record Tensor(
            Long elem_type,
            TensorShapeProto shape) {
    }

    public record Sequence(
            TypeProto elem_type) {
    }

    public record Map(
            Long key_type,
            TypeProto value_type) {
    }

    public record Optional(
            TypeProto elem_type) {
    }

    public record SparseTensor(
            Long elem_type,
            TensorShapeProto shape) {
    }

    public record OperatorSetIdProto(
            String domain,
            Long version) {
    }

    public record FunctionProto(
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
            List<StringStringEntryProto> metadata_props) {
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

    private final EnumMap<OnnxProtoPrinter, Constructor> constructors = new EnumMap<>(OnnxProtoPrinter.class);

    private Object parse(OnnxProtoPrinter type, ByteBuffer bb) {
        Object[] fieldsData = new Object[type.fields[type.fields.length - 1].fieldNum() + 1];
        while (bb.remaining() > 0) {
            long tag = decodeVarint(bb);
            int fIndex = ((int)tag >> 3) - 1;
            var f = type.fields[fIndex];
            boolean packed = (tag & 7) == 2;
            Object data = switch (f.type()) {
                case INT -> decodeVarint(bb);
                case INTS -> packed ? readPackedLongs(bb) : new long[]{decodeVarint(bb)};
                case FLOAT -> bb.getFloat();
                case FLOATS -> packed ? readPackedFloats(bb) : new float[] {bb.getFloat()};
                case DOUBLE -> bb.getDouble();
                case DOUBLES -> packed ? readPackedDoubles(bb) : new double[] {bb.getDouble()};
                case BYTES -> readBytes(bb);
                case STRING -> new String(readBytes(bb));
                default -> {
                    var size = decodeVarint(bb);
                    int limit = bb.limit();
                    Object child = parse(f.type(), bb.limit(bb.position() + (int)size));
                    bb.limit(limit);
                    yield child;
                }
            };
            if (f.repeated()) {
                if (fieldsData[f.fieldNum()] instanceof List l) {
                    l.add(data);
                } else {
                    var l = new ArrayList();
                    l.add(data);
                    fieldsData[f.fieldNum()] = l;
                }
            } else {
                fieldsData[f.fieldNum()] = data;
            }

        }
        var constructor = constructors.computeIfAbsent(type, _ ->
                Stream.of(OnnxProtoParser.class.getDeclaredClasses())
                        .filter(cls -> cls.getName().endsWith("$" + type.name()))
                        .findFirst().orElseThrow().getDeclaredConstructors()[0]);
        try {
            return constructor.newInstance(fieldsData);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public ModelProto parse(ByteBuffer onnxProtoModel) {
        return (ModelProto)parse(OnnxProtoPrinter.ModelProto, onnxProtoModel);
    }

    public static void main(String... args) throws Exception {
        var parser = new OnnxProtoParser();
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                ModelProto model = parser.parse(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()).order(ByteOrder.LITTLE_ENDIAN));
                System.out.println(model);
            }
        }
    }
}
