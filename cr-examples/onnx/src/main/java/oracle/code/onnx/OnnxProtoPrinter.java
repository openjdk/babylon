package oracle.code.onnx;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public enum OnnxProtoPrinter {
    BYTES, INT, INTS, FLOAT, FLOATS, DOUBLE, DOUBLES, STRING,
    Attribute, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, StringStringEntryProto, TensorAnnotation,
    GraphProto, TensorProto, Segment, SparseTensorProto, TensorShapeProto, Dimension, TypeProto, Tensor, Sequence,
    Map, Optional, SparseTensor, OperatorSetIdProto, FunctionProto;

    static {
        init(Attribute,
                1, "name", STRING,
                2, "f", FLOAT,
                3, "i", INT,
                4, "s", BYTES,
                5, "t", TensorProto,
                6, "g", GraphProto,
                7, "floats", FLOATS,
                8, "ints", INTS,
                9, "strings", BYTES,
               10, "tensors", TensorProto,
               11, "graphs", GraphProto,
               13, "doc_string", STRING,
               14, "tp", TypeProto,
               15, "type_protos", TypeProto,
               20, "type", INT,
               21, "ref_attr_name", STRING,
               22, "sparse_tensor", SparseTensorProto,
               23, "sparse_tensors", SparseTensorProto);
        init(ValueInfoProto,
                1, "name", STRING,
                2, "type", TypeProto,
                3, "doc_string", STRING,
                4, "metadata_props", StringStringEntryProto);
        init(NodeProto,
                1, "input", STRING,
                2, "output", STRING,
                3, "name", STRING,
                4, "op_type", STRING,
                5, "attribute", Attribute,
                6, "doc_string", STRING,
                7, "domain", STRING,
                8, "overload", STRING,
                9, "metadata_props", StringStringEntryProto);
        init(TrainingInfoProto,
                1, "initialization", GraphProto,
                2, "algorithm", GraphProto,
                3, "initialization_binding", StringStringEntryProto,
                4, "update_binding", StringStringEntryProto);
        init(ModelProto,
                1, "ir_version", INT,
                2, "producer_name", STRING,
                3, "producer_version", STRING,
                4, "domain", STRING,
                5, "model_version", INT,
                6, "doc_string", STRING,
                7, "graph", GraphProto,
                8, "opset_import", OperatorSetIdProto,
                14, "metadata_props", StringStringEntryProto,
                20, "training_info", TrainingInfoProto,
                25, "functions", FunctionProto);
        init(StringStringEntryProto,
                1, "key", STRING,
                2, "value", STRING);
        init(TensorAnnotation,
                1, "tensor_name", STRING,
                2, "quant_parameter_tensor_names", StringStringEntryProto);
        init(GraphProto,
                1, "node", NodeProto,
                2, "name", STRING,
                5, "initializer", TensorProto,
                10, "doc_string", STRING,
                11, "input", ValueInfoProto,
                12, "output", ValueInfoProto,
                13, "value_info", ValueInfoProto,
                14, "quantization_annotation", TensorAnnotation,
                15, "sparse_initializer", SparseTensorProto,
                16, "metadata_props", StringStringEntryProto);
        init(TensorProto,
                1, "dims", INTS,
                2, "data_type", INT,
                3, "segment", Segment,
                4, "float_data", FLOATS,
                5, "int32_data", INTS,
                6, "string_data", BYTES,
                7, "int64_data", INTS,
                8, "name", STRING,
                9, "raw_data", BYTES,
                10, "double_data", DOUBLES,
                11, "uint64_data", INTS,
                12, "doc_string", STRING,
                13, "external_data", StringStringEntryProto,
                14, "data_location", INT,
                16, "metadata_props", StringStringEntryProto);
        init(Segment,
                1, "begin", INT,
                2, "end", INT);
        init(SparseTensorProto,
                1, "values", TensorProto,
                2, "indices", TensorProto,
                3, "dims", INTS);
        init(TensorShapeProto,
                1, "dim", Dimension);
        init(Dimension,
                1, "dim_value", INT,
                2, "dim_param", STRING,
                3, "denotation", STRING);
        init(TypeProto,
                1, "tensor_type", Tensor,
                4, "sequence_type", Sequence,
                5, "map_type", Map,
                6, "denotation", STRING,
                8, "sparse_tensor_type", SparseTensor,
                9, "optional_type", Optional);
        init(Tensor,
                1, "elem_type", INT,
                2, "shape", TensorShapeProto);
        init(Sequence,
                1, "elem_type", TypeProto);
        init(Map,
                1, "key_type", INT,
                2, "value_type", TypeProto);
        init(Optional,
                1, "elem_type", TypeProto);
        init(SparseTensor,
                1, "elem_type", INT,
                2, "shape", TensorShapeProto);
        init(OperatorSetIdProto,
                1, "domain", STRING,
                2, "version", INT);
        init(FunctionProto,
                1, "name", STRING,
                4, "input", STRING,
                5, "output", STRING,
                6, "attribute", STRING,
                7, "node", NodeProto,
                8, "doc_string", STRING,
                9, "opset_import", OperatorSetIdProto,
                10, "domain", STRING,
                11, "attribute_proto", Attribute,
                12, "value_info", ValueInfoProto,
                13, "overload", STRING,
                14, "metadata_props", StringStringEntryProto);
    }

    private record Field(String name, OnnxProtoPrinter type) {}

    private static void init(OnnxProtoPrinter proto, Object... fields) {
        proto.fields = new Field[(int)fields[fields.length - 3]];
        for (int i = 0; i < fields.length; i += 3) {
            proto.fields[(int)fields[i] - 1] = new Field((String)fields[i + 1], (OnnxProtoPrinter)fields[i + 2]);
        }
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

    private Field[] fields;

    public void print(int indent, ByteBuffer data) {
        data = data.order(ByteOrder.nativeOrder());
        while (data.remaining() > 0) {
            long tag = decodeVarint(data);
            var f = fields[((int)tag >> 3) - 1];
            boolean packed = (tag & 7) == 2;
            System.out.print("    ".repeat(indent) + f.type() + " " + f.name() + " ");
            switch (f.type) {
                case INT ->
                    System.out.println(decodeVarint(data));
                case INTS -> {
                    if (packed) {
                        var size = decodeVarint(data);
                        var stop = data.position() + size;
                        while (data.position() < stop) {
                            System.out.print(decodeVarint(data) + " ");
                        }
                        System.out.println();
                    } else {
                        System.out.println(decodeVarint(data));
                    }
                }
                case FLOAT ->
                    System.out.println(data.getFloat());
                case FLOATS -> {
                    if (packed) {
                        var size = decodeVarint(data);
                        var stop = data.position() + size;
                        while (data.position() < stop) {
                            System.out.print(data.getFloat() + " ");
                        }
                        System.out.println();
                    } else {
                        System.out.println(data.getFloat());
                    }
                }
                case DOUBLE ->
                    System.out.println(data.getDouble());
                case DOUBLES -> {
                    if (packed) {
                        var size = decodeVarint(data);
                        var stop = data.position() + size;
                        while (data.position() < stop) {
                            System.out.print(data.getDouble() + " ");
                        }
                        System.out.println();
                    } else {
                        System.out.println(data.getDouble());
                    }
                }
                case BYTES -> {
                    var bytes = new byte[(int)decodeVarint(data)];
                    data.get(bytes);
                    System.out.println(Arrays.toString(bytes));
                }
                case STRING -> {
                    var bytes = new byte[(int)decodeVarint(data)];
                    data.get(bytes);
                    System.out.println('"' + new String(bytes) + '"');
                }
                default -> {
                    var size = decodeVarint(data);
                    int limit = data.limit();
                    System.out.println();
                    f.type().print(indent + 1, data.limit(data.position() + (int)size));
                    data.limit(limit);
                }
            }
        }
    }

    public static void printModel(byte[] model) {
        ModelProto.print(0, ByteBuffer.wrap(model));
    }

    public static void main(String... args) throws Exception {
        for (var fName : args) {
            System.out.println(fName);
            try (var in = new RandomAccessFile(fName, "r")) {
                ModelProto.print(1, in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()).order(ByteOrder.LITTLE_ENDIAN));
            }
        }
    }
}
