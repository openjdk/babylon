package oracle.code.onnx;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;

// @@@ for tests and debug purposes, later it can be moved to tests
enum OnnxProtoPrinter {
    BYTES, INT, INTS, FLOAT, FLOATS, DOUBLE, DOUBLES, STRING,
    Attribute, ValueInfoProto, NodeProto, TrainingInfoProto, ModelProto, StringStringEntryProto, TensorAnnotation,
    GraphProto, TensorProto, Segment, SparseTensorProto, TensorShapeProto, Dimension, TypeProto, Tensor, Sequence,
    Map, Optional, SparseTensor, OperatorSetIdProto, FunctionProto;

    static {
        init(Attribute,
                1, "name", STRING, false,
                2, "f", FLOAT, false,
                3, "i", INT, false,
                4, "s", BYTES, false,
                5, "t", TensorProto, false,
                6, "g", GraphProto, false,
                7, "floats", FLOATS, true,
                8, "ints", INTS, true,
                9, "strings", BYTES, true,
               10, "tensors", TensorProto, true,
               11, "graphs", GraphProto, true,
               13, "doc_string", STRING, false,
               14, "tp", TypeProto, false,
               15, "type_protos", TypeProto, true,
               20, "type", INT, false,
               21, "ref_attr_name", STRING, false,
               22, "sparse_tensor", SparseTensorProto, false,
               23, "sparse_tensors", SparseTensorProto, true);
        init(ValueInfoProto,
                1, "name", STRING, false,
                2, "type", TypeProto, false,
                3, "doc_string", STRING, false,
                4, "metadata_props", StringStringEntryProto, true);
        init(NodeProto,
                1, "input", STRING, true,
                2, "output", STRING, true,
                3, "name", STRING, false,
                4, "op_type", STRING, false,
                5, "attribute", Attribute, true,
                6, "doc_string", STRING, false,
                7, "domain", STRING, false,
                8, "overload", STRING, false,
                9, "metadata_props", StringStringEntryProto, true);
        init(TrainingInfoProto,
                1, "initialization", GraphProto, false,
                2, "algorithm", GraphProto, false,
                3, "initialization_binding", StringStringEntryProto, true,
                4, "update_binding", StringStringEntryProto, true);
        init(ModelProto,
                1, "ir_version", INT, false,
                2, "producer_name", STRING, false,
                3, "producer_version", STRING, false,
                4, "domain", STRING, false,
                5, "model_version", INT, false,
                6, "doc_string", STRING, false,
                7, "graph", GraphProto, false,
                8, "opset_import", OperatorSetIdProto, true,
                14, "metadata_props", StringStringEntryProto, true,
                20, "training_info", TrainingInfoProto, true,
                25, "functions", FunctionProto, true);
        init(StringStringEntryProto,
                1, "key", STRING, false,
                2, "value", STRING, false);
        init(TensorAnnotation,
                1, "tensor_name", STRING, false,
                2, "quant_parameter_tensor_names", StringStringEntryProto, true);
        init(GraphProto,
                1, "node", NodeProto, true,
                2, "name", STRING, false,
                5, "initializer", TensorProto, true,
                10, "doc_string", STRING, false,
                11, "input", ValueInfoProto, true,
                12, "output", ValueInfoProto, true,
                13, "value_info", ValueInfoProto, true,
                14, "quantization_annotation", TensorAnnotation, true,
                15, "sparse_initializer", SparseTensorProto, true,
                16, "metadata_props", StringStringEntryProto, true);
        init(TensorProto,
                1, "dims", INTS, true,
                2, "data_type", INT, false,
                3, "segment", Segment, false,
                4, "float_data", FLOATS, true,
                5, "int32_data", INTS, true,
                6, "string_data", BYTES, true,
                7, "int64_data", INTS, true,
                8, "name", STRING, false,
                9, "raw_data", BYTES, false,
                10, "double_data", DOUBLES, true,
                11, "uint64_data", INTS, true,
                12, "doc_string", STRING, false,
                13, "external_data", StringStringEntryProto, true,
                14, "data_location", INT, false,
                16, "metadata_props", StringStringEntryProto, true);
        init(Segment,
                1, "begin", INT, false,
                2, "end", INT, false);
        init(SparseTensorProto,
                1, "values", TensorProto, false,
                2, "indices", TensorProto, false,
                3, "dims", INTS, true);
        init(TensorShapeProto,
                1, "dim", Dimension, true);
        init(Dimension,
                1, "dim_value", INT, false,
                2, "dim_param", STRING, false,
                3, "denotation", STRING, false);
        init(TypeProto,
                1, "tensor_type", Tensor, false,
                4, "sequence_type", Sequence, false,
                5, "map_type", Map, false,
                6, "denotation", STRING, false,
                8, "sparse_tensor_type", SparseTensor, false,
                9, "optional_type", Optional, false);
        init(Tensor,
                1, "elem_type", INT, false,
                2, "shape", TensorShapeProto, false);
        init(Sequence,
                1, "elem_type", TypeProto, false);
        init(Map,
                1, "key_type", INT, false,
                2, "value_type", TypeProto, false);
        init(Optional,
                1, "elem_type", TypeProto, false);
        init(SparseTensor,
                1, "elem_type", INT, false,
                2, "shape", TensorShapeProto, false);
        init(OperatorSetIdProto,
                1, "domain", STRING, false,
                2, "version", INT, false);
        init(FunctionProto,
                1, "name", STRING, false,
                4, "input", STRING, true,
                5, "output", STRING, true,
                6, "attribute", STRING, true,
                7, "node", NodeProto, true,
                8, "doc_string", STRING, false,
                9, "opset_import", OperatorSetIdProto, true,
                10, "domain", STRING, false,
                11, "attribute_proto", Attribute, true,
                12, "value_info", ValueInfoProto, true,
                13, "overload", STRING, false,
                14, "metadata_props", StringStringEntryProto, true);
    }

    record Field(String name, OnnxProtoPrinter type, boolean repeated) {}

    private static void init(OnnxProtoPrinter proto, Object... fields) {
        proto.fields = new Field[(int)fields[fields.length - 4]];
        for (int i = 0; i < fields.length; i += 4) {
            proto.fields[(int)fields[i] - 1] = new Field((String)fields[i + 1], (OnnxProtoPrinter)fields[i + 2], (boolean)fields[i + 3]);
        }
    }

    static long decodeVarint(ByteBuffer data) {
        int i, shift = 0;
        long value = 0;
        do {
            value |= ((i = data.get()) & 0x7f) << shift;
            shift += 7;
        } while ((i & 0x80) != 0);
        return value;
    }

    Field[] fields;

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
