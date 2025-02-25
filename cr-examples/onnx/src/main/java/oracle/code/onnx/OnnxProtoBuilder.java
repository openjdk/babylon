package oracle.code.onnx;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.stream.IntStream;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.CoreOp.FuncOp;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxType;

// Generated from onnx.proto3
sealed class OnnxProtoBuilder<T extends OnnxProtoBuilder> {

    static final class Attribute extends OnnxProtoBuilder<Attribute> {
        Attribute name(String name) {return _f(1, name);}
        Attribute ref_attr_name(String ref_attr_name) {return _f(21, ref_attr_name);}
        Attribute doc_string(String doc_string) {return _f(13, doc_string);}
        Attribute type(int type) {return _f(20, type);}
        Attribute f(float f) {return _f(2, f);}
        Attribute i(long i) {return _f(3, i);}
        Attribute s(byte[] s) {return _f(4, s);}
        Attribute t(TensorProto t) {return _f(5, t);}
        Attribute g(GraphProto g) {return _f(6, g);}
        Attribute sparse_tensor(SparseTensorProto sparse_tensor) {return _f(22, sparse_tensor);}
        Attribute tp(TypeProto tp) {return _f(14, tp);}
        Attribute floats(float floats) {return _f(7, floats);}
        Attribute ints(long ints) {return _f(8, ints);}
        Attribute strings(byte[] strings) {return _f(9, strings);}
        Attribute tensors(TensorProto tensors) {return _f(10, tensors);}
        Attribute graphs(GraphProto graphs) {return _f(11, graphs);}
        Attribute sparse_tensors(SparseTensorProto sparse_tensors) {return _f(23, sparse_tensors);}
        Attribute type_protos(TypeProto type_protos) {return _f(15, type_protos);}
    }

    static final class ValueInfoProto extends OnnxProtoBuilder<ValueInfoProto> {
        ValueInfoProto name(String name) {return _f(1, name);}
        ValueInfoProto type(TypeProto type) {return _f(2, type);}
        ValueInfoProto doc_string(String doc_string) {return _f(3, doc_string);}
        ValueInfoProto metadata_props(StringStringEntryProto metadata_props) {return _f(4, metadata_props);}
    }

    static final class NodeProto extends OnnxProtoBuilder<NodeProto> {
        NodeProto input(String input) {return _f(1, input);}
        NodeProto output(String output) {return _f(2, output);}
        NodeProto name(String name) {return _f(3, name);}
        NodeProto op_type(String op_type) {return _f(4, op_type);}
        NodeProto domain(String domain) {return _f(7, domain);}
        NodeProto overload(String overload) {return _f(8, overload);}
        NodeProto attribute(Attribute attribute) {return _f(5, attribute);}
        NodeProto doc_string(String doc_string) {return _f(6, doc_string);}
        NodeProto metadata_props(StringStringEntryProto metadata_props) {return _f(9, metadata_props);}
    }

    static final class TrainingInfoProto extends OnnxProtoBuilder<TrainingInfoProto> {
        TrainingInfoProto initialization(GraphProto initialization) {return _f(1, initialization);}
        TrainingInfoProto algorithm(GraphProto algorithm) {return _f(2, algorithm);}
        TrainingInfoProto initialization_binding(StringStringEntryProto initialization_binding) {return _f(3, initialization_binding);}
        TrainingInfoProto update_binding(StringStringEntryProto update_binding) {return _f(4, update_binding);}
    }

    static final class ModelProto extends OnnxProtoBuilder<ModelProto> {
        ModelProto ir_version(long ir_version) {return _f(1, ir_version);}
        ModelProto opset_import(OperatorSetIdProto opset_import) {return _f(8, opset_import);}
        ModelProto producer_name(String producer_name) {return _f(2, producer_name);}
        ModelProto producer_version(String producer_version) {return _f(3, producer_version);}
        ModelProto domain(String domain) {return _f(4, domain);}
        ModelProto model_version(long model_version) {return _f(5, model_version);}
        ModelProto doc_string(String doc_string) {return _f(6, doc_string);}
        ModelProto graph(GraphProto graph) {return _f(7, graph);}
        ModelProto metadata_props(StringStringEntryProto metadata_props) {return _f(14, metadata_props);}
        ModelProto training_info(TrainingInfoProto training_info) {return _f(20, training_info);}
        ModelProto functions(FunctionProto functions) {return _f(25, functions);}
    }

    static final class StringStringEntryProto extends OnnxProtoBuilder<StringStringEntryProto> {
        StringStringEntryProto key(String key) {return _f(1, key);}
        StringStringEntryProto value(String value) {return _f(2, value);}
    }

    static final class TensorAnnotation extends OnnxProtoBuilder<TensorAnnotation> {
        TensorAnnotation tensor_name(String tensor_name) {return _f(1, tensor_name);}
        TensorAnnotation quant_parameter_tensor_names(StringStringEntryProto quant_parameter_tensor_names) {return _f(2, quant_parameter_tensor_names);}
    }

    static final class GraphProto extends OnnxProtoBuilder<GraphProto> {
        GraphProto node(NodeProto node) {return _f(1, node);}
        GraphProto name(String name) {return _f(2, name);}
        GraphProto initializer(TensorProto initializer) {return _f(5, initializer);}
        GraphProto sparse_initializer(SparseTensorProto sparse_initializer) {return _f(15, sparse_initializer);}
        GraphProto doc_string(String doc_string) {return _f(10, doc_string);}
        GraphProto input(ValueInfoProto input) {return _f(11, input);}
        GraphProto output(ValueInfoProto output) {return _f(12, output);}
        GraphProto value_info(ValueInfoProto value_info) {return _f(13, value_info);}
        GraphProto quantization_annotation(TensorAnnotation quantization_annotation) {return _f(14, quantization_annotation);}
        GraphProto metadata_props(StringStringEntryProto metadata_props) {return _f(16, metadata_props);}
    }

    static final class TensorProto extends OnnxProtoBuilder<TensorProto> {
        TensorProto dims(long dims) {return _f(1, dims);}
        TensorProto data_type(int data_type) {return _f(2, data_type);}
        TensorProto segment(Segment segment) {return _f(3, segment);}
        TensorProto float_data(float float_data) {return _f(4, float_data);}
        TensorProto int32_data(int int32_data) {return _f(5, int32_data);}
        TensorProto string_data(byte[] string_data) {return _f(6, string_data);}
        TensorProto int64_data(long int64_data) {return _f(7, int64_data);}
        TensorProto name(String name) {return _f(8, name);}
        TensorProto doc_string(String doc_string) {return _f(12, doc_string);}
        TensorProto raw_data(byte[] raw_data) {return _f(9, raw_data);}
        TensorProto external_data(StringStringEntryProto external_data) {return _f(13, external_data);}
        TensorProto data_location(int data_location) {return _f(14, data_location);}
        TensorProto double_data(double double_data) {return _f(10, double_data);}
        TensorProto uint64_data(long uint64_data) {return _f(11, uint64_data);}
        TensorProto metadata_props(StringStringEntryProto metadata_props) {return _f(16, metadata_props);}
    }

    static final class Segment extends OnnxProtoBuilder<Segment> {
        Segment begin(long begin) {return _f(1, begin);}
        Segment end(long end) {return _f(2, end);}
    }

    static final class SparseTensorProto extends OnnxProtoBuilder<SparseTensorProto> {
        SparseTensorProto values(TensorProto values) {return _f(1, values);}
        SparseTensorProto indices(TensorProto indices) {return _f(2, indices);}
        SparseTensorProto dims(long dims) {return _f(3, dims);}
    }

    static final class TensorShapeProto extends OnnxProtoBuilder<TensorShapeProto> {
        TensorShapeProto dim(Dimension dim) {return _f(1, dim);}
    }

    static final class Dimension extends OnnxProtoBuilder<Dimension> {
        Dimension dim_value(long dim_value) {return _f(1, dim_value);}
        Dimension dim_param(String dim_param) {return _f(2, dim_param);}
        Dimension denotation(String denotation) {return _f(3, denotation);}
    }

    static final class TypeProto extends OnnxProtoBuilder<TypeProto> {
        TypeProto tensor_type(Tensor tensor_type) {return _f(1, tensor_type);}
        TypeProto sequence_type(Sequence sequence_type) {return _f(4, sequence_type);}
        TypeProto map_type(Map map_type) {return _f(5, map_type);}
        TypeProto optional_type(Optional optional_type) {return _f(9, optional_type);}
        TypeProto sparse_tensor_type(SparseTensor sparse_tensor_type) {return _f(8, sparse_tensor_type);}
        TypeProto denotation(String denotation) {return _f(6, denotation);}
    }

    static final class Tensor extends OnnxProtoBuilder<Tensor> {
        Tensor elem_type(int elem_type) {return _f(1, elem_type);}
        Tensor shape(TensorShapeProto shape) {return _f(2, shape);}
    }

    static final class Sequence extends OnnxProtoBuilder<Sequence> {
        Sequence elem_type(TypeProto elem_type) {return _f(1, elem_type);}
    }

    static final class Map extends OnnxProtoBuilder<Map> {
        Map key_type(int key_type) {return _f(1, key_type);}
        Map value_type(TypeProto value_type) {return _f(2, value_type);}
    }

    static final class Optional extends OnnxProtoBuilder<Optional> {
        Optional elem_type(TypeProto elem_type) {return _f(1, elem_type);}
    }

    static final class SparseTensor extends OnnxProtoBuilder<SparseTensor> {
        SparseTensor elem_type(int elem_type) {return _f(1, elem_type);}
        SparseTensor shape(TensorShapeProto shape) {return _f(2, shape);}
    }

    static final class OperatorSetIdProto extends OnnxProtoBuilder<OperatorSetIdProto> {
        OperatorSetIdProto domain(String domain) {return _f(1, domain);}
        OperatorSetIdProto version(long version) {return _f(2, version);}
    }

    static final class FunctionProto extends OnnxProtoBuilder<FunctionProto> {
        FunctionProto name(String name) {return _f(1, name);}
        FunctionProto input(String input) {return _f(4, input);}
        FunctionProto output(String output) {return _f(5, output);}
        FunctionProto attribute(String attribute) {return _f(6, attribute);}
        FunctionProto attribute_proto(Attribute attribute_proto) {return _f(11, attribute_proto);}
        FunctionProto node(NodeProto node) {return _f(7, node);}
        FunctionProto doc_string(String doc_string) {return _f(8, doc_string);}
        FunctionProto opset_import(OperatorSetIdProto opset_import) {return _f(9, opset_import);}
        FunctionProto domain(String domain) {return _f(10, domain);}
        FunctionProto overload(String overload) {return _f(13, overload);}
        FunctionProto value_info(ValueInfoProto value_info) {return _f(12, value_info);}
        FunctionProto metadata_props(StringStringEntryProto metadata_props) {return _f(14, metadata_props);}
    }

    final ByteArrayOutputStream buf = new ByteArrayOutputStream();

    void _encode(long number) {
        for (int i = 64 - Long.numberOfLeadingZeros(number); i > 7; i -= 7) {
            buf.write(0x80 | (int)number & 0x7f);
            number >>= 7;
        }
        buf.write((int)number & 0x7f);
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, String value) {
        return _f(fieldIndex, value.getBytes(StandardCharsets.UTF_8));
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, byte[] bytes) {
        _encode(fieldIndex << 3 | 2);
        _encode(bytes.length);
        buf.writeBytes(bytes);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, float value) {
        _encode(fieldIndex << 3 | 5);
        int bits =  Float.floatToRawIntBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, double value) {
        _encode(fieldIndex << 3 | 1);
        long bits =  Double.doubleToRawLongBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
        buf.write((byte)(bits >> 32));
        buf.write((byte)(bits >> 40));
        buf.write((byte)(bits >> 48));
        buf.write((byte)(bits >> 56));
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, long value) {
        _encode(fieldIndex << 3);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, OnnxProtoBuilder value) {
        return _f(fieldIndex, value.buf.toByteArray());
    }

    @SuppressWarnings("unchecked")
    <P> T forEach(Iterable<P> sup, BiConsumer<T, ? super P> cons) {
        sup.forEach(p -> cons.accept((T)this, p));
        return (T)this;
    }

    static final int IR_VERSION = 10;
    static final int OPSET_VERSION = 21;

    // @@@ unchecked constraints:
    //         tensor FuncOp parameters and single tensor return type
    //         OnnxOps (with tensor operands and single tensor return value) and ReturnOp (returning single tensor)
    //         entry block only
    static byte[] build(FuncOp model) {
        var indexer = new IdentityHashMap<Value, String>() {
            String getName(Value v) {
                return computeIfAbsent(v, _ -> "#" + size());
            }
            String getName(Value v, int subIndex) {
                var name = getName(v);
                if (subIndex != 0) name += "." + subIndex;
                return name;
            }
        };
        return build(
                model.body().entryBlock().parameters().stream().map(v -> valueInfo(indexer.getName(v), ((OnnxType.TensorType)v.type()).eType().id())).toList(),
                model.body().entryBlock().ops().stream().<NodeProto>mapMulti((op, opNodes) -> {
                    switch (op) {
                        case OnnxOp onnxOp ->
                            opNodes.accept(node(
                                    onnxOp.opName(),
                                    onnxOp.operands().stream().map(v -> indexer.getName(v)).toList(),
                                    IntStream.range(0, onnxOp.onnxOutputs().size()).mapToObj(o -> indexer.getName(onnxOp.result(), o)).toList(),
                                    onnxOp.onnxAttributes()));
                        case CoreOp.ReturnOp _ -> { // skip
                        }
                        case CoreOp.TupleLoadOp tlo ->
                            indexer.put(tlo.result(), indexer.getName(tlo.operands().getFirst(), tlo.index()));
                        default ->
                            throw new UnsupportedOperationException(op.toText());
                    }
                }).toList(),
                List.of(indexer.getName(model.body().entryBlock().terminatingOp().operands().getFirst())));
    }

    static byte[] build(List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return new ModelProto()
                .ir_version(IR_VERSION)
                .graph(graph(inputs, ops, outputNames))
                .opset_import(new OperatorSetIdProto().version(OPSET_VERSION))
                .buf.toByteArray();
    }

    static GraphProto graph(List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return new GraphProto()
                .forEach(inputs, (g, i) -> g.input(i))
                .forEach(ops, (g, op) -> g.node(op))
                .forEach(outputNames, (g, oName) -> g.output(new ValueInfoProto().name(oName)));
    }

    static NodeProto node(String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        return new NodeProto()
                .forEach(inputNames, (n, iName) -> n.input(iName))
                .forEach(outputNames, (n, oName) -> n.output(oName))
                .op_type(opName)
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())));
    }

    static ValueInfoProto valueInfo(String name, int tensorElementType) {
        return new ValueInfoProto()
                .name(name)
                .type(new TypeProto()
                        .tensor_type(new Tensor()
                                .elem_type(tensorElementType)));
    }

    static Attribute attribute(String name, Object value) {
        var attr = new Attribute().name(name);
        switch (value) {
            case Float f -> {
                attr.type(1).f(f);
            }
            case Long l -> {
                attr.type(2).i(l);
            }
            case GraphProto g -> {
                attr.type(5).g(g.name(name));
            }
            case float[] floats -> {
                attr.type(6);
                for (float f : floats) attr.floats(f);
            }
            case long[] longs -> {
                attr.type(7);
                for (long l : longs) attr.ints(l);
            }
            default -> {
                throw new UnsupportedOperationException(value.getClass().toString()); // @@@ ToDo
            }
        }
        return attr;
    }
}
