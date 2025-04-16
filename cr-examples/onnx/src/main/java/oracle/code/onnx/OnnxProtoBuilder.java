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

import java.io.ByteArrayOutputStream;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.IntStream;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeItem;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.TupleType;
import jdk.incubator.code.writer.OpWriter;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
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
        Attribute floats(float... floats) {return _f(7, floats);}
        Attribute ints(long... ints) {return _f(8, ints);}
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
        TensorProto dims(long... dims) {return _f(1, dims);}
        TensorProto data_type(int data_type) {return _f(2, data_type);}
        TensorProto segment(Segment segment) {return _f(3, segment);}
        TensorProto float_data(float... float_data) {return _f(4, float_data);}
        TensorProto int32_data(int... int32_data) {return _f(5, int32_data);}
        TensorProto string_data(byte[] string_data) {return _f(6, string_data);}
        TensorProto int64_data(long... int64_data) {return _f(7, int64_data);}
        TensorProto name(String name) {return _f(8, name);}
        TensorProto doc_string(String doc_string) {return _f(12, doc_string);}
        TensorProto raw_data(byte[] raw_data) {return _f(9, raw_data);}
        TensorProto external_data(StringStringEntryProto external_data) {return _f(13, external_data);}
        TensorProto data_location(int data_location) {return _f(14, data_location);}
        TensorProto double_data(double... double_data) {return _f(10, double_data);}
        TensorProto uint64_data(long... uint64_data) {return _f(11, uint64_data);}
        TensorProto metadata_props(StringStringEntryProto metadata_props) {return _f(16, metadata_props);}
    }

    static final class Segment extends OnnxProtoBuilder<Segment> {
        Segment begin(long begin) {return _f(1, begin);}
        Segment end(long end) {return _f(2, end);}
    }

    static final class SparseTensorProto extends OnnxProtoBuilder<SparseTensorProto> {
        SparseTensorProto values(TensorProto values) {return _f(1, values);}
        SparseTensorProto indices(TensorProto indices) {return _f(2, indices);}
        SparseTensorProto dims(long... dims) {return _f(3, dims);}
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

    void _encode(float value) {
        int bits =  Float.floatToRawIntBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
    }

    void _encode(double value) {
        long bits =  Double.doubleToRawLongBits(value);
        buf.write((byte)bits);
        buf.write((byte)(bits >> 8));
        buf.write((byte)(bits >> 16));
        buf.write((byte)(bits >> 24));
        buf.write((byte)(bits >> 32));
        buf.write((byte)(bits >> 40));
        buf.write((byte)(bits >> 48));
        buf.write((byte)(bits >> 56));
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, String value) {
        return value == null ? (T)this : _f(fieldIndex, value.getBytes(StandardCharsets.UTF_8));
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
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, float... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxProtoBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, double value) {
        _encode(fieldIndex << 3 | 1);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, double... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxProtoBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, long value) {
        _encode(fieldIndex << 3);
        _encode(value);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, long... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxProtoBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")
    T _f(int fieldIndex, int... values) {
        if (values.length == 1) {
            return _f(fieldIndex, values[0]);
        }
        var b = new OnnxProtoBuilder();
        for (var v : values) b._encode(v);
        _f(fieldIndex, b);
        return (T)this;
    }

    @SuppressWarnings("unchecked")   T _f(int fieldIndex, OnnxProtoBuilder value) {
        return _f(fieldIndex, value.buf.toByteArray());
    }

    @SuppressWarnings("unchecked")
    <P> T forEach(Iterable<P> sup, BiConsumer<T, ? super P> cons) {
        sup.forEach(p -> cons.accept((T)this, p));
        return (T)this;
    }

    static final int IR_VERSION = 10;
    static final int OPSET_VERSION = 21;

    private static final class Indexer {

        private final Function<CodeItem, String> baseNames;
        private final HashMap<String, String> elementsMap;


        Indexer(Function<CodeItem, String> baseNames) {
            this.baseNames = baseNames;
            this.elementsMap = new HashMap<>();
        }

        private String baseName(Value value, int elementIndex) {
            var name = "%" + baseNames.apply(value);
            return elementIndex > 0 ? name + '.' + elementIndex : name;
        }

        String nameOf(Value value) {
            return nameOf(value, 0);
        }

        String nameOf(Value tuple, int elementIndex) {
            var name = baseName(tuple, elementIndex);
            return elementsMap.getOrDefault(name, name);
        }

        void mapTupleLoad(Value tupleLoadResult, Value tuple, int elementIndex) {
            elementsMap.put(baseName(tupleLoadResult, 0), nameOf(tuple, elementIndex));
        }

        void mapTupleElements(Value tuple, List<Value> elements) {
            for (int i = 0; i < elements.size(); i++) {
                elementsMap.put(baseName(tuple, i), nameOf(elements.get(i)));
            }
        }
    }

    static byte[] build(String domainName, CoreOp.ModuleOp module, List<oracle.code.onnx.Tensor> initializers) {
        var indexer = new Indexer(OpWriter.computeGlobalNames(module));

        var functions = new ArrayList<>(module.functionTable().sequencedValues());
        var mainFunc = functions.removeLast();
        var mainBlock = mainFunc.body().entryBlock();

        var model = build(
                graph(mainFunc.funcName(), domainName, indexer, mainBlock, initializers, 0),
                List.of(domainName),
                functions.stream().map(f ->
                        function(domainName,
                                 f.funcName(),
                                 f.parameters().stream().map(indexer::nameOf).toList(),
                                 expandTuples(indexer, f.body().entryBlock().terminatingOp().operands()),
                                 nodes(domainName, indexer, f.body().entryBlock().ops()))).toList());

//        OnnxProtoPrinter.printModel(model);
        return model;
    }

    // @@@ unchecked constraints:
    //         tensor FuncOp parameters and single tensor return type
    //         OnnxOps (with tensor operands and single tensor return value) and ReturnOp (returning single tensor)
    //         entry block only
    static byte[] build(Block block, List<oracle.code.onnx.Tensor> initializers) {
        var indexer = new Indexer(OpWriter.computeGlobalNames(block.parentBody().parentOp()));
        var model = build(graph(null, null, indexer, block, initializers, 0), List.of(), List.of());
//        OnnxProtoPrinter.printModel(model);
        return model;
    }

    static byte[] build(List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return build(graph(null, initializers, inputs, ops, outputNames), List.of(), List.of());
    }

    static byte[] build(List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames, List<String> customImportDomains, List<FunctionProto> functions) {
        return build(graph(null, initializers, inputs, ops, outputNames), customImportDomains, functions);
    }

    static byte[] build(GraphProto graph, List<String> customImportDomains, List<FunctionProto> functions) {
        return new ModelProto()
                .ir_version(IR_VERSION)
                .opset_import(new OperatorSetIdProto().version(OPSET_VERSION))
                .forEach(customImportDomains, (m, d) -> m.opset_import(new OperatorSetIdProto().domain(d)))
                .forEach(functions, (m, f) -> m.functions(f))
                .graph(graph)
                .buf.toByteArray();
    }

    static List<String> expandTuples(Indexer indexer, List<Value> values) {
        var names = new ArrayList<String>();
        expandTuples(indexer, names, values);
        return names;
    }

    static void expandTuples(Indexer indexer, List<String> names, List<Value> values) {
        for (var v : values) {
            if (v instanceof Op.Result or && or.op() instanceof CoreOp.TupleOp op) {
                expandTuples(indexer, names, op.operands());
            } else if (v.type() instanceof TupleType tt) {
                var ct = tt.componentTypes();
                for (int i = 0; i < ct.size(); i++) {
                    names.add(indexer.nameOf(v, i));
                }
            } else {
                names.add(indexer.nameOf(v));
            }
        }
    }

    static GraphProto graph(String graphName, String domainName, Indexer indexer, Block block, List<oracle.code.onnx.Tensor> initializers, int scalarArgs) {
        var params = block.parameters();
        params.forEach(indexer::nameOf);
        int firstInitializer = params.size() - initializers.size();
        var args = params.subList(0, firstInitializer);
        return graph(graphName,
                IntStream.range(0, initializers.size()).mapToObj(i -> tensorProto(indexer.nameOf(params.get(i + firstInitializer)), initializers.get(i))).toList(),
                tensorInfos(indexer, args, scalarArgs),
                nodes(domainName, indexer, block.ops()),
                expandTuples(indexer, block.terminatingOp().operands()));
    }

    static List<NodeProto> nodes(String domainName, Indexer indexer, List<Op> ops) {
        return ops.stream().<NodeProto>mapMulti((op, opNodes) -> {
            switch (op) {
                case OnnxOps.If ifOp ->
                    opNodes.accept(node(
                            ifOp.opName(),
                            List.of(indexer.nameOf(ifOp.operands().getFirst())),
                            IntStream.range(0, ifOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(ifOp.result(), o)).toList(),
                            java.util.Map.of(
                                    "then_branch", graph(null, domainName, indexer, ifOp.thenBranch().entryBlock(), List.of(), 0),
                                    "else_branch", graph(null, domainName, indexer, ifOp.elseBranch().entryBlock(), List.of(), 0))));
                case OnnxOps.Loop loopOp -> {
                    opNodes.accept(node(loopOp.opName(),
                            expandTuples(indexer, loopOp.operands()),
                            IntStream.range(0, loopOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(loopOp.result(), o)).toList(),
                            java.util.Map.of(
                                    "body", graph(null, domainName, indexer, loopOp.loopBody().entryBlock(), List.of(), 2))));
                }
                case OnnxOp onnxOp ->
                    opNodes.accept(node(
                            onnxOp.opName(),
                            onnxOp.operands().stream().map(indexer::nameOf).toList(),
                            IntStream.range(0, onnxOp.onnxOutputs().size()).mapToObj(o -> indexer.nameOf(onnxOp.result(), o)).toList(),
                            onnxOp.onnxAttributes()));
                case CoreOp.FuncCallOp fco ->
                    opNodes.accept(node(
                            domainName,
                            fco.funcName(),
                            fco.operands().stream().map(indexer::nameOf).toList(),
                            expandTuples(indexer, List.of(fco.result())),
                            java.util.Map.of()));
                case CoreOp.ReturnOp _, CoreOp.ConstantOp _ -> { // skip
                }
                case CoreOp.TupleLoadOp tlo ->
                    indexer.mapTupleLoad(tlo.result(), tlo.operands().getFirst(), tlo.index());
                case CoreOp.TupleOp to ->
                    indexer.mapTupleElements(to.result(), to.operands());
                case CoreOp.InvokeOp io when io.invokeDescriptor().refType().equals(JavaType.type(List.class)) -> {
                    if (io.invokeDescriptor().name().equals("get") && io.operands().getLast() instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp co && co.value() instanceof Integer i) {
                        indexer.mapTupleLoad(io.result(), io.operands().getFirst(), i);
                    } else if (io.invokeDescriptor().name().equals("of")) {
                        indexer.mapTupleElements(io.result(), io.operands());
                    } else {
                        throw new UnsupportedOperationException(op.toText());
                    }
                }
                default -> {
                    throw new UnsupportedOperationException(op.toText());
                }
            }
        }).toList();
    }

    static List<ValueInfoProto> tensorInfos(Indexer indexer, List<Block.Parameter> args, int scalarArgs) {
        var infos = new ArrayList<ValueInfoProto>();
        for (var arg : args) {
            switch (arg.type()) {
                case OnnxType.TensorType tt ->
                    infos.add(tensorInfo(indexer.nameOf(arg), tt.eType().id(), infos.size() < scalarArgs));
                case TupleType tt -> {
                    var ct = tt.componentTypes();
                    for (int i = 0; i < ct.size(); i++) {
                        infos.add(tensorInfo(indexer.nameOf(arg, i), ((OnnxType.TensorType)ct.get(i)).eType().id(), infos.size() < scalarArgs));
                    }
                }
                default ->
                    throw new UnsupportedOperationException(arg.type().toString());
            }
        }
        return infos;
    }

    static GraphProto graph(String name, List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return new GraphProto()
                .name(name)
                .forEach(initializers, (g, i) -> g.initializer(i))
                .forEach(inputs, (g, i) -> g.input(i))
                .forEach(ops, (g, op) -> g.node(op))
                .forEach(outputNames, (g, oName) -> g.output(new ValueInfoProto().name(oName)));
    }

    static FunctionProto function(String domain, String functionName, List<String> inputNames, List<String> outputNames, List<NodeProto> ops) {
        return new FunctionProto()
                .domain(domain)
                .name(functionName)
                .forEach(inputNames, (f, i) -> f.input(i))
                .forEach(ops, (g, op) -> g.node(op))
                .forEach(outputNames, (f, o) -> f.output(o))
                .opset_import(new OperatorSetIdProto().version(OPSET_VERSION));
    }

    static NodeProto node(String domain, String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        return new NodeProto()
                .domain(domain)
                .op_type(opName)
                .forEach(inputNames, (n, iName) -> n.input(iName))
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())))
                .forEach(outputNames, (n, oName) -> n.output(oName));
    }

    static NodeProto node(String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        return new NodeProto()
                .op_type(opName)
                .forEach(inputNames, (n, iName) -> n.input(iName))
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())))
                .forEach(outputNames, (n, oName) -> n.output(oName));
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType) {
        return tensorInfo(name, tensorElementType, false);
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType, boolean addScalarShape) {
        var t = new Tensor().elem_type(tensorElementType);
        if (addScalarShape) t.shape(new TensorShapeProto());
        return new ValueInfoProto()
                .name(name)
                .type(new TypeProto().tensor_type(t));
    }

    static TensorProto tensorProto(String name, oracle.code.onnx.Tensor tensor) {
        return new TensorProto()
                .name(name)
                .data_type(tensor.elementType().id)
                .dims(tensor.shape())
                .raw_data(tensor.data().toArray(ValueLayout.JAVA_BYTE));
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
                attr.floats(floats);
            }
            case long[] longs -> {
                attr.type(7);
                attr.ints(longs);
            }
            default -> {
                throw new UnsupportedOperationException(value.getClass().toString()); // @@@ ToDo
            }
        }
        return attr;
    }
}
