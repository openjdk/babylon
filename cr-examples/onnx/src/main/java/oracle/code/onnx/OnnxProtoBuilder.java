/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.foreign.ValueLayout;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeItem;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.OpWriter;
import oracle.code.onnx.compiler.OnnxTransformer.OnnxValueInfo;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;
import oracle.code.onnx.proto.OnnxBuilder.*;
import oracle.code.onnx.proto.OnnxConstants.*;

public final class OnnxProtoBuilder {

    static final int IR_VERSION = 10;
    static final int OPSET_VERSION = 21;

    private static final class Indexer {

        private final Function<CodeItem, String> baseNames;
        private final HashMap<String, String> remap;


        Indexer(Op root, Map<Value, String> explicitNames) {
            this.baseNames = OpWriter.computeGlobalNames(root);
            this.remap = new HashMap<>();
            explicitNames.forEach(this::setName);
        }

        void setName(Value val, String name) {
            switch (val) {
                case Op.Result or when or.op() instanceof CoreOp.TupleOp to -> {
                    remap.put(baseName(val), name);
                    for (int i = 0; i < to.operands().size(); i++) {
                        setName(to.operands().get(i), name + "." + i);
                    }
                }
                case Block.Parameter _ when val.type() instanceof TupleType tt -> {
                    for (int i = 0; i < tt.componentTypes().size(); i++) {
                        remap.put(baseName(val, i), name + "." + i);
                    }
                }
                default -> {
                    remap.put(baseName(val), name);
                    if (val instanceof Op.Result or && or.op() instanceof CoreOp.TupleLoadOp tlo) {
                        Value tr = tlo.operands().getFirst();
                        remap.put(baseName(tr, tlo.index()), name);
                        if (tr instanceof Op.Result tor && tor.op() instanceof CoreOp.TupleOp to) {
                            setName(to.operands().get(tlo.index()), name);
                        }
                    }
                }
            }
        }

        private String baseName(Value value) {
            return "%" + baseNames.apply(value);
        }

        private String baseName(Value value, int elementIndex) {
            var name = baseName(value);
            return elementIndex > 0 ? name + '.' + elementIndex : name;
        }

        String nameOf(Value value) {
            var name = baseName(value);
            return remap.getOrDefault(name, name);
        }

        String nameOf(Value tuple, int elementIndex) {
            var name = baseName(tuple, elementIndex);
            return remap.getOrDefault(name, name);
        }

        void mapTupleLoad(Value tupleLoadResult, Value tuple, int elementIndex) {
            remap.putIfAbsent(baseName(tupleLoadResult), nameOf(tuple, elementIndex));
        }

        void mapTupleElements(Value tuple, List<Value> elements) {
            for (int i = 0; i < elements.size(); i++) {
                remap.putIfAbsent(baseName(tuple, i), nameOf(elements.get(i)));
            }
        }
    }

    public record ExternalTensorDataInfo(String location, long offset, long length) {
    }

    public static byte[] buildModel(String domain, CoreOp.ModuleOp module, List<Object> initializers, Map<Value, OnnxValueInfo> valueInfo, Function<Tensor, ExternalTensorDataInfo> tensorDataExternalizer) {
        var explicitValueNames = new HashMap<Value, String>();
        valueInfo.forEach((value, info) -> {
            if (info.name() != null) {
                explicitValueNames.put(value, info.name());
            }
        });
        var indexer = new Indexer(module, explicitValueNames);
        var valueShapes = new HashMap<String, long[]>();
        valueInfo.forEach((value, info) -> {
            if (info.shape() != null) {
                valueShapes.put(info.name() != null ? info.name() : indexer.nameOf(value), info.shape());
            }
        });

        var functions = new ArrayList<>(module.functionTable().sequencedValues());
        var imports = new ArrayList<String>();
        if (functions.size() > 1) imports.add(domain); // self domain import if additional functions
        for (var f : functions) {
            for (var op : f.body().entryBlock().ops()) { // auto import of op domains
                if (op instanceof OnnxOp oop) {
                    String name = oop.schema().name();
                    int di = name.lastIndexOf('.');
                    if (di > 0) {
                        String dn = name.substring(0, di);
                        if (!imports.contains(dn)) imports.add(dn);
                    }
                }
            }
        }
        var mainFunc = functions.removeLast();
        var mainBlock = mainFunc.body().entryBlock();

        var model = buildModel(
                graph(domain, mainFunc.funcName(), indexer, mainBlock, initializers, 0, tensorDataExternalizer, valueShapes, valueInfo),
                imports,
                functions.stream().map(f ->
                        function(domain, imports, f.funcName(),
                                expandTuples(indexer, f.parameters()),
                                expandTuples(indexer, f.body().entryBlock().terminatingOp().operands()),
                                nodes(domain, indexer, f.body().entryBlock().ops(), valueShapes, Map.of()))).toList());
        return model;
    }

    // @@@ unchecked constraints:
    //         tensor FuncOp parameters and single tensor return type
    //         OnnxOps (with tensor operands and single tensor return value) and ReturnOp (returning single tensor)
    //         entry block only
    static byte[] buildModel(Block block, List<Tensor> initializers) {
        var indexer = new Indexer(block.ancestorOp(), Map.of());
        var model = buildModel(graph(null, null, indexer, block, initializers, 0), List.of(), List.of());
        return model;
    }

    static byte[] buildModel(List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return buildModel(graph(null, initializers, inputs, ops, outputNames), List.of(), List.of());
    }

    static byte[] buildModel(List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames, List<String> customImportDomains, List<FunctionProto> functions) {
        return buildModel(graph(null, initializers, inputs, ops, outputNames), customImportDomains, functions);
    }

    static byte[] buildModel(GraphProto graph, List<String> imports, List<FunctionProto> functions) {
        return new ModelProto()
                .irVersion(IR_VERSION)
                .opsetImport(new OperatorSetIdProto().version(OPSET_VERSION))
                .forEach(imports, (m, d) -> m.opsetImport(new OperatorSetIdProto().domain(d).version(1)))
                .forEach(functions, ModelProto::functions)
                .graph(graph)
                .getBytes();
    }

    static List<String> expandTuples(Indexer indexer, List<? extends Value> values) {
        var names = new ArrayList<String>();
        expandTuples(indexer, names, values);
        return names;
    }

    static void expandTuples(Indexer indexer, List<String> names, List<? extends Value> values) {
        for (var v : values) {
            if (v instanceof Op.Result or && or.op() instanceof CoreOp.TupleOp op) {
                expandTuples(indexer, names, op.operands());
            } else if (v instanceof Op.Result or && or.op() instanceof CoreOp.TupleLoadOp op) {
                names.add(indexer.nameOf(op.operands().getFirst(), op.index()));
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

    static List<ValueInfoProto> tensorInfos(Indexer indexer, List<? extends Value> values, Map<String, long[]> valueShapes) {
        var infos = new ArrayList<ValueInfoProto>();
        tensorInfos(indexer, infos, values, valueShapes);
        return infos;
    }

    static void tensorInfos(Indexer indexer, List<ValueInfoProto> infos, List<? extends Value> values, Map<String, long[]> valueShapes) {
        for (var v : values) {
            if (v instanceof Op.Result or && or.op() instanceof CoreOp.TupleOp op) {
                tensorInfos(indexer, infos, op.operands(), valueShapes);
            } else if (v instanceof Op.Result or && or.op() instanceof CoreOp.TupleLoadOp op) {
                infos.add(tensorInfo(indexer.nameOf(op.operands().getFirst(), op.index()), v.type(), valueShapes));
            } else if (v.type() instanceof TupleType tt) {
                var ct = tt.componentTypes();
                for (int i = 0; i < ct.size(); i++) {
                    infos.add(tensorInfo(indexer.nameOf(v, i), ct.get(i), valueShapes));
                }
            } else {
                infos.add(tensorInfo(indexer.nameOf(v), v.type(), valueShapes));
            }
        }
    }

    static GraphProto graph(String domain, String graphName, Indexer indexer, Block block, List<? extends Object> initializers, int scalarArgs) {
        return graph(domain, graphName, indexer, block, initializers, scalarArgs, _ -> null, Map.of(), Map.of());
    }

    static GraphProto graph(String domain, String graphName, Indexer indexer, Block block, List<? extends Object> initializers, int scalarArgs, Function<Tensor, ExternalTensorDataInfo> tensorDataExternalizer, Map<String, long[]> valueShapes, Map<Value, OnnxValueInfo> valueInfo) {
        valueShapes = new HashMap<>(valueShapes);
        var params = block.parameters();
        params.forEach(indexer::nameOf);
        int firstInitializer = params.size() - initializers.size();
        var args = params.subList(0, firstInitializer);
        List<TensorProto> graphInitializers = IntStream.range(0, initializers.size()).boxed().<TensorProto>mapMulti((i, tps) -> {
            Object val = initializers.get(i);
            if (val instanceof Record) {
                var rcs = val.getClass().getRecordComponents();
                for (int rci = 0; rci < rcs.length; rci++)
                    try {
                        tps.accept(tensorProto(indexer.nameOf(params.get(i + firstInitializer), rci), (Tensor) (rcs[rci].getAccessor().invoke(val)), tensorDataExternalizer));
                    } catch (ReflectiveOperationException e) {
                        throw new IllegalArgumentException(e);
                    }
            } else if (val instanceof Tensor[] tarr) {
                for (int tai = 0; tai < tarr.length; tai++) {
                    tps.accept(tensorProto(indexer.nameOf(params.get(i + firstInitializer), tai), tarr[tai], tensorDataExternalizer));
                }
            } else {
                tps.accept(tensorProto(indexer.nameOf(params.get(i + firstInitializer)), (Tensor) val, tensorDataExternalizer));
            }
        }).toList();
        List<ValueInfoProto> graphInputs = tensorInfos(indexer, args, scalarArgs, valueShapes);
        List<NodeProto> graphNodes = nodes(domain, indexer, block.ops(), valueShapes, valueInfo);
        List<ValueInfoProto> graphOutputs = tensorInfos(indexer, block.terminatingOp().operands(), valueShapes);
        var boundaryNames = new HashSet<String>();
        boundaryNames.addAll(expandTuples(indexer, args));
        boundaryNames.addAll(expandTuples(indexer, block.terminatingOp().operands()));
        var blockValues = blockValues(block);
        var graphValueInfoNames = new HashSet<String>();
        List<ValueInfoProto> graphValueInfos = valueInfo.entrySet().stream()
                .filter(entry -> entry.getValue().shape() != null)
                .filter(entry -> blockValues.contains(entry.getKey()))
                .<ValueInfoProto>mapMulti((entry, infos) -> {
                    String name = entry.getValue().name() != null ? entry.getValue().name() : indexer.nameOf(entry.getKey());
                    if (!boundaryNames.contains(name) && graphValueInfoNames.add(name)) {
                        infos.accept(tensorInfo(name, OnnxType.INT64.id(), shapeOf(entry.getValue().shape())));
                    }
                })
                .toList();
        return graphWithOutputs(graphName, graphInitializers, graphInputs, graphNodes, graphValueInfos, graphOutputs);
    }

    static Set<Value> blockValues(Block block) {
        var values = new HashSet<Value>();
        values.addAll(block.parameters());
        block.ops().forEach(op -> {
            if (op.result() != null) {
                values.add(op.result());
            }
        });
        return values;
    }

    static List<String> opInputNames(Indexer indexer, SequencedMap<OnnxOp.OnnxParameter, Object> inputs) {
        List<String> inputNames = inputs.sequencedValues().stream()
                .<String>mapMulti((v, dump) -> {
                    switch (v) {
                        case Value val -> dump.accept(indexer.nameOf(val));
                        case Optional<?> o when o.isPresent() && o.get() instanceof Value val ->
                                dump.accept(indexer.nameOf(val));
                        case List l -> l.forEach(val -> dump.accept(indexer.nameOf((Value) val)));
                        default -> dump.accept(""); // empty names for unused optional inputs
                    }
                }).toList();
        return inputNames.reversed().stream().dropWhile(String::isEmpty).toList().reversed();
    }

    static List<NodeProto> nodes(String domain, Indexer indexer, List<Op> ops, Map<String, long[]> valueShapes, Map<Value, OnnxValueInfo> valueInfo) {
        return ops.stream().<NodeProto>mapMulti((op, opNodes) -> {
            switch (op) {
                case OnnxOps.If ifOp -> opNodes.accept(node(
                        ifOp.schema().name(),
                        List.of(indexer.nameOf(ifOp.operands().getFirst())),
                        IntStream.range(0, ifOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(ifOp.result(), o)).toList(),
                        java.util.Map.of(
                                "then_branch", graph(domain, null, indexer, ifOp.thenBranch().entryBlock(), List.of(), 0, _ -> null, valueShapes, Map.of()),
                                "else_branch", graph(domain, null, indexer, ifOp.elseBranch().entryBlock(), List.of(), 0, _ -> null, valueShapes, Map.of()))));
                case OnnxOps.Loop loopOp -> {
                    opNodes.accept(node(loopOp.schema().name(),
                            expandTuples(indexer, loopOp.operands()),
                            IntStream.range(0, loopOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(loopOp.result(), o)).toList(),
                            java.util.Map.of(
                                    "body", graph(domain, null, indexer, loopOp.loopBody().entryBlock(), List.of(), 2, _ -> null, valueShapes, Map.of()))));
                }
                case OnnxOp onnxOp -> opNodes.accept(node(
                        onnxOp.schema().name(),
                        opInputNames(indexer, onnxOp.onnxInputs()),
                        IntStream.range(0, onnxOp.onnxOutputs().size()).mapToObj(o -> indexer.nameOf(onnxOp.result(), o)).toList(),
                        onnxOp.onnxAttributes()));
                case CoreOp.FuncCallOp fco -> opNodes.accept(node(
                        domain,
                        fco.funcName(),
                        expandTuples(indexer, fco.operands()),
                        expandTuples(indexer, List.of(fco.result())),
                        java.util.Map.of()));
                case CoreOp.ReturnOp _, CoreOp.ConstantOp _ -> { // skip
                }
                case CoreOp.TupleLoadOp tlo ->
                        indexer.mapTupleLoad(tlo.result(), tlo.operands().getFirst(), tlo.index());
                case CoreOp.TupleOp to -> indexer.mapTupleElements(to.result(), to.operands());
                case JavaOp.InvokeOp io when io.invokeReference().refType().equals(JavaType.type(List.class)) -> {
                    if (io.invokeReference().name().equals("get") && io.operands().getLast() instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp co && co.value() instanceof Integer i) {
                        indexer.mapTupleLoad(io.result(), io.operands().getFirst(), i);
                    } else if (io.invokeReference().name().equals("of")) {
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

    static List<ValueInfoProto> tensorInfos(Indexer indexer, List<Block.Parameter> args, int scalarArgs, Map<String, long[]> valueShapes) {
        var infos = new ArrayList<ValueInfoProto>();
        for (var arg : args) {
            switch (arg.type()) {
                case OnnxType.TensorType tt -> infos.add(infos.size() < scalarArgs
                        ? tensorInfo(indexer.nameOf(arg), tt.eType().id(), true, valueShapes)
                        : tensorInfo(indexer.nameOf(arg), tt, valueShapes));
                case TupleType tt -> {
                    var ct = tt.componentTypes();
                    for (int i = 0; i < ct.size(); i++) {
                        infos.add(infos.size() < scalarArgs && ct.get(i) instanceof OnnxType.TensorType tensorType
                                ? tensorInfo(indexer.nameOf(arg, i), tensorType.eType().id(), true, valueShapes)
                                : tensorInfo(indexer.nameOf(arg, i), ct.get(i), valueShapes));
                    }
                }
                default -> throw new UnsupportedOperationException(arg.type().toString());
            }
        }
        return infos;
    }

    static GraphProto graph(String name, List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<String> outputNames) {
        return graphWithOutputs(name, initializers, inputs, ops, outputNames.stream().map(oName -> new ValueInfoProto().name(oName)).toList());
    }

    static GraphProto graphWithOutputs(String name, List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<ValueInfoProto> outputs) {
        return graphWithOutputs(name, initializers, inputs, ops, List.of(), outputs);
    }

    static GraphProto graphWithOutputs(String name, List<TensorProto> initializers, List<ValueInfoProto> inputs, List<NodeProto> ops, List<ValueInfoProto> valueInfos, List<ValueInfoProto> outputs) {
        return new GraphProto()
                .name(name)
                .forEach(initializers, GraphProto::initializer)
                .forEach(inputs, GraphProto::input)
                .forEach(ops, GraphProto::node)
                .forEach(valueInfos, GraphProto::valueInfo)
                .forEach(outputs, GraphProto::output);
    }

    static FunctionProto function(String functionDomain, List<String> imports, String functionName, List<String> inputNames, List<String> outputNames, List<NodeProto> ops) {
        return new FunctionProto()
                .domain(functionDomain)
                .name(functionName)
                .forEach(inputNames, FunctionProto::input)
                .forEach(ops, FunctionProto::node)
                .forEach(outputNames, FunctionProto::output)
                .opsetImport(new OperatorSetIdProto().version(OPSET_VERSION))
                .forEach(imports, (f, d) -> f.opsetImport(new OperatorSetIdProto().domain(d).version(1)));
    }

    static NodeProto node(String domain, String opName, List<String> inputNames, List<String> outputNames, Map<String, Object> attributes) {
        return new NodeProto()
                .domain(domain)
                .opType(opName)
                .forEach(inputNames, NodeProto::input)
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())))
                .forEach(outputNames, NodeProto::output);
    }

    static NodeProto node(String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        int di = opName.lastIndexOf('.');
        return node(di < 0 ? null : opName.substring(0, di), opName.substring(di + 1), inputNames, outputNames, attributes);
    }

    static List<Object> shapeOf(long[] shape) {
        if (shape == null) {
            return null;
        }
        var dims = new ArrayList<>(shape.length);
        for (long dim : shape) {
            dims.add(dim);
        }
        return dims;
    }

    static List<Object> shapeOf(String name, Map<String, long[]> valueShapes) {
        long[] shape = valueShapes.get(name);
        if (shape == null) {
            int dot = name.lastIndexOf('.');
            if (dot > 0) {
                try {
                    Integer.parseInt(name.substring(dot + 1));
                    shape = valueShapes.get(name.substring(0, dot));
                } catch (NumberFormatException _) {
                    // Not a tuple-expanded value name.
                }
            }
        }
        return shapeOf(shape);
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType) {
        return tensorInfo(name, tensorElementType, false, Map.of());
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType, boolean addScalarShape) {
        return tensorInfo(name, tensorElementType, addScalarShape, Map.of());
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType, boolean addScalarShape, Map<String, long[]> valueShapes) {
        return tensorInfo(name, tensorElementType, addScalarShape ? List.of() : shapeOf(name, valueShapes));
    }

    static ValueInfoProto tensorInfo(String name, Object type, Map<String, long[]> valueShapes) {
        return type instanceof OnnxType.TensorType tensorType
                ? tensorInfo(name, tensorType, valueShapes)
                : new ValueInfoProto().name(name);
    }

    static ValueInfoProto tensorInfo(String name, OnnxType.TensorType tensorType, Map<String, long[]> valueShapes) {
        List<Object> shape = tensorType.shape() != null && !tensorType.shape().isEmpty() ? tensorType.shape() : shapeOf(name, valueShapes);
        return tensorInfo(name, tensorType.eType().id(), shape);
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType, List<Object> shape) {
        var t = new TypeProto.Tensor().elemType(tensorElementType);
        if (shape != null) {
            var tensorShape = new TensorShapeProto();
            for (int i = 0; i < shape.size(); i++) {
                Object dim = shape.get(i);
                tensorShape.dim(switch (dim) {
                    case Number n when n.longValue() < 0 ->
                            new TensorShapeProto.Dimension().dimParam(name + "_dim_" + i);
                    case Number n -> new TensorShapeProto.Dimension().dimValue(n.longValue());
                    case String s -> new TensorShapeProto.Dimension().dimParam(s);
                    default -> throw new IllegalArgumentException("Unsupported tensor dimension: " + dim);
                });
            }
            t.shape(tensorShape);
        }
        return new ValueInfoProto()
                .name(name)
                .type(new TypeProto().tensorType(t));
    }

    static TensorProto tensorProto(String name, Tensor tensor, Function<Tensor, ExternalTensorDataInfo> tensorDataExternalizer) {
        ExternalTensorDataInfo extInfo = tensorDataExternalizer.apply(tensor);
        TensorProto tp = new TensorProto()
                .name(name)
                .dataType(tensor.elementType().id)
                .dims(tensor.shape());
        return extInfo == null
                ? tp.rawData(tensor.data().toArray(ValueLayout.JAVA_BYTE))
                : tp.externalData(new StringStringEntryProto().key("location").value(extInfo.location()))
                .externalData(new StringStringEntryProto().key("offset").value(String.valueOf(extInfo.offset())))
                .externalData(new StringStringEntryProto().key("length").value(String.valueOf(extInfo.length())))
                .dataLocation(DataLocation.EXTERNAL);
    }

    static TensorProto tensorProto(Tensor tensor) {
        return new TensorProto()
                .dataType(tensor.elementType().id)
                .dims(tensor.shape())
                .rawData(tensor.data().toArray(ValueLayout.JAVA_BYTE));
    }

    static AttributeProto attribute(String name, Object value) {
        var attr = new AttributeProto().name(name);
        switch (value) {
            case Float f -> {
                attr.type(AttributeType.FLOAT).f(f);
            }
            case Long l -> {
                attr.type(AttributeType.INT).i(l);
            }
            case GraphProto g -> {
                attr.type(AttributeType.GRAPH).g(g.name(name));
            }
            case float[] floats -> {
                attr.type(AttributeType.FLOATS);
                attr.floats(floats);
            }
            case long[] longs -> {
                attr.type(AttributeType.INTS);
                attr.ints(longs);
            }
            case String s -> {
                attr.type(AttributeType.STRING);
                attr.s(s.getBytes());
            }
            case Tensor<?> t -> {
                attr.type(AttributeType.TENSOR);
                attr.t(tensorProto(t));
            }
            default -> {
                throw new UnsupportedOperationException(value.getClass().toString()); // @@@ ToDo
            }
        }
        return attr;
    }
}
