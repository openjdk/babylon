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

import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
import oracle.code.onnx.proto.OnnxBuilder.*;
import oracle.code.onnx.proto.OnnxConstants.*;
import oracle.code.onnx.proto.OnnxModel;

public final class OnnxProtoBuilder {

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

    public static byte[] buildModel(String domain, CoreOp.ModuleOp module, List<oracle.code.onnx.Tensor> initializers) {
        var indexer = new Indexer(OpWriter.computeGlobalNames(module));

        var functions = new ArrayList<>(module.functionTable().sequencedValues());
        var imports = new ArrayList<String>();
        if (functions.size() > 1) imports.add(domain); // self domain import if additional functions
        for (var f : functions) {
            for (var op : f.body().entryBlock().ops()) { // auto import of op domains
                if (op instanceof OnnxOp) {
                    int di = op.opName().lastIndexOf('.');
                    if (di > 0) {
                        String dn = op.opName().substring(0, di);
                        if (!imports.contains(dn)) imports.add(dn);
                    }
                }
            }
        }
        var mainFunc = functions.removeLast();
        var mainBlock = mainFunc.body().entryBlock();

        var model = buildModel(
                graph(domain, mainFunc.funcName(), indexer, mainBlock, initializers, 0),
                imports,
                functions.stream().map(f ->
                        function(domain, f.funcName(),
                                 f.parameters().stream().map(indexer::nameOf).toList(),
                                 expandTuples(indexer, f.body().entryBlock().terminatingOp().operands()),
                                 nodes(domain, indexer, f.body().entryBlock().ops()))).toList());

//        System.out.println(OnnxModel.readFrom(model).toText());
        return model;
    }

    // @@@ unchecked constraints:
    //         tensor FuncOp parameters and single tensor return type
    //         OnnxOps (with tensor operands and single tensor return value) and ReturnOp (returning single tensor)
    //         entry block only
    static byte[] buildModel(Block block, List<oracle.code.onnx.Tensor> initializers) {
        var indexer = new Indexer(OpWriter.computeGlobalNames(block.parentBody().parentOp()));
        var model = buildModel(graph(null, null, indexer, block, initializers, 0), List.of(), List.of());
//        System.out.println(OnnxModel.readFrom(model).toText());
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
                .forEach(functions, (m, f) -> m.functions(f))
                .graph(graph)
                .getBytes();
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

    static GraphProto graph(String domain, String graphName, Indexer indexer, Block block, List<oracle.code.onnx.Tensor> initializers, int scalarArgs) {
        var params = block.parameters();
        params.forEach(indexer::nameOf);
        int firstInitializer = params.size() - initializers.size();
        var args = params.subList(0, firstInitializer);
        return graph(graphName,
                IntStream.range(0, initializers.size()).mapToObj(i -> tensorProto(indexer.nameOf(params.get(i + firstInitializer)), initializers.get(i))).toList(),
                tensorInfos(indexer, args, scalarArgs),
                nodes(domain, indexer, block.ops()),
                expandTuples(indexer, block.terminatingOp().operands()));
    }

    static List<NodeProto> nodes(String domain, Indexer indexer, List<Op> ops) {
        return ops.stream().<NodeProto>mapMulti((op, opNodes) -> {
            switch (op) {
                case OnnxOps.If ifOp ->
                    opNodes.accept(node(
                            ifOp.opName(),
                            List.of(indexer.nameOf(ifOp.operands().getFirst())),
                            IntStream.range(0, ifOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(ifOp.result(), o)).toList(),
                            java.util.Map.of(
                                    "then_branch", graph(domain, null, indexer, ifOp.thenBranch().entryBlock(), List.of(), 0),
                                    "else_branch", graph(domain, null, indexer, ifOp.elseBranch().entryBlock(), List.of(), 0))));
                case OnnxOps.Loop loopOp -> {
                    opNodes.accept(node(loopOp.opName(),
                            expandTuples(indexer, loopOp.operands()),
                            IntStream.range(0, loopOp.resultType() instanceof TupleType tt ? tt.componentTypes().size() : 1).mapToObj(o -> indexer.nameOf(loopOp.result(), o)).toList(),
                            java.util.Map.of(
                                    "body", graph(domain, null, indexer, loopOp.loopBody().entryBlock(), List.of(), 2))));
                }
                case OnnxOp onnxOp ->
                    opNodes.accept(node(
                            onnxOp.opName(),
                            onnxOp.operands().stream().map(indexer::nameOf).toList(),
                            IntStream.range(0, onnxOp.onnxOutputs().size()).mapToObj(o -> indexer.nameOf(onnxOp.result(), o)).toList(),
                            onnxOp.onnxAttributes()));
                case CoreOp.FuncCallOp fco ->
                    opNodes.accept(node(
                            domain,
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

    static FunctionProto function(String functionDomain, String functionName, List<String> inputNames, List<String> outputNames, List<NodeProto> ops) {
        int di = functionName.lastIndexOf('.');
        return new FunctionProto()
                .domain(functionDomain)
                .name(functionName)
                .forEach(inputNames, (f, i) -> f.input(i))
                .forEach(ops, (g, op) -> g.node(op))
                .forEach(outputNames, (f, o) -> f.output(o))
                .opsetImport(new OperatorSetIdProto().version(OPSET_VERSION));
    }

    static NodeProto node(String domain, String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        return new NodeProto()
                .domain(domain)
                .opType(opName)
                .forEach(inputNames, (n, iName) -> n.input(iName))
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())))
                .forEach(outputNames, (n, oName) -> n.output(oName));
    }

    static NodeProto node(String opName, List<String> inputNames, List<String> outputNames, java.util.Map<String, Object> attributes) {
        int di = opName.lastIndexOf('.');
        return new NodeProto()
                .domain(di < 0 ? null : opName.substring(0, di))
                .opType(opName.substring(di + 1))
                .forEach(inputNames, (n, iName) -> n.input(iName))
                .forEach(attributes.entrySet(), (n, ae) -> n.attribute(attribute(ae.getKey(), ae.getValue())))
                .forEach(outputNames, (n, oName) -> n.output(oName));
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType) {
        return tensorInfo(name, tensorElementType, false);
    }

    static ValueInfoProto tensorInfo(String name, int tensorElementType, boolean addScalarShape) {
        var t = new TypeProto.Tensor().elemType(tensorElementType);
        if (addScalarShape) t.shape(new TensorShapeProto());
        return new ValueInfoProto()
                .name(name)
                .type(new TypeProto().tensorType(t));
    }

    static TensorProto tensorProto(String name, oracle.code.onnx.Tensor tensor) {
        return new TensorProto()
                .name(name)
                .dataType(tensor.elementType().id)
                .dims(tensor.shape())
                .rawData(tensor.data().toArray(ValueLayout.JAVA_BYTE));
    }

    static TensorProto tensorProto(oracle.code.onnx.Tensor tensor) {
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
