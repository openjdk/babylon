/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

package oracle.code.onnx.lift;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeItem;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.extern.ExternalizedOp;
import jdk.incubator.code.extern.OpFactory;
import jdk.incubator.code.extern.OpWriter;
import oracle.code.onnx.OnnxOperators;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.ExplicitOnnxOps;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;
import oracle.code.onnx.ir.OpFactoryHelper;
import oracle.code.onnx.proto.OnnxModel;


/**
 * Lifts ONNX model binary to ONNX code reflection model, extracts weights, and generates Java model source.
 */
public class OnnxLift {

    record LiftedModelWrapper(CoreOp.FuncOp func, List<String> names, List<OnnxModel.TensorProto> weights) {

        private Function<CodeItem, String> namer() {
            var defNamer = OpWriter.CodeItemNamerOption.defaultValue().namer();
            var namer = new HashMap<Value, Integer>();
            return ci -> ci instanceof Value v ? names.get(namer.computeIfAbsent(v, _ -> namer.size())) : defNamer.apply(ci);
        }

        public String toText() {
            return OpWriter.toText(func, OpWriter.CodeItemNamerOption.of(namer()));
        }

        public String toJava() {
            var out = new StringBuilder();
            var namer = namer();
            var entryBlock = func.bodies().getFirst().entryBlock();
            entryBlock.parameters().forEach(namer::apply);
            out.append("""
                    import java.io.IOException;
                    import java.io.RandomAccessFile;
                    import java.lang.foreign.Arena;
                    import java.lang.foreign.MemorySegment;
                    import java.nio.channels.FileChannel;
                    import jdk.incubator.code.CodeReflection;
                    import oracle.code.onnx.Tensor;

                    import static java.util.Optional.*;
                    import static oracle.code.onnx.OnnxOperators.*;
                    import static oracle.code.onnx.Tensor.ElementType.*;

                    public class Model {

                        final Arena arena = Arena.ofAuto();

                        MemorySegment mmap(String pathname) {
                            try (var f = new RandomAccessFile(pathname, "r")) {
                                return f.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, f.length(), arena);
                            } catch (IOException e) {
                                throw new RuntimeException(e);
                            }
                        }

                        <T> Tensor<T> load(String path, Tensor.ElementType type, long... shape) {
                            return new Tensor<>(arena, mmap(path), type, shape);
                        }

                    """);
            int pSize = entryBlock.parameters().size();
            int realParamsSize = pSize - weights().size();
            var weightParams = entryBlock.parameters().subList(realParamsSize, pSize);
            var wMap = weights.stream().collect(Collectors.toUnmodifiableMap(OnnxModel.TensorProto::name, Function.identity()));
            for (int i = 0; i < weightParams.size(); i++) {
                Block.Parameter wp = weightParams.get(i);
                OnnxModel.TensorProto w = wMap.get(namer.apply(wp));
                String name = toJavaName(w.name());
                long[] dims = OnnxLift.joinLongArray(w.dims());
                out.append("    final " + toJavaType(wp.type()) + " " + name + " = load(\""
                        + name + "\", "
                        + Tensor.ElementType.fromOnnxId(w.dataType()).name()
                        + (dims.length > 0 ? (", " + longJoin(dims)) : "") + ");\n");
            }
            out.append("""

                        @CodeReflection
                        public Object mainGraph(
                    """);
            for (int i = 0; i < realParamsSize; i++) {
                if (i > 0) {
                    out.append(",\n");
                }
                var param = entryBlock.parameters().get(i);
                out.append("            ").append(toJavaType(param.type())).append(' ').append(toJavaName(namer.apply(param)));
            }
            out.append(") {\n");
            for (var op : entryBlock.ops()) {
                if (!(op instanceof CoreOp.TupleLoadOp)) { // lazy tupple loads
                    out.append("        ");
                    if (!op.resultType().equals(JavaType.VOID)) {
                        out.append(toJavaType(op.resultType())).append(' ').append(toJavaName(namer.apply(op.result()))).append(" = ");
                    }

                    switch (op) {
                        case OnnxOp oo -> {
                            String opName = op.externalizeOpName();
                            out.append(opName.substring(opName.lastIndexOf('.') + 1)).append('(');
                            var schema = getSchema(oo);
                            var inputs = oo.onnxInputs();
                            boolean first = true;
                            for (var oi : schema.inputs()) {
                                if (first) {
                                    first = false;
                                } else {
                                    out.append(", ");
                                }
                                out.append(toJava(namer, inputs.get(oi)));
                            }
                            var attrs = oo.onnxAttributes();
                            for (var oa : schema.attributes()) {
                                if (first) {
                                    first = false;
                                } else {
                                    out.append(", ");
                                }
                                var a = attrs.get(oa.name());
                                if (a == null) {
                                    out.append("empty()");
                                } else if (oa.isOptional()) {
                                    out.append("of(").append(toString(a)).append(')');
                                } else {
                                    out.append(toString(a));
                                }
                            }
                            out.append(");\n");
                        }
                        case CoreOp.TupleOp to -> {
                            out.append("List.of(");
                            boolean first = true;
                            for (var te : to.operands()) {
                                if (first) {
                                    first = false;
                                } else {
                                    out.append(", ");
                                }
                                out.append(toJava(namer, te));
                            }
                            out.append(");\n");
                        }
                        case CoreOp.ReturnOp ro -> {
                            out.append("return ").append(toJava(namer, ro.operands().getFirst())).append(";\n");
                        }
                        default -> throw new UnsupportedOperationException(op.toText());
                    }
                } else {
                    namer.apply(op.result());
                }
            }
            out.append("    }\n}\n");
            return out.toString();
        }

        private static String toString(Object o) {
            return switch (o) {
                case long[] la -> newArray(la);
                case float[] fa -> newArray(fa);
                case Long l -> l.toString() + "L";
                case Float f -> f.toString() + "F";
                case String s -> "\"" + s + "\"";
                case Tensor t when t.shape().length == 0 && t.elementType() == Tensor.ElementType.BOOL ->
                        "Tensor.ofScalar(" + t.data().get(ValueLayout.JAVA_BOOLEAN, 0) + ")";
                case Tensor t when t.shape().length == 0 && t.elementType() == Tensor.ElementType.INT8 ->
                        "Tensor.ofScalar((byte)" + t.data().get(ValueLayout.JAVA_BYTE, 0) + ")";
                case Tensor t -> "Tensor.ofShape(" + toString(t.shape()) + ", "
                    + switch (t.elementType()) {
                        case FLOAT -> toString(t.data().toArray(ValueLayout.JAVA_FLOAT));
                        case INT64 -> toString(t.data().toArray(ValueLayout.JAVA_LONG));
                        default -> "HexFormat.of().parseHex(\"" + HexFormat.of().formatHex(t.data().toArray(ValueLayout.JAVA_BYTE))
                                                            + "\"), " + t.elementType().name();
                    } + ")";
                default -> o.toString();
            };
        }

        private static String newArray(long[] la) {
            for (long l : la) {
                if (l != 0l) {
                    return "new long[] {" + longJoin(la) + "}";
                }
            }
            return "new long[" + la.length + "]";
        }

        private static String longJoin(long[] la) {
            return LongStream.of(la).mapToObj(d -> String.valueOf(d) + "L")
                                     .collect(Collectors.joining(", "));
        }

        private static String newArray(float[] fa) {
            for (float f : fa) {
                if (f != 0f) {
                    return IntStream.range(0, fa.length).mapToObj(i -> String.valueOf(fa[i]) + "F")
                                    .collect(Collectors.joining(", ",  "new float[] {", "}"));
                }
            }
            return "new float[" + fa.length + "]";
        }

        private static String tupleAccessor(Value tuple, int componentIndex) {
            if (tuple instanceof Op.Result or && or.op() instanceof OnnxOp oo) {
                String mName = oo.externalizeOpName();
                mName = mName.substring(mName.lastIndexOf('.') + 1);
                for (Method m : OnnxOperators.class.getMethods()) {
                    if (m.getName().equals(mName)) {
                        return m.getReturnType().getRecordComponents()[componentIndex].getAccessor().getName() + "()";
                    }
                }
                throw new IllegalStateException(mName);
            }
            return "get(" + componentIndex + ")"; // fallback to List
        }

        private static String toJava(Function<CodeItem, String> namer, Object value) {
            return switch (value) {
                case Optional o when o.isEmpty() -> "empty()";
                case Optional o -> "of(" + toJava(namer, o.get()) + ")";
                case List l -> "List.of(" + l.stream().map(le -> toJava(namer, le)).collect(Collectors.joining(", ")) + ")";
                case Op.Result or when or.op() instanceof CoreOp.TupleLoadOp tlo ->
                    toJavaName(namer.apply(tlo.operands().getFirst())) + '.' + tupleAccessor(tlo.operands().getFirst(), tlo.index());
                case Value v -> toJavaName(namer.apply(v));
                default -> throw new UnsupportedOperationException(value.toString());
            };
        }

        private static OnnxOp.OnnxSchema getSchema(OnnxOp oo) {
            try {
                return (OnnxOp.OnnxSchema) oo.getClass().getDeclaredField("SCHEMA").get(null);
            } catch (ReflectiveOperationException ex) {
                throw new RuntimeException(ex);
            }
        }

        private static String toJavaType(TypeElement t) {
            return switch (t) {
                case OnnxType.TensorType tt ->
                    "Tensor<" + switch (tt.eType()) {
                        case OnnxType.Float32Type _ -> "Float";
                        case OnnxType.Int64Type _ -> "Long";
                        case OnnxType.Int32Type _ -> "Integer";
                        case OnnxType.UInt8Type _ -> "Byte";
                        case OnnxType.BoolType _ -> "Boolean";
                        default -> throw new UnsupportedOperationException(t.toString());
                    } + ">";
                default -> "var";
            };
        }
    }

    static OnnxType toOnnxType(OnnxModel.TypeProto tp) {
        if (tp.tensorType() instanceof OnnxModel.TypeProto.Tensor t) {
            return toTensorType(t.elemType());
        } else if (tp.optionalType() instanceof OnnxModel.TypeProto.Optional o) {
            return OnnxType.optional(toOnnxType(o.elemType()));
        } else if (tp.sequenceType()  instanceof OnnxModel.TypeProto.Sequence s) {
            return OnnxType.seq(toOnnxType(s.elemType()));
        } else if (tp.mapType() instanceof OnnxModel.TypeProto.Map m) {
            return OnnxType.map(toKeyType(m.keyType()), toOnnxType(m.valueType()));
        } else if (tp.sparseTensorType() instanceof OnnxModel.TypeProto.SparseTensor st) {
            throw new UnsupportedOperationException("Sparse tensors not supported yet.");
        }
        throw new IllegalArgumentException("No type specified.");
    }

    static FunctionType toFunctionType(OnnxModel.GraphProto g) {
        var paramTypes = new ArrayList<TypeElement>();
        Set<String> dedup = new HashSet();
        for (OnnxModel.ValueInfoProto input : g.input()) {
            if (dedup.add(input.name())) {
                paramTypes.add(toOnnxType(input.type()));
            }
        }

        for (OnnxModel.TensorProto init : g.initializer()) {
            if (dedup.add(init.name())) {
                paramTypes.add(toTensorType(init.dataType()));
            }
        }
        var returnType = g.output().size() == 1
                ? toOnnxType(g.output().getFirst().type())
                : CoreType.tupleType(g.output().stream().map(OnnxModel.ValueInfoProto::type).map(OnnxLift::toOnnxType).toList());
        return CoreType.functionType(returnType, paramTypes);
    }

    static OnnxType toKeyType(int kt) {
        return switch (kt) {
            case 2 -> OnnxType.UINT8;
            case 3 -> OnnxType.INT8;
            case 4 -> OnnxType.UINT16;
            case 5 -> OnnxType.INT16;
            case 6 -> OnnxType.INT32;
            case 7 -> OnnxType.INT64;
            case 8 -> OnnxType.STRING;
            case 12 -> OnnxType.UINT32;
            case 13 -> OnnxType.UINT64;
            default -> throw new IllegalArgumentException("Invalid key type: " + kt);
        };
    }

    static OnnxType.TensorType toTensorType(int tt) {
        return switch (tt) {
            case 1 -> OnnxType.TENSOR_FLOAT32;
            case 2 -> OnnxType.TENSOR_UINT8;
            case 3 -> OnnxType.TENSOR_INT8;
            case 4 -> OnnxType.TENSOR_UINT16;
            case 5 -> OnnxType.TENSOR_INT16;
            case 6 -> OnnxType.TENSOR_INT32;
            case 7 -> OnnxType.TENSOR_INT64;
            case 8 -> OnnxType.TENSOR_STRING;
            case 9 -> OnnxType.TENSOR_BOOL;
            case 10 -> OnnxType.TENSOR_FLOAT16;
            case 11 -> OnnxType.TENSOR_FLOAT64;
            case 12 -> OnnxType.TENSOR_UINT32;
            case 13 -> OnnxType.TENSOR_UINT64;
            case 14 -> OnnxType.TENSOR_COMPLEX64;
            case 15 -> OnnxType.TENSOR_COMPLEX128;
            case 16 -> OnnxType.TENSOR_BFLOAT16;
            case 17 -> OnnxType.TENSOR_FLOAT8E4M3FN;
            case 18 -> OnnxType.TENSOR_FLOAT8E4M3FNUZ;
            case 19 -> OnnxType.TENSOR_FLOAT8E5M2;
            case 20 -> OnnxType.TENSOR_FLOAT8E5M2FNUZ;
            case 21 -> OnnxType.TENSOR_UINT4;
            case 22 -> OnnxType.TENSOR_INT4;
            case 23 -> OnnxType.TENSOR_FLOAT4E2M1;
            default -> OnnxType.tensor(null);
        };
    }

    static final OpFactory ONNX_OP_FACTORY = OpFactoryHelper.OP_FACTORY.get(ExplicitOnnxOps.class).andThen(OpFactoryHelper.OP_FACTORY.get(OnnxOps.class));

    static final Map<String, OnnxOp.OnnxSchema> ONNX_SCHEMA_REGISTRY = collectSchemas(ExplicitOnnxOps.class, OnnxOps.class);

    static Map<String, OnnxOp.OnnxSchema> collectSchemas(Class<?>... cls) {
        Map<String, OnnxOp.OnnxSchema> reg = new HashMap<>();
        for (Class<?> c : cls) {
            for (Class<?> nm : c.getNestMembers()) {
                for (Field f : nm.getFields()) {
                    if (f.accessFlags().contains(AccessFlag.STATIC) && OnnxOp.OnnxSchema.class.isAssignableFrom(f.getType())) try {
                        OnnxOp.OnnxSchema sch = (OnnxOp.OnnxSchema)f.get(null);
                        reg.put(sch.name(), sch);
                    } catch (ReflectiveOperationException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
        }
        return reg;
    }

    static LiftedModelWrapper lift(OnnxModel.GraphProto g) {
        var valueMap = new LinkedHashMap<String, Value>();
        var func = CoreOp.FuncOp.func(g.name(), toFunctionType(g)).body(fb -> {

            { // fill value map for parameters and initializers
                Iterator<Block.Parameter> params = fb.entryBlock().parameters().iterator();
                for (OnnxModel.ValueInfoProto input : g.input()) {
                    valueMap.put(input.name(), params.next());
                }
                for (OnnxModel.TensorProto init : g.initializer()) {
                    valueMap.computeIfAbsent(init.name(), _ -> params.next());
                }
            }

            for (OnnxModel.NodeProto n : g.node()) {
                String opType = n.opType();

                 // @@@ an old alias ? could not find the spec
                if (opType.equals("SimplifiedLayerNormalization")) {
                    opType = "LayerNormalization";
                }

                if (n.domain() != null && !n.domain().isEmpty() && !n.domain().equals("ai.onnx")) {
                    opType = n.domain() + "." + opType;
                }

                OnnxOp.OnnxSchema schema = ONNX_SCHEMA_REGISTRY.computeIfAbsent(opType, ot -> {throw new IllegalArgumentException("Unknown op type: " + ot);});
                Map<String, Object> attributes = new LinkedHashMap<>();
                if (n.attribute() != null) {
                    for (OnnxModel.AttributeProto a : n.attribute()) {
                        attributes.put(a.name(), toAttributeValue(a));
                    }
                }

                // map inputs
                List<Value> inputs = new ArrayList<>();
                if (n.input() != null) {
                    List<OnnxOp.OnnxParameter> optionalInputs = new ArrayList<>();
                    for (int i = 0; i < n.input().size(); i++) {
                        OnnxOp.OnnxParameter param = i < schema.inputs().size() ? schema.inputs().get(i) : schema.inputs().getLast();
                        Value v = valueMap.get(n.input().get(i));
                        if (v != null) {
                            switch (param.quantifier()) {
                                case REQUIRED -> {
                                    inputs.add(v);
                                }
                                case OPTIONAL -> {
                                    optionalInputs.add(param);
                                    inputs.add(v);
                                }
                                case VARIADIC -> {
                                    inputs.add(v);
                                }
                            }
                        }
                    }
                    if (!optionalInputs.isEmpty()) {
                        attributes.put("optional_inputs", optionalInputs);
                    }
                }

                // map outputs
                List<OnnxOp.OnnxParameter> optionalOutputs = new ArrayList<>();
                List<String> outputNames = new ArrayList<>();
                if (n.output() != null) {
                    for (int i = 0; i < n.output().size(); i++) {
                        OnnxOp.OnnxParameter param = schema.outputs().get(i);
                        if (!n.output().get(i).isEmpty()) {
                            outputNames.add(n.output().get(i));
                            if (param.quantifier() == OnnxOp.OnnxParameter.Quantifier.OPTIONAL) {
                                optionalOutputs.add(param);
                            }
                        }
                    }
                    if (!optionalOutputs.isEmpty()) {
                        attributes.put("optional_outputs", optionalOutputs);
                    }
                }

                // inline Constant op tensor attribute as value
                if (opType.equals("Constant") && attributes.remove(OnnxOps.Constant.Attribute.value.name()) instanceof Tensor t) {
                    switch (t.shape().length) {
                        case 0 -> { // scalar
                            switch (t.elementType()) {
                                case FLOAT -> attributes.put(OnnxOps.Constant.Attribute.value_float.name(), t.data().get(ValueLayout.JAVA_FLOAT, 0));
                                case INT64 -> attributes.put(OnnxOps.Constant.Attribute.value_int.name(), t.data().get(ValueLayout.JAVA_LONG, 0));
                                default -> attributes.put(OnnxOps.Constant.Attribute.value.name(), t);
                            }
                        }
                        case 1 -> { // 1d tensor
                            switch (t.elementType()) {
                                case FLOAT -> attributes.put(OnnxOps.Constant.Attribute.value_floats.name(), t.data().toArray(ValueLayout.JAVA_FLOAT));
                                case INT64 -> attributes.put(OnnxOps.Constant.Attribute.value_ints.name(), t.data().toArray(ValueLayout.JAVA_LONG));
                                default -> attributes.put(OnnxOps.Constant.Attribute.value.name(), t);
                            }
                        }
                        default ->  attributes.put(OnnxOps.Constant.Attribute.value.name(), t);
                    }
                }

                // get the op
                ExternalizedOp extOp = new ExternalizedOp(
                        opType,
                        Location.NO_LOCATION,
                        inputs,
                        List.of(),
                        new OnnxType.TensorType(null),
                        attributes,
                        List.of());
                OnnxOp rawOp = (OnnxOp)ONNX_OP_FACTORY.constructOpOrFail(extOp);

                // patch the op return type
                TypeElement returnType = schema.outputs().size() == 1
                        ? inferTypeVariableType(rawOp.onnxOutputs().getFirst().type(), rawOp, n)
                        : CoreType.tupleType(rawOp.onnxOutputs().stream().map(o -> inferTypeVariableType(o.type(), rawOp, n)).toList());
                extOp = new ExternalizedOp(
                        extOp.name(),
                        Location.NO_LOCATION,
                        extOp.operands(),
                        extOp.successors(),
                        returnType,
                        extOp.attributes(),
                        extOp.bodyDefinitions());
                Op.Result res = fb.op((OnnxOp)ONNX_OP_FACTORY.constructOpOrFail(extOp));

                // map outputs
                if (schema.outputs().size() == 1) {
                    valueMap.put(n.output().getFirst(), res);
                } else {
                    valueMap.put(n.name(), res);
                    for (int i = 0; i < outputNames.size(); i++) {
                        valueMap.put(outputNames.get(i), fb.op(CoreOp.tupleLoad(res, i)));
                    }
                }
            }

            if (g.output().size() == 1) {
                fb.op(CoreOp.return_(valueMap.get(g.output().getFirst().name())));
            } else {
                Op.Result ret = fb.op(CoreOp.tuple(g.output().stream().map(OnnxModel.ValueInfoProto::name).map(valueMap::get).toList()));
                valueMap.put(g.name() + "_return", ret);
                fb.op(CoreOp.return_(ret));
            }
        });
        return new LiftedModelWrapper(func, List.of(valueMap.sequencedKeySet().toArray(String[]::new)), g.initializer());
    }

    static OnnxType inferTypeVariableType(OnnxType type, OnnxOp op, OnnxModel.NodeProto n) {
        if (type instanceof OnnxType.TypeVariable tv) {
            if (tv.types().size() == 1) {
                return tv.types().getFirst();
            }
            // search for the same type variable across inputs
            for (var ie : op.onnxInputs().entrySet()) {
                if (ie.getKey().type().equals(tv)) {
                    if (ie.getValue() instanceof Value v && v.type() instanceof OnnxType ot) {
                        return ot;
                    } else if (ie.getValue() instanceof List l && !l.isEmpty() && l.getFirst() instanceof Value v && v.type() instanceof OnnxType ot) {
                        return ot;
                    }
                }
            }

            // special cases
            return switch (op) {
                case OnnxOps.Cast c ->
                    toTensorType((int)c.to());
                case OnnxOps.ConstantOfShape _, OnnxOps.Constant _-> // get tensor type from tensor attribute
                    n.attribute() != null
                    && !n.attribute().isEmpty()
                    && n.attribute().getFirst().t() instanceof OnnxModel.TensorProto tp
                            ? toTensorType(tp.dataType())
                            : OnnxType.TENSOR_FLOAT32; // default
                default ->
                    throw new IllegalArgumentException("Could not infer op type for: " + op.toText());
            };
        }
        return type;
    }

    static Object toAttributeValue(OnnxModel.AttributeProto a) {
        return switch (a.type()) {
            case FLOAT -> a.f();
            case INT -> a.i();
            case STRING -> new String(a.s());
            case TENSOR -> toTensor(a.t());
//    GRAPH = 5;
//    SPARSE_TENSOR = 11;
//    TYPE_PROTO = 13;
            case FLOATS -> joinFloatArray(a.floats());
            case INTS -> joinLongArray(a.ints());
            case STRINGS -> a.strings();
            case TENSORS -> a.tensors().stream().map(OnnxLift::toTensor).toArray(Tensor[]::new);
//    GRAPHS = 10;
//    SPARSE_TENSORS = 12;
//    TYPE_PROTOS = 14;
            default -> throw new UnsupportedOperationException("Unsupported " + a.type());
        };
    }

    static Tensor toTensor(OnnxModel.TensorProto tensorProto) {
        // @@@ floatData, longData, stringData...
        // @@@ externalData
        // @@@ segments
        return Tensor.ofShape(joinLongArray(tensorProto.dims()), tensorProto.rawData(), Tensor.ElementType.fromOnnxId(tensorProto.dataType()));
    }

    static float[] joinFloatArray(List<float[]> floats) {
        if (floats == null) return new float[0];
        float[] join = new float[floats.stream().mapToInt(f -> f.length).sum()];
        int i = 0;
        for (float[] f : floats) {
            System.arraycopy(f, 0, join, i, f.length);
            i += f.length;
        }
        return join;
    }

    static long[] joinLongArray(List<long[]> longs) {
        if (longs == null) return new long[0];
        long[] join = new long[longs.stream().mapToInt(f -> f.length).sum()];
        int i = 0;
        for (long[] f : longs) {
            System.arraycopy(f, 0, join, i, f.length);
            i += f.length;
        }
        return join;
    }

    static String toJavaName(String name) {
        name = Pattern.compile("[^\\p{Alnum}]+(\\p{Alnum})").matcher(name).replaceAll(mr -> mr.group(1).toUpperCase());
        name = name.replace("Output", "");
        return Character.toLowerCase(name.charAt(0)) + name.substring(1);
    }

    static String colorModelToANSI(String codeModel) {
        return codeModel.replaceAll("(%\\d+)([; ])", "\033[31m$1\033[0m$2")
                        .replaceAll(" : ([A-Za-z0-9:.<>\\[\\]\", ]+)", " : \033[32m$1\033[0m")
                        .replaceAll("\\)([A-Za-z0-9:.<>\\[\\]\", ]+) -> \\{", ")\033[32m$1\033[0m -> {")
                        .replaceAll("(^|    |= )([A-Za-z0-9.]+)", "$1\033[34m$2\033[0m")
                        .replaceAll("(\\^block_[0-9_]+)([;: ])", "\033[35m$1\033[0m$2");
    }

    static void extractWeights(OnnxModel.ModelProto model, Path sourceFolder, Path targetFolder) throws IOException {
        targetFolder.toFile().mkdirs();
        for (OnnxModel.TensorProto i : model.graph().initializer()) {
            Path weightFile = targetFolder.resolve(toJavaName(i.name()));
            try (var weightStream = new FileOutputStream(weightFile.toFile())) {
                if (i.floatData() instanceof List<float[]> fd) {
                    for (var fa : fd) {
                        var data = MemorySegment.ofArray(fa).toArray(ValueLayout.JAVA_BYTE);
                        weightStream.write(data);
                    }
                } else if (i.int64Data() instanceof List<long[]> ld) {
                    for (var la : ld) {
                        var data = MemorySegment.ofArray(la).toArray(ValueLayout.JAVA_BYTE);
                        weightStream.write(data);
                    }
                } else if (i.int32Data() instanceof List<int[]> id) {
                    for (var ia : id) {
                        var data = MemorySegment.ofArray(ia).toArray(ValueLayout.JAVA_BYTE);
                        weightStream.write(data);
                    }
                } else if (i.rawData() instanceof byte[] bd) {
                    weightStream.write(bd);
                } else if (i.externalData() instanceof List<OnnxModel.StringStringEntryProto> ssep) {
                    var map = ssep.stream().collect(Collectors.toUnmodifiableMap(OnnxModel.StringStringEntryProto::key, OnnxModel.StringStringEntryProto::value));
                    Path dataFilePath = sourceFolder.resolve(map.get("location"));
                    long offset = Long.parseLong(map.get("offset"));
                    int length = Integer.parseInt(map.get("length"));
                    try (var dataFile = new RandomAccessFile(dataFilePath.toString(), "r");Arena a = Arena.ofConfined()) {
                        var ms = dataFile.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, length, a);
                        weightStream.write(ms.toArray(ValueLayout.JAVA_BYTE));
                    }
                } else {
                    throw new UnsupportedOperationException();
                }
            }
            IO.println(weightFile + " extracted.");
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            IO.println("Usage: OnnxLift <model.onnx> <target folder>");
        } else {
            Path source = Path.of(args[0]);
            Path targetFolder = Path.of(args[1]);
            try (var in = new RandomAccessFile(source.toFile(), "r")) {
                OnnxModel.ModelProto protoModel = OnnxModel.readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()));
                LiftedModelWrapper liftedModel = lift(protoModel.graph());

                IO.println(colorModelToANSI(liftedModel.toText()));

                Path java = targetFolder.resolve("Model.java");
                Files.writeString(java, liftedModel.toJava());
                IO.println(java + " generated.");

                extractWeights(protoModel, source.getParent(), targetFolder);
            }
        }
    }
}
