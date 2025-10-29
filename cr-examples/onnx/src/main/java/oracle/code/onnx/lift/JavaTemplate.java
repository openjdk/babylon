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

import java.lang.foreign.ValueLayout;
import java.lang.reflect.Method;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SequencedMap;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeItem;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import oracle.code.onnx.OnnxOperators;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxType;
import oracle.code.onnx.proto.OnnxModel;

final class JavaTemplate {

    private static final String WEIGHT_FIELD_TEMPLATE = """
            final %s %s = load("%2$s", %s%s);
        """;

    private static final String TEMPLATE = """
        import java.io.IOException;
        import java.io.RandomAccessFile;
        import java.lang.foreign.Arena;
        import java.lang.foreign.MemorySegment;
        import java.nio.channels.FileChannel;
        import java.util.HexFormat;
        import java.util.List;
        import jdk.incubator.code.CodeReflection;
        import oracle.code.onnx.Tensor;

        import static java.util.Optional.*;
        import static oracle.code.onnx.OnnxOperators.*;
        import static oracle.code.onnx.Tensor.ElementType.*;

        public class %s {

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

        %s
            @CodeReflection
            public Object mainGraph(
        %s) {%s
            }
        }
        """;

    static String toJava(OnnxLift.LiftedModelWrapper model, String className) {
        Block entryBlock = model.func().bodies().getFirst().entryBlock();
        List<Block.Parameter> parameters = entryBlock.parameters();
        Function<CodeItem, String> namer = OnnxLift.namer(model.names());
        parameters.forEach(namer::apply); // initialize namer with all parameters first

        return TEMPLATE.formatted(
                className,
                weightFields(namer, parameters, model.weights()),
                parameters(namer, parameters, model.weights()),
                body(namer, entryBlock.ops()));
    }

    private static String weightFields(Function<CodeItem, String> namer, List<Block.Parameter> parameters, List<OnnxModel.TensorProto> weights) {
        StringBuilder out = new StringBuilder();
        List<jdk.incubator.code.Block.Parameter> weightParams = parameters.subList(parameters.size() - weights.size(), parameters.size());
        Map<String, oracle.code.onnx.proto.OnnxModel.TensorProto> wMap = weights.stream().collect(Collectors.toUnmodifiableMap(OnnxModel.TensorProto::name, Function.identity()));
        for (int i = 0; i < weightParams.size(); i++) {
            Block.Parameter wp = weightParams.get(i);
            OnnxModel.TensorProto w = wMap.get(namer.apply(wp));
            String name = OnnxLift.toJavaName(w.name());
            long[] dims = OnnxLift.joinLongArray(w.dims());
            out.append(WEIGHT_FIELD_TEMPLATE.formatted(toJavaType(wp.type()), name, Tensor.ElementType.fromOnnxId(w.dataType()).name(), dims.length > 0 ? (", " + longJoin(dims)) : ""));
        }
        return out.toString();
    }

    private static String parameters(Function<CodeItem, String> namer, List<Block.Parameter> parameters, List<OnnxModel.TensorProto> weights) {
        StringBuilder out = new StringBuilder();
        int realParamsSize = parameters.size() - weights.size();
        for (int i = 0; i < realParamsSize; i++) {
            if (i > 0) {
                out.append(",\n");
            }
            Block.Parameter param = parameters.get(i);
            out.append("            ").append(toJavaType(param.type())).append(' ').append(OnnxLift.toJavaName(namer.apply(param)));
        }
        return out.toString();
    }

    private static String body(Function<CodeItem, String> namer, List<Op> ops) {
        StringBuilder out = new StringBuilder();
        for (jdk.incubator.code.Op op : ops) {
            if (!(op instanceof CoreOp.TupleLoadOp)) {
                // lazy tupple loads
                out.append("\n        ");
                if (!op.resultType().equals(JavaType.VOID)) {
                    out.append(toJavaType(op.resultType())).append(' ').append(OnnxLift.toJavaName(namer.apply(op.result()))).append(" = ");
                }
                switch (op) {
                    case OnnxOp oo -> {
                        String opName = op.externalizeOpName();
                        out.append(opName.substring(opName.lastIndexOf('.') + 1)).append('(');
                        OnnxOp.OnnxSchema schema = getSchema(oo);
                        SequencedMap<OnnxOp.OnnxParameter, Object> inputs = oo.onnxInputs();
                        boolean first = true;
                        for (OnnxOp.OnnxParameter oi : schema.inputs()) {
                            if (first) {
                                first = false;
                            } else {
                                out.append(", ");
                            }
                            out.append(toJava(namer, inputs.get(oi)));
                        }
                        Map<String, Object> attrs = oo.onnxAttributes();
                        for (OnnxOp.OnnxAttribute oa : schema.attributes()) {
                            if (first) {
                                first = false;
                            } else {
                                out.append(", ");
                            }
                            Object a = attrs.get(oa.name());
                            if (a == null) {
                                out.append("empty()");
                            } else if (oa.isOptional()) {
                                out.append("of(").append(toString(a)).append(')');
                            } else {
                                out.append(toString(a));
                            }
                        }
                        out.append(");");
                    }
                    case CoreOp.TupleOp to -> {
                        out.append("List.of(");
                        boolean first = true;
                        for (jdk.incubator.code.Value te : to.operands()) {
                            if (first) {
                                first = false;
                            } else {
                                out.append(", ");
                            }
                            out.append(toJava(namer, te));
                        }
                        out.append(");");
                    }
                    case CoreOp.ReturnOp ro -> {
                        out.append("return ").append(toJava(namer, ro.operands().getFirst())).append(';');
                    }
                    default -> throw new UnsupportedOperationException(op.toText());
                }
            } else {
                namer.apply(op.result());
            }
        }
        return out.toString();
    }

    private static String toString(Object o) {
        return switch (o) {
            case long[] la -> newArray(la);
            case float[] fa -> newArray(fa);
            case Long l -> l.toString() + "L";
            case Float f -> f.toString() + "F";
            case String s -> "\"" + s + "\"";
            case Tensor t when t.shape().length == 0 && t.elementType() == Tensor.ElementType.BOOL -> "Tensor.ofScalar(" + t.data().get(ValueLayout.JAVA_BOOLEAN, 0) + ")";
            case Tensor t when t.shape().length == 0 && t.elementType() == Tensor.ElementType.INT8 -> "Tensor.ofScalar((byte)" + t.data().get(ValueLayout.JAVA_BYTE, 0) + ")";
            case Tensor t -> "Tensor.ofShape(" + toString(t.shape()) + ", " + switch (t.elementType()) {
                case FLOAT -> toString(t.data().toArray(ValueLayout.JAVA_FLOAT));
                case INT64 -> toString(t.data().toArray(ValueLayout.JAVA_LONG));
                default -> "HexFormat.of().parseHex(\"" + HexFormat.of().formatHex(t.data().toArray(ValueLayout.JAVA_BYTE)) + "\"), " + t.elementType().name();
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
        return LongStream.of(la).mapToObj(d -> String.valueOf(d) + "L").collect(Collectors.joining(", "));
    }

    private static String newArray(float[] fa) {
        for (float f : fa) {
            if (f != 0f) {
                return IntStream.range(0, fa.length).mapToObj(i -> String.valueOf(fa[i]) + "F").collect(Collectors.joining(", ", "new float[] {", "}"));
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
            case Op.Result or when or.op() instanceof CoreOp.TupleLoadOp tlo -> OnnxLift.toJavaName(namer.apply(tlo.operands().getFirst())) + '.' + tupleAccessor(tlo.operands().getFirst(), tlo.index());
            case Value v -> OnnxLift.toJavaName(namer.apply(v));
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
