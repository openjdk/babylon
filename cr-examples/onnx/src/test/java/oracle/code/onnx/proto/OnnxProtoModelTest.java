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

package oracle.code.onnx.proto;

import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedSet;
import java.util.stream.Collectors;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.op.OpFactory;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.TupleType;
import jdk.incubator.code.writer.OpWriter;
import oracle.code.onnx.ir.OnnxOp;
import oracle.code.onnx.ir.OnnxOps;
import oracle.code.onnx.ir.OnnxType;


public class OnnxProtoModelTest {

    static OnnxType toOnnxType(OnnxProtoModel.TypeProto tp) {
        if (tp.tensorType() instanceof OnnxProtoModel.TypeProto.Tensor t) {
            return toTensorType(t.elemType());
        } else if (tp.optionalType() instanceof OnnxProtoModel.TypeProto.Optional o) {
            return OnnxType.optional(toOnnxType(o.elemType()));
        } else if (tp.sequenceType()  instanceof OnnxProtoModel.TypeProto.Sequence s) {
            return OnnxType.seq(toOnnxType(s.elemType()));
        } else if (tp.mapType() instanceof OnnxProtoModel.TypeProto.Map m) {
            return OnnxType.map(toKeyType(m.keyType()), toOnnxType(m.valueType()));
        } else if (tp.sparseTensorType() instanceof OnnxProtoModel.TypeProto.SparseTensor st) {
            throw new UnsupportedOperationException("Sparse tensors not supported yet."); // @@@
        }
        throw new IllegalArgumentException("No type specified.");
    }

    static FunctionType toFunctionType(OnnxProtoModel.GraphProto g) {
        var paramTypes = new ArrayList<TypeElement>();
        for (OnnxProtoModel.ValueInfoProto input : g.inputs()) {
            paramTypes.add(toOnnxType(input.type()));
        }
        for (OnnxProtoModel.TensorProto init : g.initializers()) {
            paramTypes.add(toTensorType(init.dataType()));
        }
        var returnType = g.outputs().size() == 1
                ? toOnnxType(g.outputs().getFirst().type())
                : TupleType.tupleType(g.outputs().stream().map(OnnxProtoModel.ValueInfoProto::type).map(OnnxProtoModelTest::toOnnxType).toList());
        return FunctionType.functionType(returnType, paramTypes);
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

    static final OpFactory ONNX_FACTORY = OpFactory.OP_FACTORY.get(OnnxOps.class);


    record OpWithNames<T extends Op> (T op, List<String> names) {
        public String toText() {
            var defNamer = OpWriter.CodeItemNamerOption.defaultValue().namer();
            var namer = new HashMap<Value, Integer>();
            return OpWriter.toText(op, OpWriter.CodeItemNamerOption.of(ci -> ci instanceof Value v ? names.get(namer.computeIfAbsent(v, _ -> namer.size())) : defNamer.apply(ci)));

        }
    }

    static OpWithNames toFuncOp(OnnxProtoModel.GraphProto g) {
        var valueMap = new LinkedHashMap<String, Value>();
        var func = CoreOp.FuncOp.func(g.name(), toFunctionType(g)).body(fb -> {

            { // fill value map for parameters and initializers
                Iterator<Block.Parameter> params = fb.entryBlock().parameters().iterator();
                for (OnnxProtoModel.ValueInfoProto input : g.inputs()) {
                    valueMap.put(input.name(), params.next());
                }
                for (OnnxProtoModel.TensorProto init : g.initializers()) {
                    valueMap.put(init.name(), params.next());
                }
            }

            for (OnnxProtoModel.NodeProto n : g.nodes()) {
                // get the op
                ExternalizableOp.ExternalizedOp extOp = new ExternalizableOp.ExternalizedOp(
                        n.opType(),
                        n.inputs() == null ? List.of() : n.inputs().stream().map(valueMap::get).toList(),
                        List.of(),
                        new OnnxType.TensorType(null),
                        n.attributes() == null ? Map.of() : n.attributes().stream().collect(Collectors.toMap(OnnxProtoModel.Attribute::name, OnnxProtoModelTest::toAttributeValue)),
                        List.of());
                OnnxOp rawOp = (OnnxOp)ONNX_FACTORY.constructOpOrFail(extOp);

                // patch the op return type
                TypeElement returnType = rawOp.onnxOutputs().size() == 1
                        ? inferTypeVariableType(rawOp.onnxOutputs().getFirst().type(), rawOp, n)
                        : TupleType.tupleType(rawOp.onnxOutputs().stream().map(o -> inferTypeVariableType(o.type(), rawOp, n)).toList());
                extOp = new ExternalizableOp.ExternalizedOp(
                        extOp.name(),
                        extOp.operands(),
                        extOp.successors(),
                        returnType,
                        extOp.attributes(),
                        extOp.bodyDefinitions());
                Op.Result res = fb.op((OnnxOp)ONNX_FACTORY.constructOpOrFail(extOp));

                // map outputs
                if (rawOp.onnxOutputs().size() == 1) {
                    valueMap.put(n.outputs().getFirst(), res);
                } else {
                    valueMap.put(n.name(), res);
                    for (int i = 0; i < n.outputs().size(); i++) {
                        valueMap.put(n.outputs().get(i), fb.op(CoreOp.tupleLoad(res, i)));
                    }
                }
            }

            if (g.outputs().size() == 1) {
                fb.op(CoreOp._return(valueMap.get(g.outputs().getFirst().name())));
            } else {
                Op.Result ret = fb.op(CoreOp.tuple(g.outputs().stream().map(OnnxProtoModel.ValueInfoProto::name).map(valueMap::get).toList()));
                valueMap.put(g.name() + "_return", ret);
                fb.op(CoreOp._return(ret));
            }
        });

        return new OpWithNames(func, List.of(valueMap.sequencedKeySet().toArray(String[]::new)));
    }

    static OnnxType inferTypeVariableType(OnnxType type, OnnxOp op, OnnxProtoModel.NodeProto n) {
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
                case OnnxOps.ConstantOfShape cos -> // get tensor type from tensor attribute
                    n.attributes() != null
                    && !n.attributes().isEmpty()
                    && n.attributes().getFirst().t() instanceof OnnxProtoModel.TensorProto tp
                            ? toTensorType(tp.dataType())
                            : OnnxType.TENSOR_FLOAT32; // default
                default ->
                    throw new IllegalArgumentException("Could not infer op type for: " + op.toText());
            };
        }
        return type;
    }

    static Object toAttributeValue(OnnxProtoModel.Attribute a) {
        return switch (a.type()) {
            case 1 -> a.f();
            case 2 -> a.i();
            case 3 -> a.s();
            case 4 -> a.t().rawData(); // @@@ need to store all tensor info + data
//    GRAPH = 5;
//    SPARSE_TENSOR = 11;
//    TYPE_PROTO = 13;
            case 6 -> joinFloatArray(a.floats());
            case 7 -> joinLongArray(a.ints());
            case 8 -> a.strings();
//    TENSORS = 9;
//    GRAPHS = 10;
//    SPARSE_TENSORS = 12;
//    TYPE_PROTOS = 14;
            default -> throw new UnsupportedOperationException("Unsupported " + a.type());
        };
    }

    static float[] joinFloatArray(List<float[]> floats) {
        float[] join = new float[floats.stream().mapToInt(f -> f.length).sum()];
        int i = 0;
        for (float[] f : floats) {
            System.arraycopy(f, 0, join, i, f.length);
            i += f.length;
        }
        return join;
    }

    static long[] joinLongArray(List<long[]> floats) {
        long[] join = new long[floats.stream().mapToInt(f -> f.length).sum()];
        int i = 0;
        for (long[] f : floats) {
            System.arraycopy(f, 0, join, i, f.length);
            i += f.length;
        }
        return join;
    }

    public static void main(String[] args) throws Exception {
        for (var fName : args) {
            try (var in = new RandomAccessFile(fName, "r")) {
                OnnxProtoModel model = OnnxProtoModel.readFrom(in.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, in.length()));
//                System.out.println(model.toText());
                System.out.println(toFuncOp(model.graph()).toText());
            }
        }
    }
}
