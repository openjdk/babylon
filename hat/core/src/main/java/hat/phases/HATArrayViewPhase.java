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
package hat.phases;

import hat.dialect.HATPtrOp;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.Trxfmr;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.VarTable;
import optkl.util.ops.VarLikeOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static hat.phases.HATPhaseUtils.findOpInResultFromFirstOperandsOrNull;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.classTypeToTypeOrThrow;
import static optkl.OpHelper.copyLocation;
import static optkl.OpHelper.firstOperandOrThrow;
import static optkl.OpHelper.opFromFirstOperandOrNull;
import static optkl.OpHelper.opFromFirstOperandOrThrow;
import static optkl.OpHelper.resultFromFirstOperandOrNull;
import static optkl.OpHelper.resultFromFirstOperandOrThrow;
import static optkl.OpHelper.resultFromOperandN;

public record HATArrayViewPhase() implements HATPhase {
    public static boolean isVectorOp(MethodHandles.Lookup lookup, Op op) {
        if (!op.operands().isEmpty()) {
            CodeType type = switch (op) {
                case JavaOp.ArrayAccessOp.ArrayLoadOp load -> load.resultType();
                case JavaOp.ArrayAccessOp.ArrayStoreOp store -> store.operands().getLast().type();
                default -> OpHelper.firstOperandOrThrow(op).type();
            };
            if (type instanceof ArrayType at) {
                type = at.componentType();
            }
            if (type instanceof ClassType ct) {
                try {
                    return IfaceValue.Vector.class.isAssignableFrom((Class<?>) ct.resolve(lookup));
                } catch (ReflectiveOperationException e) {
                    throw new IllegalStateException(e);
                }
            }
        }
        return false;
    }

    public static boolean isVectorBinaryOp(MethodHandles.Lookup lookup, OpHelper.Invoke invoke) {
        return isVectorOp(lookup, invoke.op()) && invoke.nameMatchesRegex("(add|sub|mul|div)");
    }

    public static boolean isBufferArray(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrNull(op, JavaOp.InvokeOp.class);
        return iop != null && iop.invokeReference().name().toLowerCase().contains("arrayview"); // we need a better way
    }

    public static boolean isBufferInitialize(Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop && vop.varValueType() instanceof ArrayType
                || op instanceof JavaOp.ArrayAccessOp
                || op.resultType() instanceof ArrayType) return isBufferArray(op);
        return false;
    }

    public static boolean isLocalSharedOrPrivate(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrNull(op, JavaOp.InvokeOp.class);
        return iop != null
                && (iop.invokeReference().name().toLowerCase().contains("shared")
                || iop.invokeReference().name().toLowerCase().contains("local")
                || iop.invokeReference().name().toLowerCase().contains("private")
        );
    }

    static HATArrayViewPhase.ArrayAccessInfo arrayAccessInfo(Value value, Map<Op.Result, Op.Result> replaced) {
        return expressionGraph(value).getInfo(replaced);
    }

    static HATArrayViewPhase.Node<Value> expressionGraph(Value value) {
        return expressionGraph(new HashMap<>(), value);
    }

    static HATArrayViewPhase.Node<Value> expressionGraph(Map<Value, HATArrayViewPhase.Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the expression graphs for each operand
        List<HATArrayViewPhase.Node<Value>> edges = new ArrayList<>();

        // looks like
        for (Value operand : value.dependsOn()) {
            if (operand instanceof Op.Result res &&
                    res.op() instanceof JavaOp.InvokeOp iop
                    && iop.invokeReference().name().toLowerCase().contains("arrayview")) { // We need to find a better way
                continue;
            }
            edges.add(expressionGraph(operand));
        }
        HATArrayViewPhase.Node<Value> node = new HATArrayViewPhase.Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        if (Invoke.stream(lookup, funcOp).noneMatch(
                invoke -> isBufferArray(invoke.op())
        )) return funcOp;

        funcOp = applyArrayView(lookup, funcOp);

        if (funcOp.elements().filter(e -> e instanceof CoreOp.VarOp).anyMatch(
                e -> isVectorOp(lookup, ((CoreOp.VarOp) e))
        )) funcOp = applyVectorView(lookup, funcOp, varTable);
        return funcOp;
    }

    public CoreOp.FuncOp applyVectorView(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        return Trxfmr.of(lookup, funcOp).transform((blockBuilder, op) -> {
            switch (op) {
                case JavaOp.InvokeOp iOp when invoke(lookup, iOp) instanceof Invoke invoke && isVectorBinaryOp(invoke.lookup(), invoke) ->
                        blockBuilder.add(op);
                case CoreOp.VarOp varOp when isVectorOp(lookup, varOp) -> {
                    Op.Result op1 = blockBuilder.add(varOp);
                    String functionName = funcOp.funcName();
                    varTable.addIfNeededOrThrow(functionName, op1.op(), VarTable.HATOpAttribute.VECTOR);
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                    if (isVectorOp(lookup, arrayLoadOp)) {
                        blockBuilder.add(op);
                    }
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                    if (isVectorOp(lookup, arrayStoreOp)) {
                        blockBuilder.add(op);
                    }
                    return blockBuilder;
                }
                default -> {
                }
            }
            blockBuilder.add(op);
            return blockBuilder;
        }).funcOp();
    }

    public CoreOp.FuncOp applyArrayView(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return Trxfmr.of(lookup, funcOp).transform((blockBuilder, op) -> {
            var context = blockBuilder.context();
            switch (op) {
                case JavaOp.InvokeOp invokeOp when invoke(lookup, invokeOp) instanceof Invoke invoke && isBufferArray(invoke.op()) -> {
                    Op.Result result = invoke.resultFromFirstOperandOrNull();
                    replaced.put(invoke.returnResult(), result);
                    // map buffer VarOp to its corresponding VarLoadOp
                    bufferVarLoads.put((opFromFirstOperandOrNull(result.op())), (CoreOp.VarAccessOp.VarLoadOp) result.op());
                    return blockBuilder;
                }
                case CoreOp.VarOp varOp when isBufferInitialize(varOp) -> {
                    Op bufferLoad = replaced.get(resultFromFirstOperandOrThrow(varOp)).op(); // gets VarLoadOp associated w/ og buffer
                    replaced.put(varOp.result(), resultFromFirstOperandOrNull(bufferLoad)); // gets VarOp associated w/ og buffer
                    return blockBuilder;
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp when (isBufferInitialize(varLoadOp)) -> {
                    Op.Result r = resultFromFirstOperandOrThrow(varLoadOp);
                    Op.Result replacement;
                    if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                        replacement = (isLocalSharedOrPrivate(varLoadOp)) ?
                                resultFromFirstOperandOrNull(opFromFirstOperandOrThrow(r.op())) :
                                bufferVarLoads.get(replaced.get(r).op()).result();
                    } else { // if this is a VarLoadOp loading the buffer
                        CoreOp.VarAccessOp.VarLoadOp newVarLoad = CoreOp.VarAccessOp.varLoad(blockBuilder.context().getValue(replaced.get(r)));
                        replacement = blockBuilder.add(copyLocation(varLoadOp, newVarLoad));
                        context.mapValue(varLoadOp.result(), replacement);
                    }
                    replaced.put(varLoadOp.result(), replacement);
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp when isBufferArray(arrayLoadOp) -> {
                    Op replacementOp;
                    if (isVectorOp(lookup, arrayLoadOp)) {
                        replacementOp = JavaOp.arrayLoadOp(
                                context.getValue(replaced.get((Op.Result) arrayLoadOp.operands().getFirst())),
                                context.getValue(arrayLoadOp.operands().getLast()),
                                arrayLoadOp.resultType()
                        );
                    } else if (((ArrayType) firstOperandOrThrow(op).type()).dimensions() == 1) {
                        var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                        var operands = arrayAccessInfo.bufferAndIndicesAsValues();
                        replacementOp = new HATPtrOp.HATPtrLoadOp(
                                arrayAccessInfo.bufferName(),
                                arrayLoadOp.resultType(),
                                (Class<?>) classTypeToTypeOrThrow(lookup, (ClassType) arrayAccessInfo.buffer().type()),
                                context.getValues(operands)
                        );
                    } else { // we only use the last array load
                        return blockBuilder;
                    }
                    context.mapValue(arrayLoadOp.result(), blockBuilder.add(copyLocation(arrayLoadOp, replacementOp)));
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp when isBufferArray(arrayStoreOp) -> {
                    Op replacementOp;
                    if (isVectorOp(lookup, arrayStoreOp)) {
                        replacementOp = JavaOp.arrayStoreOp(
                                context.getValue(replaced.get((Op.Result) arrayStoreOp.operands().getFirst())),
                                context.getValue(arrayStoreOp.operands().get(1)),
                                context.getValue(arrayStoreOp.operands().getLast())
                        );
                    } else if (((ArrayType) firstOperandOrThrow(op).type()).dimensions() == 1) { // we only use the last array load
                        var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                        var operands = arrayAccessInfo.bufferAndIndicesAsValues();
                        operands.add(arrayStoreOp.operands().getLast());
                        replacementOp = new HATPtrOp.HATPtrStoreOp(
                                arrayAccessInfo.bufferName(),
                                arrayStoreOp.resultType(),
                                (Class<?>) classTypeToTypeOrThrow(lookup, (ClassType) arrayAccessInfo.buffer().type()),
                                context.getValues(operands)
                        );
                    } else {
                        return blockBuilder;
                    }
                    context.mapValue(arrayStoreOp.result(), blockBuilder.add(copyLocation(arrayStoreOp, replacementOp)));
                    return blockBuilder;
                }
                case JavaOp.ArrayLengthOp arrayLengthOp when
                        isBufferArray(arrayLengthOp) && resultFromFirstOperandOrThrow(arrayLengthOp) != null -> {
                    var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                    var hatPtrLengthOp = new HATPtrOp.HATPtrLengthOp(
                            arrayAccessInfo.bufferName(),
                            arrayLengthOp.resultType(),
                            (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup, (ClassType) arrayAccessInfo.buffer().type()),
                            context.getValues(List.of(arrayAccessInfo.buffer()))
                    );
                    context.mapValue(arrayLengthOp.result(), blockBuilder.add(copyLocation(arrayLengthOp, hatPtrLengthOp)));
                    return blockBuilder;
                }
                default -> {
                }
            }
            blockBuilder.add(op);
            return blockBuilder;
        }).funcOp();
    }

    record ArrayAccessInfo(Op.Result buffer, String bufferName, List<Op.Result> indices) {
        public List<Value> bufferAndIndicesAsValues() {
            List<Value> operands = new ArrayList<>(List.of(buffer));
            operands.addAll(indices);
            return operands;
        }
    }

    record Node<T>(T value, List<Node<T>> edges) {
        ArrayAccessInfo getInfo(Map<Op.Result, Op.Result> replaced) {
            List<Node<T>> nodeList = new ArrayList<>(List.of(this));
            Set<Node<T>> handled = new HashSet<>();
            Op.Result buffer = null;
            List<Op.Result> indices = new ArrayList<>();
            while (!nodeList.isEmpty()) {
                Node<T> node = nodeList.removeFirst();
                handled.add(node);
                if (node.value instanceof Op.Result res &&
                        (res.op() instanceof JavaOp.ArrayAccessOp || res.op() instanceof JavaOp.ArrayLengthOp)) {
                    buffer = res;
                    // idx location differs between ArrayAccessOp and ArrayLengthOp
                    indices.addFirst(res.op() instanceof JavaOp.ArrayAccessOp
                            ? resultFromOperandN(res.op(), 1)
                            : resultFromOperandN(res.op(), 0)
                    );
                }
                if (!node.edges().isEmpty()) {
                    Node<T> next = node.edges().getFirst(); // we only traverse through the index-related ops
                    if (!handled.contains(next)) {
                        nodeList.add(next);
                    }
                }
            }
            if (buffer != null) {
                buffer = replaced.get(resultFromFirstOperandOrNull(buffer.op()));
                String bufferName = hatPtrName(opFromFirstOperandOrNull(buffer.op()));
                return new ArrayAccessInfo(buffer, bufferName, indices);
            } else {
                return null;
            }
        }
    }

    public static String hatPtrName(Op op) {
        return switch (op) {
            case CoreOp.VarOp varOp -> varOp.varName();
            case VarLikeOp varLikeOp -> varLikeOp.varName();
            case null, default -> "";
        };
    }
}
