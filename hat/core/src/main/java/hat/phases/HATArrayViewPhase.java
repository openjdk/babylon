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

import hat.dialect.*;
import jdk.incubator.code.TypeElement;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.Trxfmr;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.*;
import optkl.util.ops.VarLikeOp;

import java.lang.invoke.MethodHandles;
import java.util.*;

import static hat.phases.HATPhaseUtils.findOpInResultFromFirstOperandsOrNull;
import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.*;
import static optkl.OpHelper.Invoke.invoke;

public record HATArrayViewPhase() implements HATPhase {
    static public boolean isVectorOp(MethodHandles.Lookup lookup, Op op) {
        if (!op.operands().isEmpty()) {
            TypeElement type = switch(op) {
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
                    throw new RuntimeException(e);
                }
            }
        }
        return false;
    }

    static public boolean isVectorBinaryOp(MethodHandles.Lookup lookup, OpHelper.Invoke invoke) {
        return isVectorOp(lookup, invoke.op()) && invoke.nameMatchesRegex("(add|sub|mul|div)");
    }


    static HATVectorOp.HATVectorBinaryOp buildVectorBinaryOp(String varName, String opType, IfaceValue.Vector.Shape vectorShape, List<Value> outputOperands) {
        return switch (opType) {
            case "add" -> new HATVectorOp.HATVectorBinaryOp.HATVectorAddOp(varName,  vectorShape, outputOperands);
            case "sub" -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName,  vectorShape, outputOperands);
            case "mul" -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName,  vectorShape, outputOperands);
            case "div" -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName,  vectorShape, outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + opType);
        };
    }
    static public boolean isBufferArray(MethodHandles.Lookup lookup, Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrNull(op, JavaOp.InvokeOp.class);
        return iop != null && iop.invokeReference().name().toLowerCase().contains("arrayview"); // we need a better way
    }

    static public boolean isBufferInitialize(MethodHandles.Lookup lookup, Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop && vop.varValueType() instanceof ArrayType
                || op instanceof JavaOp.ArrayAccessOp
                || op.resultType() instanceof ArrayType) return isBufferArray(lookup, op);
        return false;
    }
    static public boolean isLocalSharedOrPrivate(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrNull(op, JavaOp.InvokeOp.class);
        return iop != null
                && (iop.invokeReference().name().toLowerCase().contains("shared")
                || iop.invokeReference().name().toLowerCase().contains("local")
                || iop.invokeReference().name().toLowerCase().contains("private")
        );
    }

    static public HATVectorOp buildArrayViewVector(Op op, String name, TypeElement resultType, IfaceValue.Vector.Shape vectorShape, List<Value> operands) {
        if (isLocalSharedOrPrivate(op)) {
            if (op instanceof JavaOp.ArrayAccessOp.ArrayLoadOp) {
                return new HATVectorOp.HATVectorLoadOp.HATSharedVectorLoadOp(name, resultType, vectorShape, operands);
            }
            return new HATVectorOp.HATVectorStoreView.HATSharedVectorStoreView(name, resultType, vectorShape, operands);
        } else {
            if (op instanceof JavaOp.ArrayAccessOp.ArrayLoadOp) {
                return new HATVectorOp.HATVectorLoadOp.HATPrivateVectorLoadOp(name, resultType, vectorShape, operands);
            }
            return new HATVectorOp.HATVectorStoreView.HATPrivateVectorStoreView(name, resultType, vectorShape, operands);
        }
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
                    && iop.invokeReference().name().toLowerCase().contains("arrayview")){ // We need to find a better way
                continue;
            }
            edges.add(expressionGraph(operand));
        }
        HATArrayViewPhase.Node<Value> node = new HATArrayViewPhase.Node<>(value, edges);
        visited.put(value, node);
        return node;
    }
    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        if (Invoke.stream(lookup, funcOp).noneMatch(
                invoke -> isBufferArray(lookup, invoke.op())
        )) return funcOp;

        funcOp = applyArrayView(lookup,funcOp);

        if (funcOp.elements().filter(e -> e instanceof CoreOp.VarOp).anyMatch(
                e -> isVectorOp(lookup, ((CoreOp.VarOp) e))
        )) funcOp = applyVectorView(lookup,funcOp);
        return funcOp;
    }

    public CoreOp.FuncOp applyVectorView(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        return Trxfmr.of(lookup,funcOp).transform((blockBuilder, op) -> {
            var context = blockBuilder.context();
            switch (op) {
                case JavaOp.InvokeOp $ when invoke(lookup, $) instanceof Invoke invoke -> {
                    if (isVectorBinaryOp(invoke.lookup(), invoke)){
                        var hatVectorBinaryOp = buildVectorBinaryOp(
                                invoke.varOpFromFirstUseOrThrow().varName(),
                                invoke.name(),// so mul, sub etc
                                getVectorShape(lookup,invoke.returnType()),
                                blockBuilder.context().getValues(invoke.op().operands())
                        );
                        context.mapValue(invoke.returnResult(), blockBuilder.op(copyLocation(invoke.op(),hatVectorBinaryOp)));
                        return blockBuilder;
                    }
                }
                case CoreOp.VarOp varOp -> {
                    if (isVectorOp(lookup,varOp)) {
                        var hatVectorVarOp = new HATVectorOp.HATVectorVarOp(
                                varOp.varName(),
                                varOp.resultType(),
                                getVectorShape(lookup,varOp.resultType().valueType()),
                                context.getValues(OpHelper.firstOperandAsListOrEmpty(varOp))
                        );
                        context.mapValue(varOp.result(), blockBuilder.op(copyLocation(varOp,hatVectorVarOp)));
                        return blockBuilder;
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                    if (isVectorOp(lookup,arrayLoadOp)) {
                        Op.Result buffer = resultFromFirstOperandOrNull(arrayLoadOp);
                        String name = hatPtrName(opFromFirstOperandOrThrow(buffer.op()));
                        var resultType = CoreType.varType(arrayLoadOp.resultType());
                        var vectorShape = getVectorShape(lookup,arrayLoadOp.resultType());
                        List<Value> operands = context.getValues(List.of(buffer, arrayLoadOp.operands().getLast()));
                        HATVectorOp vLoadOp = buildArrayViewVector(arrayLoadOp, name, resultType, vectorShape, operands);
                        context.mapValue(arrayLoadOp.result(), blockBuilder.op(copyLocation(arrayLoadOp,vLoadOp)));
                    }
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                    if (isVectorOp(lookup,arrayStoreOp)) {
                        Op.Result buffer = resultFromFirstOperandOrThrow(arrayStoreOp);
                        Op varOp = opFromFirstOperandOrNull(((Op.Result) arrayStoreOp.operands().getLast()).op());
                        String name = hatPtrName(varOp);
                        var resultType = (varOp).resultType();
                        var vectorShape = getVectorShape(lookup,arrayStoreOp.operands().getLast().type());
                        List<Value> operands = context.getValues(List.of(buffer, arrayStoreOp.operands().getLast(), arrayStoreOp.operands().get(1)));
                        HATVectorOp vStoreOp = buildArrayViewVector(arrayStoreOp, name, resultType, vectorShape, operands);
                        context.mapValue(arrayStoreOp.result(), blockBuilder.op(copyLocation(arrayStoreOp,vStoreOp)));
                    }
                    return blockBuilder;
                }
                default -> {
                }
            }
            blockBuilder.op(op);
            return blockBuilder;
        }).funcOp();
    }

    public CoreOp.FuncOp applyArrayView(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return Trxfmr.of(lookup,funcOp).transform((blockBuilder, op) -> {
            var context = blockBuilder.context();
            switch (op) {
                case JavaOp.InvokeOp $ when invoke(lookup, $) instanceof Invoke invoke -> {
                    if (isBufferArray(lookup, invoke.op())) { // ensures we can use iop as key for replaced vvv
                        Op.Result result = invoke.resultFromFirstOperandOrNull();
                        replaced.put(invoke.returnResult(), result);
                        // map buffer VarOp to its corresponding VarLoadOp
                        bufferVarLoads.put((opFromFirstOperandOrNull(result.op())), (CoreOp.VarAccessOp.VarLoadOp) result.op());
                        return blockBuilder;
                    }
                }
                case CoreOp.VarOp varOp -> {
                    if (isBufferInitialize(lookup, varOp)) {
                        // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(resultFromFirstOperandOrThrow(varOp)).op(); // gets VarLoadOp associated w/ og buffer
                        replaced.put(varOp.result(), resultFromFirstOperandOrNull(bufferLoad)); // gets VarOp associated w/ og buffer
                        return blockBuilder;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                    if ((isBufferInitialize(lookup, varLoadOp))) {
                        Op.Result r = resultFromFirstOperandOrThrow(varLoadOp);
                        Op.Result replacement;
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            replacement = (isLocalSharedOrPrivate(varLoadOp)) ?
                                    resultFromFirstOperandOrNull(opFromFirstOperandOrThrow(r.op())) :
                                    bufferVarLoads.get(replaced.get(r).op()).result();
                        } else { // if this is a VarLoadOp loading the buffer
                            CoreOp.VarAccessOp.VarLoadOp newVarLoad = CoreOp.VarAccessOp.varLoad(blockBuilder.context().getValue(replaced.get(r)));
                            replacement = blockBuilder.op(copyLocation(varLoadOp,newVarLoad));
                            context.mapValue(varLoadOp.result(), replacement);
                        }
                        replaced.put(varLoadOp.result(), replacement);
                        return blockBuilder;
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                    if (isBufferArray(lookup, arrayLoadOp)) {
                        Op replacementOp=null;
                        if (isVectorOp(lookup,arrayLoadOp)) {
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
                        context.mapValue(arrayLoadOp.result(), blockBuilder.op(copyLocation(arrayLoadOp,replacementOp)));
                        return blockBuilder;
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                    if (isBufferArray(lookup, arrayStoreOp)) {
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
                        context.mapValue(arrayStoreOp.result(), blockBuilder.op(copyLocation(arrayStoreOp, replacementOp)));
                        return blockBuilder;
                    }
                }
                case JavaOp.ArrayLengthOp arrayLengthOp  when
                        isBufferArray(lookup, arrayLengthOp) && resultFromFirstOperandOrThrow(arrayLengthOp) != null ->{
                    var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                    var hatPtrLengthOp = new HATPtrOp.HATPtrLengthOp(
                            arrayAccessInfo.bufferName(),
                            arrayLengthOp.resultType(),
                            (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup, (ClassType) arrayAccessInfo.buffer().type()),
                            context.getValues(List.of(arrayAccessInfo.buffer()))
                    );
                    context.mapValue(arrayLengthOp.result(), blockBuilder.op(copyLocation(arrayLengthOp,hatPtrLengthOp)));
                    return blockBuilder;
                }
                default -> {
                }
            }
            blockBuilder.op(op);
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
