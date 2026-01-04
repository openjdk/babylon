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

import hat.callgraph.KernelCallGraph;
import hat.device.DeviceType;
import hat.dialect.*;
import optkl.Invoke;
import optkl.Trxfmr;
import optkl.util.CallSite;
import optkl.ifacemapper.MappableIface;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.*;

import java.util.*;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.classTypeToTypeOrThrow;
import static optkl.Trxfmr.copyLocation;

public record HATArrayViewPhase(KernelCallGraph kernelCallGraph) implements HATPhase {


    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (Invoke.stream(lookup(), funcOp).anyMatch(invoke ->
                            invoke.returnsArray()
                        && invoke.refIs(MappableIface.class,DeviceType.class))) {
            Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
            Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

            return new Trxfmr(funcOp).transform( (blockBuilder, op) -> {
                var context = blockBuilder.context();
                switch (op) {
                    case JavaOp.InvokeOp i when invokeOpHelper(lookup(), i) instanceof Invoke invoke -> {
                        if (isHatVectorBinaryOperation(invoke)) {
                            // catching HATVectorBinaryOps not stored in VarOps
                            var hatVectorBinaryOp = copyLocation(invoke.op(), buildVectorBinaryOp(
                                    invoke.name(),
                                    varNameFromInvokeFirstUse(invoke),
                                    invoke.returnType(),
                                    blockBuilder.context().getValues(invoke.op().operands())
                            ));
                            Op.Result binaryResult = blockBuilder.op(hatVectorBinaryOp);
                           context.mapValue(invoke.returnResult(), binaryResult);
                            replaced.put(invoke.returnResult(), binaryResult);
                            return blockBuilder;
                        } else if (isBufferArray(invoke.op()) && invoke.firstOperandAsResultOrNull() instanceof Op.Result result) { // ensures we can use iop as key for replaced vvv
                            replaced.put(invoke.returnResult(), result);
                            // map buffer VarOp to its corresponding VarLoadOp
                            bufferVarLoads.put((firstOperandAsResultOrNull(result.op())).op(), (CoreOp.VarAccessOp.VarLoadOp) result.op());
                            return blockBuilder;
                        } else{
                            // or else
                        }
                    }
                    case CoreOp.VarOp varOp -> {
                        if (isBufferInitialize(varOp) && firstOperand(varOp) instanceof Op.Result result) { // makes sure we don't process a new int[] for example
                            Op bufferLoad = replaced.get(result).op(); // gets VarLoadOp associated w/ og buffer
                            replaced.put(varOp.result(), firstOperandAsResultOrNull(bufferLoad)); // gets VarOp associated w/ og buffer
                            return blockBuilder;
                        } else if (isVectorOp(varOp)) {
                            List<Value> operands = (varOp.operands().isEmpty()) ? List.of() : List.of(firstOperand(varOp));
                            var vectorMetaData = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(lookup(),varOp.resultType().valueType());
                            var hatVectorVarOp = copyLocation(varOp,new HATVectorOp.HATVectorVarOp(
                                    varOp.varName(),
                                    varOp.resultType(),
                                    vectorMetaData.vectorTypeElement(),
                                    vectorMetaData.lanes(),
                                   context.getValues(operands)
                            ));
                            context.mapValue(varOp.result(), blockBuilder.op(hatVectorVarOp));
                            return blockBuilder;
                        }else{
                            // or else
                        }
                    }
                    case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                        if ((isBufferInitialize(varLoadOp)) && firstOperand(varLoadOp) instanceof Op.Result r) {
                            if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                                Op.Result replacement = (notGlobalVarOp(varLoadOp)) ?
                                        firstOperandAsResultOrNull((firstOperandAsResultOrNull(r.op())).op()) :
                                        bufferVarLoads.get(replaced.get(r).op()).result();
                                replaced.put(varLoadOp.result(), replacement);
                            } else { // if this is a VarLoadOp loading the buffer
                                // is this not just bb.op(varLoadOp)?
                                CoreOp.VarAccessOp.VarLoadOp newVarLoad = copyLocation(varLoadOp,
                                        CoreOp.VarAccessOp.varLoad(getValue(blockBuilder, replaced.get(r))));
                                Op.Result res = blockBuilder.op(newVarLoad);
                                context.mapValue(varLoadOp.result(), res);
                                replaced.put(varLoadOp.result(), res);
                            }
                            return blockBuilder;
                        }else{
                            // or else
                        }
                    }
                    case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                        if (isBufferArray(arrayLoadOp) && firstOperand(arrayLoadOp) instanceof Op.Result r) {
                            Op.Result buffer = replaced.getOrDefault(r, r);
                            if (isVectorOp(arrayLoadOp)) {
                                Op vop = (firstOperandAsResultOrNull(buffer.op())).op();
                                String name = switch (vop) {
                                    case CoreOp.VarOp varOp -> varOp.varName();
                                    case HATMemoryVarOp.HATLocalVarOp hatLocalVarOp -> hatLocalVarOp.varName();
                                    case HATMemoryVarOp.HATPrivateVarOp hatPrivateVarOp -> hatPrivateVarOp.varName();
                                    default -> throw new IllegalStateException("Unexpected value: " + vop);
                                };
                                var  hatVectorMetaData = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(lookup(),arrayLoadOp.resultType());
                                HATVectorOp.HATVectorLoadOp vLoadOp = copyLocation(arrayLoadOp,new HATVectorOp.HATVectorLoadOp(
                                        name,
                                        CoreType.varType(((ArrayType) firstOperand(arrayLoadOp).type()).componentType()),
                                        hatVectorMetaData.vectorTypeElement(),
                                        hatVectorMetaData.lanes(),
                                        notGlobalVarOp(arrayLoadOp),
                                        context.getValues(List.of(buffer, arrayLoadOp.operands().getLast()))
                                ));
                                context.mapValue(arrayLoadOp.result(), blockBuilder.op(vLoadOp));
                            } else if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we only use the last array load
                                var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                                var operands = arrayAccessInfo.bufferAndIndicesAsValues();

                                var hatPtrLoadOp = copyLocation(arrayLoadOp,new HATPtrOp.HATPtrLoadOp(
                                        arrayLoadOp.resultType(),
                                        (Class<?>) classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
                                        context.getValues(operands)
                                ));
                                context.mapValue(arrayLoadOp.result(), blockBuilder.op(hatPtrLoadOp));
                            }
                        } else {
                            // or else?
                        }
                        return blockBuilder;
                    }
                    case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                        if (isBufferArray(arrayStoreOp) && firstOperand(arrayStoreOp) instanceof Op.Result r) {
                            Op.Result buffer = replaced.getOrDefault(r, r);
                            if (isVectorOp(arrayStoreOp)) {
                                Op varOp = findVarOpOrHATVarOP(((Op.Result) arrayStoreOp.operands().getLast()).op());
                                var name = (varOp instanceof HATVectorOp.HATVectorVarOp)
                                        ? ((HATVectorOp.HATVectorVarOp) varOp).varName()
                                        : ((CoreOp.VarOp) varOp).varName();
                                var resultType = (varOp instanceof HATVectorOp.HATVectorVarOp)
                                        ? (varOp).resultType()
                                        : ((CoreOp.VarOp) varOp).resultType();
                                var classType = ((ClassType) ((ArrayType) firstOperand(arrayStoreOp).type()).componentType());
                                var vectorMetaData = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(lookup(),classType);
                                HATVectorOp.HATVectorStoreView vStoreOp = copyLocation(arrayStoreOp,new HATVectorOp.HATVectorStoreView(
                                        name,
                                        resultType,
                                        vectorMetaData.lanes(),
                                        vectorMetaData.vectorTypeElement(),
                                        notGlobalVarOp(arrayStoreOp),
                                        context.getValues(List.of(buffer, arrayStoreOp.operands().getLast(), arrayStoreOp.operands().get(1)))
                                ));
                                context.mapValue(arrayStoreOp.result(), blockBuilder.op(vStoreOp));
                            } else if (((ArrayType) firstOperand(op).type()).dimensions() == 1) { // we only use the last array load
                                var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                                var operands = arrayAccessInfo.bufferAndIndicesAsValues();
                                operands.add(arrayStoreOp.operands().getLast());
                                HATPtrOp.HATPtrStoreOp ptrLoadOp = copyLocation(arrayStoreOp,new HATPtrOp.HATPtrStoreOp(
                                        arrayStoreOp.resultType(),
                                        (Class<?>) classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
                                        context.getValues(operands)
                                ));
                                context.mapValue(arrayStoreOp.result(), blockBuilder.op(ptrLoadOp));
                            }else{
                                // or else
                            }
                        }else{
                            // or else?
                        }
                        return blockBuilder;
                    }
                    case JavaOp.ArrayLengthOp arrayLengthOp -> {
                        if (isBufferArray(arrayLengthOp) && firstOperand(arrayLengthOp) instanceof Op.Result) {
                            var arrayAccessInfo = arrayAccessInfo(op.result(), replaced);
                            var hatPtrLengthOp = copyLocation(arrayLengthOp,new HATPtrOp.HATPtrLengthOp(
                                    arrayLengthOp.resultType(),
                                    (Class<?>) classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
                                    context.getValues(List.of(arrayAccessInfo.buffer()))
                            ));
                            context.mapValue(arrayLengthOp.result(), blockBuilder.op(hatPtrLengthOp));
                            return blockBuilder;
                        }else{
                            // or else
                        }
                    }
                    default -> {
                    }
                }
                blockBuilder.op(op);
                return blockBuilder;
            }).funcOp();
        }else {
            return funcOp;
        }
    }

    record ArrayAccessInfo(Op.Result buffer, List<Op.Result> indices) {
        public List<Value> bufferAndIndicesAsValues() {
            List<Value> operands = new ArrayList<>(List.of(buffer));
            operands.addAll(indices);
            return operands;
        }
    };

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
                        indices.addFirst(res.op() instanceof JavaOp.ArrayAccessOp ? ((Op.Result) res.op().operands().get(1)) : ((Op.Result) res.op().operands().get(0)));

                }
                // I think we need to comment this.  Not so obvious.
                if (!node.edges().isEmpty()) {
                    Node<T> next = node.edges().getFirst();
                    if (!handled.contains(next)) {
                        nodeList.add(next);
                    }
                }
            }
            buffer = replaced.get((Op.Result) firstOperand(buffer.op()));
            return new ArrayAccessInfo(buffer, indices);
        }
    }

    static ArrayAccessInfo arrayAccessInfo(Value value, Map<Op.Result, Op.Result> replaced) {
        return expressionGraph(value).getInfo(replaced);
    }

    static Node<Value> expressionGraph(Value value) {
        return expressionGraph(new HashMap<>(), value);
    }

    static Node<Value> expressionGraph(Map<Value, Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the expression graphs for each operand
        List<Node<Value>> edges = new ArrayList<>();

        // looks like
        for (Value operand : value.dependsOn()) {
            if (operand instanceof Op.Result res &&
                    res.op() instanceof JavaOp.InvokeOp iop
                    && iop.invokeDescriptor().name().toLowerCase().contains("arrayview")){
                continue;
            }
            edges.add(expressionGraph(operand));
        }
        Node<Value> node = new Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    /*
     * Helper functions:
     */

    private HATVectorOp.HATVectorBinaryOp buildVectorBinaryOp(String opType, String varName, TypeElement resultType, List<Value> outputOperands) {
        HATPhaseUtils.VectorMetaData md = HATPhaseUtils.getVectorTypeInfoWithCodeReflection(lookup(),resultType);
        return switch (opType) {
            case "add" -> new HATVectorOp.HATVectorBinaryOp.HATVectorAddOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "sub" -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "mul" -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "div" -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + opType);
        };
    }

    private boolean isHatVectorBinaryOperation(Invoke invoke) {
        // no! lets not compare strings what if we refactor the class names?  This is brittle
        return invoke.returnType().toString().startsWith("hat.buffer.Float")
                   && invoke.name().toLowerCase() instanceof String name
                   &&(name.equals("add")|| name.equals("sub")||name.equals("mul")||name.equals("div"));
    }

    private Op findVarOpOrHATVarOP(Op op) {
        return searchForOpOrNull(op, CoreOp.VarOp.class, HATVectorOp.HATVectorVarOp.class);
    }

    public boolean isVectorOp(Op op) {
        if (!op.operands().isEmpty()) {
           TypeElement type = firstOperand(op).type();
           if (type instanceof ArrayType at) {
               type = at.componentType();
           }
           if (type instanceof ClassType ct) {
               try {
                   return _V.class.isAssignableFrom((Class<?>) ct.resolve(lookup()));
               } catch (ReflectiveOperationException e) {
                   throw new RuntimeException(e);
              }
           }
        }
        return false;
    }

    public static Op.Result firstOperandAsResultOrNull(Op op) {
        return (firstOperand(op) instanceof Op.Result result) ? result : null;
    }

    public static Value firstOperand(Op op) {
        return op.operands().getFirst();
    }

    public static Value getValue(Block.Builder bb, Value value) {
        return bb.context().getValueOrDefault(value, value);
    }

    public boolean isBufferArray(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOpOrNull(op, JavaOp.InvokeOp.class);
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview");
    }

    public boolean notGlobalVarOp(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) searchForOpOrNull(op, JavaOp.InvokeOp.class);
        return iop.invokeDescriptor().name().toLowerCase().contains("local") ||
                iop.invokeDescriptor().name().toLowerCase().contains("shared") ||
                iop.invokeDescriptor().name().toLowerCase().contains("private");
    }

    public Op searchForOpOrNull(Op op, Class<?> ...classes) {
        Set<Class<?>> set =Set.of(classes);
        while (!(set.contains(op.getClass()))) {
            if (!op.operands().isEmpty() && firstOperand(op) instanceof Op.Result result) {
                op = result.op();
            } else {
                return null;
            }
        }
        return op;
    }

    public boolean isBufferInitialize(Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop) {
            if (!(vop.varValueType() instanceof ArrayType)){
                return false;
            }
        } else if (!(op instanceof JavaOp.ArrayAccessOp)) {
            if (!(op.resultType() instanceof ArrayType)) {
                return false;
            }
        }
        return isBufferArray(op);
    }

    private String varNameFromInvokeFirstUse(Invoke invoke) {
        var uses= invoke.op().result().uses();
        if (!uses.isEmpty()) {
            Op.Result result = uses.stream().toList().getFirst();
            if (result.op() instanceof CoreOp.VarOp varOp) {
                return varOp.varName();
            }
        }
        return invoke.op().externalizeOpName();
    }
}
