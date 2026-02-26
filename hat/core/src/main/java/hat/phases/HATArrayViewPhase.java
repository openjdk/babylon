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
import hat.dialect.*;
import optkl.OpHelper;
import optkl.Trxfmr;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.*;
import optkl.util.ops.VarLikeOp;

import java.util.*;

import static hat.phases.HATPhaseUtils.getVectorShape;
import static optkl.OpHelper.*;
import static optkl.OpHelper.Invoke.invoke;

public record HATArrayViewPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (Invoke.stream(lookup(), funcOp).noneMatch(
                invoke -> HATPhaseUtils.isBufferArray(lookup(), invoke.op())
        )) return funcOp;

        funcOp = applyArrayView(funcOp);

        if (funcOp.elements().filter(e -> e instanceof CoreOp.VarOp).anyMatch(
                e -> HATPhaseUtils.isVectorOp(lookup(), ((CoreOp.VarOp) e))
        )) funcOp = applyVectorView(funcOp);
        return funcOp;
    }

    public CoreOp.FuncOp applyVectorView(CoreOp.FuncOp funcOp) {
        return Trxfmr.of(this,funcOp).transform((blockBuilder, op) -> {
            var context = blockBuilder.context();
            switch (op) {
                case JavaOp.InvokeOp $ when invoke(lookup(), $) instanceof Invoke invoke -> {
                    if (HATPhaseUtils.isVectorBinaryOp(invoke.lookup(), invoke)){
                        var hatVectorBinaryOp = HATPhaseUtils.buildVectorBinaryOp(
                                invoke.varOpFromFirstUseOrThrow().varName(),
                                invoke.name(),// so mul, sub etc
                                getVectorShape(lookup(),invoke.returnType()),
                                blockBuilder.context().getValues(invoke.op().operands())
                        );
                        context.mapValue(invoke.returnResult(), blockBuilder.op(copyLocation(invoke.op(),hatVectorBinaryOp)));
                        return blockBuilder;
                    }
                }
                case CoreOp.VarOp varOp -> {
                    if (HATPhaseUtils.isVectorOp(lookup(),varOp)) {
                        var hatVectorVarOp = new HATVectorOp.HATVectorVarOp(
                                varOp.varName(),
                                varOp.resultType(),
                                getVectorShape(lookup(),varOp.resultType().valueType()),
                                context.getValues(OpHelper.firstOperandAsListOrEmpty(varOp))
                        );
                        context.mapValue(varOp.result(), blockBuilder.op(copyLocation(varOp,hatVectorVarOp)));
                        return blockBuilder;
                    }
                }
                case JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp -> {
                    if (HATPhaseUtils.isVectorOp(lookup(),arrayLoadOp)) {
                        Op.Result buffer = resultFromFirstOperandOrNull(arrayLoadOp);
                        String name = hatPtrName(opFromFirstOperandOrThrow(buffer.op()));
                        var  vectorShape = getVectorShape(lookup(),arrayLoadOp.resultType());
                        HATVectorOp.HATVectorLoadOp vLoadOp = HATPhaseUtils.isLocalSharedOrPrivate(arrayLoadOp)
                                ? new HATVectorOp.HATVectorLoadOp.HATSharedVectorLoadOp(
                                name,
                                CoreType.varType(arrayLoadOp.resultType()),
                                vectorShape,
                                context.getValues(List.of(buffer, arrayLoadOp.operands().getLast())))
                                :new HATVectorOp.HATVectorLoadOp.HATPrivateVectorLoadOp(
                                name,
                                CoreType.varType(arrayLoadOp.resultType()),
                                vectorShape,
                                context.getValues(List.of(buffer, arrayLoadOp.operands().getLast()))
                        );
                        context.mapValue(arrayLoadOp.result(), blockBuilder.op(copyLocation(arrayLoadOp,vLoadOp)));
                    }
                    return blockBuilder;
                }
                case JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp -> {
                    if (HATPhaseUtils.isVectorOp(lookup(),arrayStoreOp)) {
                        Op.Result buffer = resultFromFirstOperandOrThrow(arrayStoreOp);
                        Op varOp =
                                HATPhaseUtils.findOpInResultFromFirstOperandsOrThrow(((Op.Result) arrayStoreOp.operands().getLast()).op(), CoreOp.VarOp.class, HATVectorOp.HATVectorVarOp.class);
                        var name = hatPtrName(varOp);
                        var resultType = (varOp instanceof HATVectorOp.HATVectorVarOp)
                                ? (varOp).resultType()
                                : ((CoreOp.VarOp) varOp).resultType();
                        var vectorShape = getVectorShape(lookup(),arrayStoreOp.operands().getLast().type());
                        HATVectorOp.HATVectorStoreView vStoreOp = HATPhaseUtils.isLocalSharedOrPrivate(arrayStoreOp)
                                ? new HATVectorOp.HATVectorStoreView.HATSharedVectorStoreView(
                                name,
                                resultType,
                                vectorShape,
                                context.getValues(List.of(buffer, arrayStoreOp.operands().getLast(), arrayStoreOp.operands().get(1))))
                                : new HATVectorOp.HATVectorStoreView.HATPrivateVectorStoreView(
                                name,
                                resultType,
                                vectorShape,
                                context.getValues(List.of(buffer, arrayStoreOp.operands().getLast(), arrayStoreOp.operands().get(1)))
                        );
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

    public CoreOp.FuncOp applyArrayView(CoreOp.FuncOp funcOp) {
        Map<Op.Result, Op.Result> replaced = new HashMap<>(); // maps a result to the result it should be replaced by
        Map<Op, CoreOp.VarAccessOp.VarLoadOp> bufferVarLoads = new HashMap<>();

        return Trxfmr.of(this,funcOp).transform((blockBuilder, op) -> {
            var context = blockBuilder.context();
            switch (op) {
                case JavaOp.InvokeOp $ when invoke(lookup(), $) instanceof Invoke invoke -> {
                    if (HATPhaseUtils.isBufferArray(invoke.lookup(), invoke.op())) { // ensures we can use iop as key for replaced vvv
                        Op.Result result = invoke.resultFromFirstOperandOrNull();
                        replaced.put(invoke.returnResult(), result);
                        // map buffer VarOp to its corresponding VarLoadOp
                        bufferVarLoads.put((opFromFirstOperandOrNull(result.op())), (CoreOp.VarAccessOp.VarLoadOp) result.op());
                        return blockBuilder;
                    }
                }
                case CoreOp.VarOp varOp -> {
                    if (HATPhaseUtils.isBufferInitialize(lookup(), varOp)) {
                        // makes sure we don't process a new int[] for example
                        Op bufferLoad = replaced.get(resultFromFirstOperandOrThrow(varOp)).op(); // gets VarLoadOp associated w/ og buffer
                        replaced.put(varOp.result(), resultFromFirstOperandOrNull(bufferLoad)); // gets VarOp associated w/ og buffer
                        return blockBuilder;
                    }
                }
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                    if ((HATPhaseUtils.isBufferInitialize(lookup(), varLoadOp))) {
                        Op.Result r = resultFromFirstOperandOrThrow(varLoadOp);
                        Op.Result replacement;
                        if (r.op() instanceof CoreOp.VarOp) { // if this is the VarLoadOp after the .arrayView() InvokeOp
                            replacement = (HATPhaseUtils.isLocalSharedOrPrivate(varLoadOp)) ?
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
                    if (HATPhaseUtils.isBufferArray(lookup(), arrayLoadOp)) {
                        Op replacementOp;
                        if (HATPhaseUtils.isVectorOp(lookup(),arrayLoadOp)) {
                            replacementOp = JavaOp.arrayLoadOp(
                                    context.getValue(replaced.get((Op.Result) arrayLoadOp.operands().getFirst())),
                                    context.getValue(arrayLoadOp.operands().getLast()),
                                    arrayLoadOp.resultType()
                            );
                        } else if (((ArrayType) firstOperandOrThrow(op).type()).dimensions() == 1) {
                            var arrayAccessInfo = HATPhaseUtils.arrayAccessInfo(op.result(), replaced);
                            var operands = arrayAccessInfo.bufferAndIndicesAsValues();
                            replacementOp = new HATPtrOp.HATPtrLoadOp(
                                    arrayAccessInfo.bufferName(),
                                    arrayLoadOp.resultType(),
                                    (Class<?>) classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
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
                    if (HATPhaseUtils.isBufferArray(lookup(), arrayStoreOp)) {
                        Op replacementOp;
                        if (HATPhaseUtils.isVectorOp(lookup(), arrayStoreOp)) {
                            replacementOp = JavaOp.arrayStoreOp(
                                    context.getValue(replaced.get((Op.Result) arrayStoreOp.operands().getFirst())),
                                    context.getValue(arrayStoreOp.operands().get(1)),
                                    context.getValue(arrayStoreOp.operands().getLast())
                            );
                        } else if (((ArrayType) firstOperandOrThrow(op).type()).dimensions() == 1) { // we only use the last array load
                            var arrayAccessInfo = HATPhaseUtils.arrayAccessInfo(op.result(), replaced);
                            var operands = arrayAccessInfo.bufferAndIndicesAsValues();
                            operands.add(arrayStoreOp.operands().getLast());
                            replacementOp = new HATPtrOp.HATPtrStoreOp(
                                    arrayAccessInfo.bufferName(),
                                    arrayStoreOp.resultType(),
                                    (Class<?>) classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
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
                        HATPhaseUtils.isBufferArray(lookup(), arrayLengthOp) && resultFromFirstOperandOrThrow(arrayLengthOp) != null ->{
                    var arrayAccessInfo = HATPhaseUtils.arrayAccessInfo(op.result(), replaced);
                    var hatPtrLengthOp = new HATPtrOp.HATPtrLengthOp(
                            arrayAccessInfo.bufferName(),
                            arrayLengthOp.resultType(),
                            (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup(), (ClassType) arrayAccessInfo.buffer().type()),
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
                            : resultFromFirstOperandOrThrow(res.op()));
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
