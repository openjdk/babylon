/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import hat.types.Tensor;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreOp.VarAccessOp.VarLoadOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SequencedSet;
import java.util.Set;
import java.util.stream.Collectors;

import static jdk.incubator.code.dialect.core.CoreOp.varLoad;
import static jdk.incubator.code.dialect.java.JavaType.FLOAT;
import static jdk.incubator.code.dialect.java.JavaType.VOID;

public record HATTensorsPhase() implements HATPhase {

    private interface TensorTransformer {
        void transform(CoreOp.FuncOp funcOp, Block.Builder blockBuilder, Op op, VarTable varTable);
    }

    private static class TensorView implements TensorTransformer {

        @Override
        public void transform(CoreOp.FuncOp funcOp, Block.Builder blockBuilder, Op op, VarTable varTable) {
            if (Objects.requireNonNull(op) instanceof CoreOp.VarOp varOp) {
                Op.Result opResult = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(funcOp.funcName(), opResult.op(), VarTable.HATOpAttribute.TENSOR);
            } else {
                blockBuilder.add(op);
            }
        }
    }

    private CoreOp.FuncOp transformWithPredicate(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, TensorTransformer function, Set<Op> opsToProcess, VarTable varTable) {
        return Trxfmr.of(lookup, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            function.transform(funcOp, blockBuilder, op, varTable);
            return blockBuilder;
        }, varTable).funcOp();
    }

    private record DeclTensorData(Op marker, JavaOp.InvokeOp invokeOp, CoreOp.VarOp varOp) {
    }

    record ControlFlowLastOp(CodeElement<?, ?> previous, boolean hitControlFlow) {}

    private ControlFlowLastOp obtainLastOpBeforeControlFlow(CoreOp.FuncOp funcOp) {
        CodeElement<?, ?> previousOp = funcOp.bodies().getFirst().blocks().getFirst().firstOp();
        List<CodeElement<?, ?>> elements = funcOp.elements().toList();
        boolean hitControlFlow = false;
        for (CodeElement<?, ?> element : elements) {
            if (Objects.requireNonNull(element) instanceof Op.Loop
                    || element instanceof JavaOp.IfOp
                    || element instanceof JavaOp.SwitchExpressionOp) {
                hitControlFlow = true;
                break;
            }
            previousOp = element;
        }

        if (!hitControlFlow) {
            // We do not have any control-flow flow, we point to the first instruction instead.
            previousOp = funcOp.bodies().getFirst().blocks().getFirst().firstOp();
        }
        return new ControlFlowLastOp(previousOp, hitControlFlow);
    }

    public static class TensorMarkers {
        public static Tensor create() {
            return null;
        }

        private TensorMarkers() {}
    }

    private static final MethodRef CREATE_FUNCTION = MethodRef.method(TensorMarkers.class, "create", Tensor.class);
    private static final MethodRef FILL_FUNCTION = MethodRef.method(Tensor.class, "fill", void.class, Tensor.class, float.class);
    private static final MethodRef MMA_FUNCTION = MethodRef.method(Tensor.class, "mma", Tensor.class, Tensor.class, Tensor.class, Tensor.class);

    private void appendTensorDeclarationToBlock0(List<DeclTensorData> declTensorList, Block.Builder blockBuilder, Map<CoreOp.VarOp, Value> mapValueTensor, VarTable varTable, String functionName) {
        // And add the missing declarations
        for (DeclTensorData c : declTensorList) {
            // Add a TensorCreateOp
            JavaOp.InvokeOp declInvoke = c.invokeOp;
            CoreOp.VarOp declVar = c.varOp;
            JavaOp.InvokeOp invoke = JavaOp.invoke(CREATE_FUNCTION, List.of());
            Op.Result op1 = blockBuilder.add(invoke);

            // Add a TensorVarOp associated with teh TensorCreateOp
            CoreOp.VarOp varOp = CoreOp.var(declVar.varName(), op1);
            Op.Result op2 = blockBuilder.add(varOp);
            varTable.addIfNeededOrThrow(functionName, op2.op(), VarTable.HATOpAttribute.TENSOR);

            // Include in a new HashMap the new tensorVarOp to be propagated for the Stores and VarLoadOps.
            mapValueTensor.put(declVar, op2);

            blockBuilder.context().mapValue(declInvoke.result(), op2);
        }
    }

    private CoreOp.FuncOp transformWithRelocate(CoreOp.FuncOp funcOp, VarTable varTable, Set<Op> opsToProcess, Map<Op, List<DeclTensorData>> markerTable, Op markerOp) {
        // 3. Transform the model to insert:
        // 3.1: All tensor declaration from the marker.
        // 3.2 The load-invoke keeps intact since it will be processed in another phase
        // 3.3 Replace the VarOp declaration associated with a load with a TensorStoreLoadOp
        // 3.4 Replace the reference of subsequent VarLoapOp with the new declaration
        Map<CoreOp.VarOp, Value> mapValueTensor = new HashMap<>();
        Map<Op, Value> mapUsages = new HashMap<>();
        Map<Op, Value> invokeMap = new HashMap<>();
        final String functionName = funcOp.funcName();
        CoreOp.FuncOp finalFuncOp = funcOp;
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!opsToProcess.contains(op)) {
                Op.Result opNew = blockBuilder.add(op);
                varTable.passthrough(functionName, op, opNew.op());
            } else if (markerTable.containsKey(op)) {
                // In this block, we insert all pending tensor declaration, starting with the marker.

                List<DeclTensorData> declTensorList = markerTable.get(markerOp);
                // Insert the marker
                blockBuilder.add(markerOp);
                //appendTensorDeclarationToBlock0(declTensorList, blockBuilder, mapValueTensor, varTable, finalFuncOp.funcName());
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // The Load/LoadF16 is propagated

                List<DeclTensorData> declTensorList = markerTable.get(markerOp);
                JavaOp.InvokeOp declInvoke = invokeOp;

                if (invokeOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp oldVarOp) {

                    CoreOp.VarOp declVar = oldVarOp;
                    JavaOp.InvokeOp invoke = JavaOp.invoke(CREATE_FUNCTION, List.of());
                    Op.Result op1 = blockBuilder.add(invoke);

                    // Add a TensorVarOp associated with teh TensorCreateOp
                    CoreOp.VarOp varOp = CoreOp.var(declVar.varName(), op1);
                    Op.Result op2 = blockBuilder.add(varOp);
                    varTable.addIfNeededOrThrow(functionName, op2.op(), VarTable.HATOpAttribute.TENSOR);

                    // Include in a new HashMap the new tensorVarOp to be propagated for the Stores and VarLoadOps.
                    mapValueTensor.put(declVar, op2);

                    blockBuilder.context().mapValue(declInvoke.result(), op2);
                } else {
                    throw new IllegalStateException(" Error: found " + invokeOp.result().uses().getFirst().declaringElement().getClass());
                }

                Op.Result invokeResult = blockBuilder.add(invokeOp);
                invokeMap.put(invokeOp, invokeResult);

            } else if (op instanceof CoreOp.VarOp varOp) {
                // Replace the VarOp with a TensorStoreLoadOp using the reference of the tensorVarOp
                // declared in a previous block (block 0)
                Value tensorVarOp = mapValueTensor.get(varOp);

                // Update the usages
                SequencedSet<Op.Result> uses = varOp.result().uses();
                for (Op.Result r : uses) {
                    opsToProcess.add(r.op());
                    mapUsages.put(r.op(), tensorVarOp);
                }

                Value v;
                if (varOp.operands().getFirst().declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
                    v = invokeMap.get(invokeOp);
                } else {
                    throw new IllegalStateException("Expected an invokeOp");
                }

                CoreOp.VarAccessOp.VarStoreOp varStoreOp = CoreOp.varStore(tensorVarOp, v);
                Op.Result storeResult = blockBuilder.add(varStoreOp);
                blockBuilder.context().mapValue(varOp.result(), storeResult);

            } else if (op instanceof VarLoadOp varLoadOp) {
                // This means a tensor is loading, and we expect the varLoadOp to be present already in the mapUsages,
                // since the declaration is always traversed before teh varLoadOp.
                Value tensorValue = mapUsages.get(varLoadOp);
                VarLoadOp varLoadOpNew = varLoad(tensorValue);
                Op.Result op1 = blockBuilder.add(varLoadOpNew);
                blockBuilder.context().mapValue(varLoadOp.result(), op1);
            }
            return blockBuilder;
        });
        return funcOp;
    }

    private CoreOp.FuncOp createTensorsToRelocate(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {

        // 1. Obtain the last Op before any control flow. We need to find a suitable location in basic block 0
        // to insert the declaration of tensors coming from the Load operations
        ControlFlowLastOp controlFlowLastOp = obtainLastOpBeforeControlFlow(funcOp);

        // 2. Analyze the code model to obtain:
        // 2.1: A set for all nodes to be processed (load-invoke, and the varOp associated with it)
        // 2.2: A map that relates the marker position (suitable last op) with a list of pending declaration to perform
        Map<Op, List<DeclTensorData>> markerTable  = new HashMap<>();
        Op markerOp = (Op) controlFlowLastOp.previous;
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("load") || invoke.name().equals("loadF16"))
                .forEach(invoke -> {
                    markerTable.putIfAbsent(markerOp, new ArrayList<>());
                    opsToProcess.add(markerOp);
                    opsToProcess.add(invoke.op());
                    invoke.op().result().uses().stream()
                            .filter(result -> (result.op() instanceof CoreOp.VarOp))
                            .map(result -> (CoreOp.VarOp) result.op())
                            .forEach(varOp -> {
                                opsToProcess.add(varOp);
                                markerTable.get(markerOp).add(new DeclTensorData(markerOp, invoke.op(), varOp));
                            });
                });
        return transformWithRelocate(funcOp, varTable, opsToProcess, markerTable, markerOp);
    }

    private CoreOp.FuncOp createTensors(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("create") || invoke.name().equals("of"))
                .forEach( invoke ->
                        invoke.op().result().uses().stream()
                        .filter(result -> (result.op() instanceof CoreOp.VarOp))
                        .map(result -> (CoreOp.VarOp) result.op())
                        .findFirst()
                        .ifPresent(opsToProcess::add));

        return transformWithPredicate(lookup, funcOp, new TensorView(), opsToProcess, varTable);
    }

    private CoreOp.FuncOp tensorShape(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("shape"))
                .forEach( invoke ->
                        invoke.op().result().uses().stream()
                                .filter(result -> (result.op() instanceof CoreOp.VarOp))
                                .map(result -> (CoreOp.VarOp) result.op())
                                .findFirst()
                                .ifPresent(opsToProcess::add));

        CoreOp.FuncOp finalFuncOp = funcOp;
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!opsToProcess.contains(op)) {
                Op.Result opNew = blockBuilder.add(op);
                varTable.passthrough(finalFuncOp.funcName(), op, opNew.op());
            } else if (op instanceof CoreOp.VarOp varOp) {
                Op.Result tensorShape = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(finalFuncOp.funcName(), tensorShape.op(), VarTable.HATOpAttribute.TENSOR_SHAPE);
            }
            return blockBuilder;
        });
        return funcOp;

    }

    private CoreOp.FuncOp zerosTensors(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = new HashSet<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("zeros"))
                .forEach(invoke -> {
                    opsToProcess.add(invoke.op());
                    Value varValue = invoke.op().result().uses().getFirst();
                    if (varValue.declaringElement() instanceof CoreOp.VarOp varOp) {
                        opsToProcess.add(varOp);
                    }
                });

        Map<Op, Value> map = new HashMap<>();
        CoreOp.FuncOp finalFuncOp = funcOp;
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!opsToProcess.contains(op)) {
                Op.Result opNew = blockBuilder.add(op);
                varTable.passthrough(finalFuncOp.funcName(), op, opNew.op());
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Create Tensor
                Op.Result op1 = blockBuilder.add(invokeOp);
                Op.Result valueVar = invokeOp.result().uses().getFirst();
                if (valueVar.declaringElement() instanceof CoreOp.VarOp varOp) {
                    // Add Var
                    CoreOp.VarOp varOp1 = CoreOp.var(varOp.varName(), op1);
                    Op.Result op2 = blockBuilder.add(varOp1);
                    varTable.addIfNeededOrThrow(finalFuncOp.funcName(), op2.op(), VarTable.HATOpAttribute.TENSOR);

                    // Add VarLoadOp
                    VarLoadOp varLoadOp = varLoad(op2);
                    Op.Result op3 = blockBuilder.add(varLoadOp);

                    // Add Fill Invoke
                    CoreOp.ConstantOp constant = CoreOp.constant(FLOAT, 0.0f);
                    Op.Result op4 = blockBuilder.add(constant);

                    List<Value> argsFill = List.of(op3, op4);
                    JavaOp.InvokeOp fillInvoke = JavaOp.invoke(VOID, FILL_FUNCTION, argsFill);
                    Op.Result op5 = blockBuilder.add(fillInvoke);
                    map.put(varOp, op2);
                    blockBuilder.context().mapValue(invokeOp.result(), op5);
                } else {
                    throw new IllegalStateException("Expected a VarOp");
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                // Pass through value using the TensorVarOp created before
                blockBuilder.context().mapValue(varOp.result(), map.get(varOp));
            }
            return blockBuilder;
        });
        return funcOp;
    }

    private CoreOp.FuncOp mmaTensorWithStore(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid())
                .filter(invoke -> invoke.refIs(Tensor.class))
                .filter(invoke -> invoke.name().equals("mma"))
                .map(OpHelper::op)
                .collect(Collectors.toSet());

        CoreOp.FuncOp finalFuncOp = funcOp;
        funcOp = funcOp.transform((blockBuilder, op) -> {
            if (!opsToProcess.contains(op)) {
                Op.Result opNew = blockBuilder.add(op);
                varTable.passthrough(finalFuncOp.funcName(), op, opNew.op());
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                Op.Result result = invokeOp.result().uses().getFirst();
                if (result.declaringElement() instanceof CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
                    // insert a VarLodOp for the result operand that was already declared
                    List<Value> args = new ArrayList<>();
                    args.add(varStoreOp.operands().getFirst());
                    List<Value> operands = blockBuilder.context().getValues(args);
                    VarLoadOp varLoadOp = varLoad(operands.getFirst());
                    Op.Result op1 = blockBuilder.add(varLoadOp);

                    args.clear();
                    args.add(op1);
                    args.addAll(blockBuilder.context().getValues(invokeOp.operands()));

                    // insert an MMA Invoke
                    JavaOp.InvokeOp tensorMMAOp = JavaOp.invoke(MMA_FUNCTION, args);
                    tensorMMAOp.setLocation(invokeOp.location());
                    Op.Result op2 = blockBuilder.add(tensorMMAOp);
                    blockBuilder.context().mapValue(invokeOp.result(), op2);
                } else {
                    throw new IllegalStateException("Expected a VarStoreOp, but found " + result.op());
                }
            }
            return blockBuilder;
        });
        return funcOp;
    }

    @FunctionalInterface
    private interface ActionTensorTransformer {
        CoreOp.FuncOp apply(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable);
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        List<ActionTensorTransformer> tensorTransformer = List.of(
                this::createTensorsToRelocate,
                this::createTensors,
                this::tensorShape,
                this::zerosTensors,
                this::mmaTensorWithStore);
        CoreOp.FuncOp[] function =  new CoreOp.FuncOp[] {funcOp};
        tensorTransformer.forEach(action ->
                function[0] = action.apply(lookup, function[0], varTable)
        );
        return function[0];
    }
}
