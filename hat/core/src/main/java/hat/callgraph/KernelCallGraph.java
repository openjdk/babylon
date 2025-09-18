/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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
package hat.callgraph;

import hat.BufferTagger;
import hat.buffer.Buffer;
import hat.dialect.HatBarrierOp;
import hat.dialect.HatLocalVarOp;
import hat.dialect.HatMemoryOp;
import hat.dialect.HatPrivateVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.reflect.Method;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public final ComputeCallGraph computeCallGraph;
    public final Map<MethodRef, MethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();
    public final List<BufferTagger.AccessType> bufferAccessList;

    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method, funcOp);
        }
    }

    public static class KernelReachableUnresolvedMethodCall extends UnresolvedMethodCall implements KernelReachable {
        KernelReachableUnresolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }


    public static class KernelReachableUnresolvedIfaceMappedMethodCall extends KernelReachableUnresolvedMethodCall {
        KernelReachableUnresolvedIfaceMappedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class KidAccessor extends MethodCall {
        KidAccessor(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public Stream<KernelReachableResolvedMethodCall> kernelReachableResolvedStream() {
        return methodRefToMethodCallMap.values().stream()
                .filter(call -> call instanceof KernelReachableResolvedMethodCall)
                .map(kernelReachable -> (KernelReachableResolvedMethodCall) kernelReachable);
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, MethodRef methodRef, Method method, CoreOp.FuncOp funcOp) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(null, methodRef, method, funcOp));
        entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        System.out.println("-DbufferTagging="+CallGraph.bufferTagging);
        System.out.println("-DnoModuleOp="+CallGraph.noModuleOp);
        bufferAccessList = CallGraph.bufferTagging?BufferTagger.getAccessList(computeContext.accelerator.lookup, entrypoint.funcOp()):List.of();
    }

    void updateDag(KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {
        /*
         * A ResolvedKernelMethodCall (entrypoint or java  method reachable from a compute entrypojnt)  has the following calls
         * <p>
         * 1) java calls to compute class static functions provided they follow the kernel restrictions
         *    a) we must have the code model available for these and must extend the dag
         * 2) calls to buffer based interface mappings
         *    a) getters (return non void)
         *    b) setters (return void)
         * 3) calls on the NDRange id
         */

        kernelReachableResolvedMethodCall.funcOp().traverse(null, (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
              //  MethodRef methodRef = invokeOp.invokeDescriptor();
                Class<?> javaRefTypeClass = OpTk.javaRefClassOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                Method invokeOpCalledMethod = OpTk.methodOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
                        kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedIfaceMappedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                } else if (entrypoint.method.getDeclaringClass().equals(javaRefTypeClass)) {
                    Optional<CoreOp.FuncOp> optionalFuncOp = Op.ofMethod(invokeOpCalledMethod);
                    if (optionalFuncOp.isPresent()) {
                             kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableResolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod, optionalFuncOp.get()
                                )));
                    } else {
                           kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                        ));
                    }
                } else {
                       kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                    // System.out.println("Were we expecting " + methodRef + " here ");
                }
            }
            return map;
        });

        boolean updated = true;
        kernelReachableResolvedMethodCall.closed = true;
        while (updated) {
            updated = false;
            var unclosed = callStream().filter(m -> !m.closed).findFirst();
            if (unclosed.isPresent()) {
                if (unclosed.get() instanceof KernelReachableResolvedMethodCall reachableResolvedMethodCall) {
                    updateDag(reachableResolvedMethodCall);
                } else {
                    unclosed.get().closed = true;
                }
                updated = true;
            }
        }
    }

    KernelCallGraph close() {
        updateDag(entrypoint);
        // now lets sort the MethodCalls into a dependency list
        calls.forEach(m -> m.rank = 0);
        entrypoint.rankRecurse();
        return this;
    }

    KernelCallGraph closeWithModuleOp() {
        moduleOp = OpTk.createTransitiveInvokeModule(computeContext.accelerator.lookup, entrypoint.funcOp(), this);
        return this;
    }

    @Override
    public boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass) {
        if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
            bufferAccessToMethodCallMap.computeIfAbsent(methodRef, _ ->
                    new KernelReachableUnresolvedIfaceMappedMethodCall(this, methodRef, method)
            );
        } else {
            return false;
        }
        return true;
    }

    private boolean isMethodFromHatKernelContext(JavaOp.InvokeOp invokeOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return invokeOp.invokeDescriptor().refType().toString().equals(kernelContextCanonicalName);
    }

    private boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    private void createBarrierNodeOp(CopyContext context, JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = context.getValues(inputOperands);
        HatBarrierOp hatBarrierOp = new HatBarrierOp(outputOperands);
        Op.Result outputResult = blockBuilder.op(hatBarrierOp);
        Op.Result inputResult = invokeOp.result();
        context.mapValue(inputResult, outputResult);
    }

    public void dialectifyToHatBarriers() {
        CoreOp.FuncOp funcOp = entrypoint.funcOp();
        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                if (isMethodFromHatKernelContext(invokeOp) && isMethod(invokeOp, HatBarrierOp.INTRINSIC_NAME)) {
                    createBarrierNodeOp(context, invokeOp, blockBuilder);
                } else {
                    blockBuilder.op(op);
                }
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });
        // System.out.println("[INFO] Code model: " + funcOp.toText());
        entrypoint.funcOp(funcOp);
    }

    public void dialectifyToHatMemorySpace(Space memorySpace) {

        String nameNode = switch (memorySpace) {
            case PRIVATE -> HatPrivateVarOp.INTRINSIC_NAME;
            case SHARED -> HatLocalVarOp.INTRINSIC_NAME;
        };

        CoreOp.FuncOp funcOp = entrypoint.funcOp();
        //IO.println("ORIGINAL: " + funcOp.toText());
        Stream<CodeElement<?, ?>> elements = entrypoint.funcOp().elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsAdd = varOp.operands();
                        for (Value inputOperand : inputOperandsAdd) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (OpTk.isIfaceBufferMethod(computeContext.accelerator.lookup, invokeOp) && isMethod(invokeOp, nameNode)) {
                                        // It is the node we are looking for
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            // No memory nodes involved
            return;
        }

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Don't insert the invoke node
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        // That's the node we want
                        List<Value> inputOperandsAdd = invokeOp.operands();
                        List<Value> outputOperandsAdd = context.getValues(inputOperandsAdd);
                        HatMemoryOp memoryOp;
                        if (memorySpace == Space.SHARED) {
                            memoryOp = new HatLocalVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsAdd);
                        } else {
                            memoryOp = new HatPrivateVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsAdd);
                        }
                        Op.Result hatLocalResult = blockBuilder.op(memoryOp);
                        context.mapValue(invokeOp.result(), hatLocalResult);
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                // pass value
                context.mapValue(varOp.result(), context.getValue(varOp.operands().getFirst()));
            }
            return blockBuilder;
        });
        // IO.println("[INFO] Code model: " + funcOp.toText());
        entrypoint.funcOp(funcOp);
    }

    private enum Space {
        PRIVATE,
        SHARED,
    }

    public void dialectifyToHat() {
        // Phases
        dialectifyToHatBarriers();
        dialectifyToHatMemorySpace(Space.SHARED);
        dialectifyToHatMemorySpace(Space.PRIVATE);
    }

}
