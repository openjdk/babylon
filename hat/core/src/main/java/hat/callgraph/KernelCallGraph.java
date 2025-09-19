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
import hat.dialect.HatBlockThreadIdOp;
import hat.dialect.HatGlobalThreadIdOp;
import hat.dialect.HatGlobalSizeOp;
import hat.dialect.HatLocalSizeOp;
import hat.dialect.HatLocalThreadIdOp;
import hat.dialect.HatLocalVarOp;
import hat.dialect.HatMemoryOp;
import hat.dialect.HatPrivateVarOp;
import hat.dialect.HatThreadOP;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.reflect.Method;
import java.util.Arrays;
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

    private boolean isMethodFromHatKernelContext(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();
        return varLoadOp.resultType().toString().equals(kernelContextCanonicalName);
    }

    private boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    private boolean isFieldLoadGlobalThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("x")
                || fieldLoadOp.fieldDescriptor().name().equals("y")
                ||  fieldLoadOp.fieldDescriptor().name().equals("z")
                || fieldLoadOp.fieldDescriptor().name().equals("gix")
                || fieldLoadOp.fieldDescriptor().name().equals("giy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("giz");
    }

    private boolean isFieldLoadGlobalSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("gsx")
                || fieldLoadOp.fieldDescriptor().name().equals("gsy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("gsz")
                || fieldLoadOp.fieldDescriptor().name().equals("maxX")
                || fieldLoadOp.fieldDescriptor().name().equals("maxY")
                ||  fieldLoadOp.fieldDescriptor().name().equals("maxZ");
    }

    private boolean isFieldLoadThreadId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("lix")
                || fieldLoadOp.fieldDescriptor().name().equals("liy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("liz");
    }

    private boolean isFieldLoadThreadSize(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("lsx")
                || fieldLoadOp.fieldDescriptor().name().equals("lsy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("lsz");
    }

    private boolean isFieldLoadBlockId(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        return fieldLoadOp.fieldDescriptor().name().equals("bix")
                || fieldLoadOp.fieldDescriptor().name().equals("biy")
                ||  fieldLoadOp.fieldDescriptor().name().equals("biz");
    }

    private void createBarrierNodeOp(CopyContext context, JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = context.getValues(inputOperands);
        HatBarrierOp hatBarrierOp = new HatBarrierOp(outputOperands);
        Op.Result outputResult = blockBuilder.op(hatBarrierOp);
        Op.Result inputResult = invokeOp.result();
        context.mapValue(inputResult, outputResult);
    }

    public CoreOp.FuncOp dialectifyToHatBarriers(CoreOp.FuncOp funcOp) {
        Stream<CodeElement<?, ?>> elements = funcOp
                .elements()
                .mapMulti((element, consumer) -> {
                    if (element instanceof JavaOp.InvokeOp invokeOp) {
                        if (isMethodFromHatKernelContext(invokeOp) && isMethod(invokeOp, HatBarrierOp.INTRINSIC_NAME)) {
                            consumer.accept(invokeOp);
                        }
                    }
                });
        Set<CodeElement<?, ?>> collect = elements.collect(Collectors.toSet());
        if (collect.isEmpty()) {
            // Return the function with no modifications
            return funcOp;
        }
        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!collect.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                createBarrierNodeOp(context, invokeOp, blockBuilder);
            }
            return blockBuilder;
        });
        // System.out.println("[INFO] Code model: " + funcOp.toText());
        //entrypoint.funcOp(funcOp);
        return funcOp;
    }

    public CoreOp.FuncOp dialectifyToHatMemorySpace(CoreOp.FuncOp funcOp, Space memorySpace) {

        String nameNode = switch (memorySpace) {
            case PRIVATE -> HatPrivateVarOp.INTRINSIC_NAME;
            case SHARED -> HatLocalVarOp.INTRINSIC_NAME;
        };

        //IO.println("ORIGINAL: " + funcOp.toText());
        Stream<CodeElement<?, ?>> elements = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
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
            return funcOp;
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
                        List<Value> inputOperandsVarOp = invokeOp.operands();
                        List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                        HatMemoryOp memoryOp = switch (memorySpace) {
                            case SHARED ->
                                    new HatLocalVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsVarOp);
                            default ->
                                    new HatPrivateVarOp(varOp.varName(), (ClassType) varOp.varValueType(), varOp.resultType(), invokeOp.resultType(), outputOperandsVarOp);
                        };
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
        //entrypoint.funcOp(funcOp);
        return funcOp;
    }

    private enum Space {
        PRIVATE,
        SHARED,
    }

    private int getDimension(ThreadAccess threadAccess, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        String fieldName = fieldLoadOp.fieldDescriptor().name();
        switch (threadAccess) {
            case GLOBAL_ID -> {
                if (fieldName.equals("y")) {
                    return 1;
                } else if (fieldName.equals("z")) {
                    return 2;
                }
                return 0;
            }
            case GLOBAL_SIZE -> {
                if (fieldName.equals("gsy")) {
                    return 1;
                } else if (fieldName.equals("gsz")) {
                    return 2;
                }
                return 0;
            }
            case LOCAL_ID -> {
                if (fieldName.equals("liy")) {
                    return 1;
                } else if (fieldName.equals("lyz")) {
                    return 2;
                }
                return 0;
            }
            case LOCAL_SIZE -> {
                if (fieldName.equals("lsy")) {
                    return 1;
                } else if (fieldName.equals("lsz")) {
                    return 2;
                }
                return 0;
            }
            case BLOCK_ID ->  {
                if (fieldName.equals("biy")) {
                    return 1;
                } else if (fieldName.equals("biz")) {
                    return 2;
                }
                return 0;
            }
        }
        return -1;
    }

    public CoreOp.FuncOp dialectifyToHatThreadIds(CoreOp.FuncOp funcOp, ThreadAccess threadAccess) {
        Stream<CodeElement<?, ?>> elements = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                        List<Value> operands = fieldLoadOp.operands();
                        for (Value inputOperand : operands) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                    boolean isThreadIntrinsic = switch (threadAccess) {
                                        case GLOBAL_ID -> isFieldLoadGlobalThreadId(fieldLoadOp);
                                        case GLOBAL_SIZE -> isFieldLoadGlobalSize(fieldLoadOp);
                                        case LOCAL_ID -> isFieldLoadThreadId(fieldLoadOp);
                                        case LOCAL_SIZE -> isFieldLoadThreadSize(fieldLoadOp);
                                        case BLOCK_ID ->  isFieldLoadBlockId(fieldLoadOp);
                                    };
                                    if (isMethodFromHatKernelContext(varLoadOp) && isThreadIntrinsic) {
                                        consumer.accept(fieldLoadOp);
                                        consumer.accept(varLoadOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            // No memory nodes involved
            return funcOp;
        }

        funcOp = funcOp.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            } else if (op instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                List<Value> operands = fieldLoadOp.operands();
                for (Value operand : operands) {
                    if (operand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        List<Value> varLoadOperands = varLoadOp.operands();
                        List<Value> outputOperands = context.getValues(varLoadOperands);
                        int dim = getDimension(threadAccess, fieldLoadOp);
                        if (dim < 0) {
                            throw new IllegalStateException("Thread Access can't be below 0!");
                        }
                        HatThreadOP threadOP = switch (threadAccess) {
                            case GLOBAL_ID -> new HatGlobalThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case GLOBAL_SIZE -> new HatGlobalSizeOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case LOCAL_ID -> new HatLocalThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case LOCAL_SIZE -> new HatLocalSizeOp(dim, fieldLoadOp.resultType(), outputOperands);
                            case BLOCK_ID -> new HatBlockThreadIdOp(dim, fieldLoadOp.resultType(), outputOperands);
                        };
                        Op.Result threadResult = blockBuilder.op(threadOP);
                        context.mapValue(fieldLoadOp.result(), threadResult);
                    }
                }
            }
            return blockBuilder;
        });
        IO.println("[INFO] Code model: " + funcOp.toText());
        //entrypoint.funcOp(funcOp);
        return funcOp;
    }

    private enum ThreadAccess {
        GLOBAL_ID,
        GLOBAL_SIZE,
        LOCAL_ID,
        LOCAL_SIZE,
        BLOCK_ID,
    }

    private CoreOp.FuncOp dialectifyToHat(CoreOp.FuncOp funcOp) {
        CoreOp.FuncOp f = dialectifyToHatBarriers(funcOp);
        for (Space space : Space.values()) {
            f = dialectifyToHatMemorySpace(f, space);
        }
        for (ThreadAccess threadAccess : ThreadAccess.values()) {
            f = dialectifyToHatThreadIds(f, threadAccess);
        }
        return f;
    }

    public void dialectifyToHat() {
        // Analysis Phases to transform the Java Code Model to a HAT Code Model

        // Main kernel
        {
            CoreOp.FuncOp f = dialectifyToHat(entrypoint.funcOp());
            entrypoint.funcOp(f);
        }

//        // Reachable functions
//        if (moduleOp != null) {
//            moduleOp.functionTable().forEach((entryPoint, kernelOp) -> {
//                CoreOp.FuncOp f = dialectifyToHat(kernelOp);
//                moduleOp.functionTable().put(entryPoint, f);
//            });
//        }
//        } else {
//            kernelReachableResolvedStream().forEach((kernel) -> {
//                CoreOp.FuncOp f = dialectifyToHat(kernel.funcOp());
//                kernel.funcOp(f);
//            });
//        }
    }
}
