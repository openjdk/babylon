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
import hat.Inliner;
import hat.KernelContext;
import hat.phases.HATTier;
import hat.types.BF16;
import hat.types.F16;
import hat.types._F16;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.IfaceValue;
import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.Buffer;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {


    public final ComputeCallGraph computeCallGraph;
    public final CoreOp.FuncOp inlinedEntryPoint;
    public final MethodCallDag callDag;

    public class State {
        public Map<MethodRef, AbstractMethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();
        public List<AccessType> bufferAccessList;
        public Set<TypeElement> accessedTypes;
        public Set<Class<?>> accessedClasses;
        public boolean usesVecTypes;
        public boolean usesFp16;
        public boolean usesBarrier;
        public boolean usesAtomics;
        public Set<String> accessedKcFields;

        public State(MethodHandles.Lookup lookup, CoreOp.FuncOp inlinedEntryPoint) {
            this.usesBarrier = OpHelper.Invoke.stream(lookup, inlinedEntryPoint)
                    .anyMatch(invoke -> invoke.refIs(KernelContext.class) && invoke.named("barrier"));
            this.accessedKcFields = new HashSet<>(OpHelper.FieldAccess.stream(lookup, inlinedEntryPoint)
                    .filter(fieldAccess -> fieldAccess.refType(KernelContext.class)).map(OpHelper.FieldAccess::name).toList());
            this.accessedTypes = new HashSet<>(inlinedEntryPoint.elements().filter(ce -> ce instanceof Op).map(ce -> ((Op) ce).resultType()).toList());
            this.accessedClasses = new HashSet<>(this.accessedTypes.stream().filter(te -> te instanceof ClassType).map(te -> (ClassType) te).map(ct -> (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup(), ct)).toList());
            this.usesVecTypes = this.accessedClasses.stream().anyMatch(IfaceValue.vec.class::isAssignableFrom);
            this.usesFp16 = this.accessedClasses.stream().anyMatch(
                    clazz -> clazz.isAssignableFrom(_F16.class) || clazz.isAssignableFrom(F16.class) || clazz.isAssignableFrom(BF16.class));
            this.usesAtomics = OpHelper.Invoke.stream(lookup, inlinedEntryPoint)
                    .anyMatch(invoke -> invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex("(atomic.*)Inc"));
            this.bufferAccessList = BufferTagger.getAccessList(computeContext.lookup(), inlinedEntryPoint);
        }

        @Override
        public String toString() {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("UsesVecTypes:").append(usesVecTypes).append(", ");
            stringBuilder.append("UsesFp16:").append(usesFp16).append(", ");
            stringBuilder.append("UsesAtomics:").append(usesAtomics).append(", ");
            stringBuilder.append("UsesBarrier:").append(usesBarrier).append(", ");
            stringBuilder.append("AccessedKernelContextFields:").append("[").append(String.join(", ", accessedKcFields)).append("]");
            return stringBuilder.toString();
        }
    }

    public final State state;


    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, method, funcOp);
        }
    }

    public static class KernelReachableUnresolvedMethodCall extends UnresolvedMethodCall implements KernelReachable {
        KernelReachableUnresolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, Method method) {
            super(callGraph, method);
        }
    }


    public static class KernelReachableUnresolvedIfaceMappedMethodCall extends KernelReachableUnresolvedMethodCall {
        KernelReachableUnresolvedIfaceMappedMethodCall(CallGraph<KernelEntrypoint> callGraph, Method method) {
            super(callGraph, method);
        }
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, Method method, CoreOp.FuncOp funcOp) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(computeCallGraph.computeContext.lookup(), null, method, funcOp));
        this.entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        this.inlinedEntryPoint = Inliner.inlineEntrypoint(computeContext.lookup(), entrypoint.funcOp());
        this.state = new State(computeCallGraph.lookup(), this.inlinedEntryPoint);
        HATTier tier = new HATTier(this);
        CoreOp.FuncOp initialEntrypointFuncOp = tier.apply(entrypoint.funcOp());
        entrypoint.funcOp(initialEntrypointFuncOp);
        this.callDag = MethodCallDag.of(lookup(), method, initialEntrypointFuncOp, this.inlinedEntryPoint);
        //  if (this.callDag.isDag()) {
            // this.callDag.view("kernelDag", n->n.funcOp.funcName());
        //  }
       callDag.rankOrdered.stream()
                .filter(methodInfo -> methodInfo.methodRef != null && methodInfo.method.getDeclaringClass().isAssignableFrom(Buffer.class)).forEach(methodInfo ->
                        state.bufferAccessToMethodCallMap.computeIfAbsent(methodInfo.methodRef, _ ->
                                new KernelReachableUnresolvedIfaceMappedMethodCall(this, methodInfo.method)
                        )
                );
        CoreOp.ModuleOp initialModuleOp = callDag.toModuleOp();

        List<CoreOp.FuncOp> initialFuncOps = new ArrayList<>();
        initialModuleOp.functionTable().forEach((_, accessableFuncOp) ->
                initialFuncOps.add(tier.apply(accessableFuncOp))
        );

        setModuleOp(CoreOp.module(initialFuncOps));
    }

    @Override
    public boolean filterCalls(CoreOp.FuncOp f, OpHelper.Invoke invoke) {
        if (Buffer.class.isAssignableFrom(invoke.classOrThrow())) {
            // TODO this side effect seems scary lets do this in a separate pass
            state.bufferAccessToMethodCallMap.computeIfAbsent(invoke.op().invokeReference(), _ ->
                    new KernelReachableUnresolvedIfaceMappedMethodCall(this, invoke.resolveMethodOrThrow())
            );
            return true;
        } else {
            return false;
        }
    }

}
