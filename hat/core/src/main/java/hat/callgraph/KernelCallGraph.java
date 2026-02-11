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

import optkl.OpHelper;
import optkl.ifacemapper.AccessType;
import hat.BufferTagger;
import optkl.ifacemapper.Buffer;

import hat.phases.HATTier;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.*;

import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Stream;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public final ComputeCallGraph computeCallGraph;
    public final Map<MethodRef, AbstractMethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();
    public static class Traits{
        public final List<AccessType> bufferAccessList;
        public boolean usesArrayView;
        Traits(List<AccessType> bufferAccessList){
            this.bufferAccessList=bufferAccessList;
        }
    }
    final public Traits traits;


    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph,  Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, method, funcOp);
        }
    }

    public static class KernelReachableUnresolvedMethodCall extends UnresolvedMethodCall implements KernelReachable {
        KernelReachableUnresolvedMethodCall(CallGraph<KernelEntrypoint> callGraph,  Method method) {
            super(callGraph,  method);
        }
    }


    public static class KernelReachableUnresolvedIfaceMappedMethodCall extends KernelReachableUnresolvedMethodCall {
        KernelReachableUnresolvedIfaceMappedMethodCall(CallGraph<KernelEntrypoint> callGraph,  Method method) {
            super(callGraph,  method);
        }
    }


    public Stream<KernelReachableResolvedMethodCall> kernelReachableResolvedStream() {
        return methodRefToMethodCallMap.values().stream()
                .filter(call -> call instanceof KernelReachableResolvedMethodCall)
                .map(kernelReachable -> (KernelReachableResolvedMethodCall) kernelReachable);
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, Method method, CoreOp.FuncOp funcOp) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(computeCallGraph.computeContext.lookup(),null,  method, funcOp));
        this.entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        this.traits = new Traits(BufferTagger.getAccessList(computeContext.lookup(), entrypoint.funcOp()));

        HATTier tier = new HATTier(this);
        CoreOp.FuncOp initialEntrypointFuncOp = tier.apply(entrypoint.funcOp());

        entrypoint.funcOp(initialEntrypointFuncOp);
        List<CoreOp.FuncOp> initialFuncOps = new ArrayList<>();

        CoreOp.ModuleOp initialModuleOp = createTransitiveInvokeModule(computeContext.lookup(), method,entrypoint.funcOp());

        initialModuleOp.functionTable().forEach((_, accessableFuncOp) ->
                initialFuncOps.add( tier.apply(accessableFuncOp))
        );

        setModuleOp(CoreOp.module(initialFuncOps));
    }

    @Override
    public boolean filterCalls(CoreOp.FuncOp f, OpHelper.Invoke invoke) {
        var methodRef = invoke.op().invokeDescriptor();
        Class<?> javaRefTypeClass = invoke.classOrThrow();
        if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
            // TODO this side effect seems scary lets do this in a separate pass
            bufferAccessToMethodCallMap.computeIfAbsent(methodRef, _ ->
                    new KernelReachableUnresolvedIfaceMappedMethodCall(this, invoke.resolveMethodOrThrow())
            );
        } else {
            return false;
        }
        return true;
    }


}
