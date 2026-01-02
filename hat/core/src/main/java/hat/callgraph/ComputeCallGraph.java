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

import hat.ComputeContext;
import hat.Config;
import hat.KernelContext;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.FuncOpParams;


import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.util.*;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.isAssignable;

public class ComputeCallGraph extends CallGraph<ComputeEntrypoint> {

    public final Map<MethodRef, MethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();

    ComputeContextMethodCall computeContextMethodCall;

    public Config config() {
        return computeContext.config();
    }

    public interface ComputeReachable {
    }

    public abstract static class ComputeReachableResolvedMethodCall extends ResolvedMethodCall implements ComputeReachable {
        public ComputeReachableResolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method, funcOp);
        }
    }

    public static class ComputeReachableUnresolvedMethodCall extends UnresolvedMethodCall implements ComputeReachable {
        ComputeReachableUnresolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeReachableIfaceMappedMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeReachableIfaceMappedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeReachableAcceleratorMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeReachableAcceleratorMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class ComputeContextMethodCall extends ComputeReachableUnresolvedMethodCall {
        ComputeContextMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class OtherComputeReachableResolvedMethodCall extends ComputeReachableResolvedMethodCall {
        OtherComputeReachableResolvedMethodCall(CallGraph<ComputeEntrypoint> callGraph, MethodRef targetMethodRef, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method, funcOp);
        }
    }

    static boolean isValidKernelDispatch(MethodHandles.Lookup lookup, Method calledMethod, CoreOp.FuncOp fow) {
        // We check that the proposed kernel returns void, the first arg is an KernelContext and we have more args
        // We also check that other args are primitive or ifacebuffers  (or atomics?)...
        class Traits{
            boolean firstArgKernelContext = false;
            boolean atLeastOneIfaceBufferParam=false;
            boolean hasOnlyPrimitiveAndIfaceBufferParams=true;
            boolean ok(){
                return firstArgKernelContext &&atLeastOneIfaceBufferParam&&hasOnlyPrimitiveAndIfaceBufferParams;
            }
        }
        var traits = new Traits();
        if (fow.body().yieldType().equals(JavaType.VOID)
                && calledMethod.getParameterTypes() instanceof Class<?>[] parameterTypes
                && parameterTypes.length > 1) {
                FuncOpParams paramTable = new FuncOpParams(fow);
                paramTable.stream().forEach(paramInfo -> {
                    if (paramInfo.idx == 0) {
                        traits.firstArgKernelContext = parameterTypes[0].isAssignableFrom(KernelContext.class);
                    } else {
                        if (paramInfo.isPrimitive()) {
                            // OK
                        } else if (isAssignable(lookup,paramInfo.javaType, MappableIface.class)){
                            traits.atLeastOneIfaceBufferParam= true;
                        } else {
                            traits.hasOnlyPrimitiveAndIfaceBufferParams=false;
                        }
                    }
                });
            }
            return traits.ok();
    }

    public final Map<MethodRef, KernelCallGraph> kernelCallGraphMap = new HashMap<>();


    public ComputeCallGraph(ComputeContext computeContext, Method method, CoreOp.FuncOp funcOp) {
        super(computeContext, new ComputeEntrypoint(null, method, funcOp));
        entrypoint.callGraph = this;
        setModuleOp(createTransitiveInvokeModule(computeContext.lookup(), entrypoint.funcOp()));
    }


    @Override
    public boolean filterCalls(CoreOp.FuncOp funcOp, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass) {
        var invoke = invokeOpHelper(computeContext.lookup(),invokeOp);
        if (entrypoint.method.getDeclaringClass().equals(invoke.classOrThrow())
                && isValidKernelDispatch(computeContext.lookup(),method, funcOp)) {
            // TODO this side effect is not good.  we should do this when we construct !
            kernelCallGraphMap.computeIfAbsent(methodRef, _ ->
                    new KernelCallGraph(this, methodRef, method, funcOp)
            );
        } else if (ComputeContext.class.isAssignableFrom(javaRefTypeClass)) {
            computeContextMethodCall = new ComputeContextMethodCall(this, methodRef, method);
        } else if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
            bufferAccessToMethodCallMap.computeIfAbsent(methodRef, _ ->
                    new ComputeReachableIfaceMappedMethodCall(this, methodRef, method)
            );
        } else {
            return false;
        }
        return true;
    }

}