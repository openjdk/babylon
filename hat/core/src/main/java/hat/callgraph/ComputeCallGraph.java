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

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.FuncOpParams;
import hat.optools.OpTk;
import optkl.StreamMutable;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.util.*;

import static optkl.OpTkl.isAssignable;
import static optkl.OpTkl.javaRefClassOrThrow;

public class ComputeCallGraph extends CallGraph<ComputeEntrypoint> {

    public final Map<MethodRef, MethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();

    ComputeContextMethodCall computeContextMethodCall;

    @Override
    public MethodHandles.Lookup lookup() {
        return computeContext.lookup();
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

    static boolean isKernelDispatch(MethodHandles.Lookup lookup,Method calledMethod, CoreOp.FuncOp fow) {
        if (fow.body().yieldType().equals(JavaType.VOID)
                && calledMethod.getParameterTypes() instanceof Class<?>[] parameterTypes
                && parameterTypes.length > 1) {
                // We check that the proposed kernel returns void, the first arg is an KernelContext and we have more args
                // We also check that other args are primitive or ifacebuffers  (or atomics?)...
                var firstArgIsKid = StreamMutable.of(false);
                var atLeastOneIfaceBufferParam = StreamMutable.of(false);
                var hasOnlyPrimitiveAndIfaceBufferParams = StreamMutable.of(true);
                FuncOpParams paramTable = new FuncOpParams(fow);
                paramTable.stream().forEach(paramInfo -> {
                    if (paramInfo.idx == 0) {
                        firstArgIsKid.set(parameterTypes[0].isAssignableFrom(KernelContext.class));
                    } else {
                        if (paramInfo.isPrimitive()) {
                            // OK
                        } else if (isAssignable(lookup,paramInfo.javaType, MappableIface.class)){
                            atLeastOneIfaceBufferParam.set(true);
                        } else {
                            hasOnlyPrimitiveAndIfaceBufferParams.set(false);
                        }
                    }
                });
                return true;
            }
            return false;
    }

    public final Map<MethodRef, KernelCallGraph> kernelCallGraphMap = new HashMap<>();


    public ComputeCallGraph(ComputeContext computeContext, Method method, CoreOp.FuncOp funcOp) {
        super(computeContext, new ComputeEntrypoint(null, method, funcOp));
        entrypoint.callGraph = this;
        setModuleOp(OpTk.createTransitiveInvokeModule(computeContext.lookup(), entrypoint.funcOp(), this));
        //close(entrypoint);
    }


    @Override
    public boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass) {
        if (entrypoint.method.getDeclaringClass().equals(javaRefClassOrThrow(computeContext.lookup(),invokeOp))
                && isKernelDispatch(computeContext.lookup(),method, f)) {
            // TODO this side effect is not good.  we should do this when we construct !
            kernelCallGraphMap.computeIfAbsent(methodRef, _ ->
                    new KernelCallGraph(this, methodRef, method, f)
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