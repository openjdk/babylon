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
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import optkl.OpHelper;
import optkl.ifacemapper.MappableIface;
import optkl.FuncOpParams;


import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.util.carriers.LookupCarrier;


public class ComputeCallGraph implements LookupCarrier {
    @Override public MethodHandles.Lookup lookup(){
        return computeContext.lookup();
    }

    public static final boolean  showComputeCallDag =Boolean.getBoolean("showComputeCallDag");
    public final ComputeContext computeContext;
    public final MethodCallDag callDag;
    private CoreOp.FuncOp lowered;
    private MethodHandle bytecodeGeneratedMethodHandle;

    static boolean isValidKernelDispatch(MethodHandles.Lookup lookup, Method calledMethod, CoreOp.FuncOp funcOp) {
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
        if (funcOp.body().yieldType().equals(JavaType.VOID)
                && calledMethod.getParameterTypes() instanceof Class<?>[] parameterTypes
                && parameterTypes.length > 1) {
                FuncOpParams paramTable = new FuncOpParams(funcOp);
                paramTable.stream().forEach(paramInfo -> {
                    if (paramInfo.idx == 0) {
                        traits.firstArgKernelContext = parameterTypes[0].isAssignableFrom(KernelContext.class);
                    } else {
                        if (paramInfo.isPrimitive()) {
                            // OK
                        } else if (OpHelper.isAssignable(lookup,paramInfo.javaType, MappableIface.class)){
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

    public ComputeCallGraph(ComputeContext computeContext, Method method, CoreOp.FuncOp entry) {
        this.computeContext = computeContext;
        this.callDag = new MethodCallDag(lookup(), method,entry,null);
        if (showComputeCallDag){
            this.callDag.view("computeCallDag", n -> n.funcOp().funcName());
        }

        callDag.methodCalls()
                .filter(m->
                        this.callDag.entryPoint.method().getDeclaringClass().equals(m.method().getDeclaringClass())
                                && isValidKernelDispatch(computeContext.lookup(),m.method(),m.funcOp()))
                .forEach(m-> kernelCallGraphMap.computeIfAbsent( m.methodRef(), _ ->
                    new KernelCallGraph(this, m.method(), m.funcOp())
            )
        );

    }

    public CoreOp.FuncOp lazyLower(){
        if (lowered == null) {
            lowered =callDag.entryPoint.funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER);
        }
        return lowered;
    }

    public void invokeWithArgs(Object[] args) {
        try {
            if (bytecodeGeneratedMethodHandle == null) {
                bytecodeGeneratedMethodHandle = BytecodeGenerator.generate(lookup(),lazyLower());
            }
            bytecodeGeneratedMethodHandle.invokeWithArguments(args);
        }catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}