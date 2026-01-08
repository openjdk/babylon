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
package hat.backend;


import hat.ComputeContext;
import hat.Config;
import hat.KernelContext;
//import hat.backend.java.JavaMultiThreadedBackend;
//import hat.backend.java.JavaSequentialBackend;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.FuncOpParams;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.ifacemapper.AccessType;
import optkl.ifacemapper.BufferAllocator;
import hat.callgraph.KernelCallGraph;
import optkl.ifacemapper.MappableIface;
import optkl.util.carriers.LookupCarrier;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.ServiceLoader;
import java.util.function.Predicate;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;
import static optkl.OpHelper.Named.NamedStaticOrInstance.Invoke.invoke;

public  abstract class Backend implements BufferAllocator, LookupCarrier {
    private final Config config;

    public Config config(){
        return config;
    }

    private final Arena arena;
    @Override public Arena arena(){
        return arena;
    }
    private final MethodHandles.Lookup lookup;
    @Override public MethodHandles.Lookup lookup(){
        return lookup;
    }

    protected Backend(Arena arena, MethodHandles.Lookup lookup,Config config){
        this.lookup = lookup;
        this.arena =arena;
        this.config = config;
    }

    final public String getName() {
        return this.getClass().getName();
    }

    final public static Predicate<Backend> PROPERTY = (backend) -> {
        String requestedBackendName = System.getProperty("hat.backend");
        if (requestedBackendName == null) {
            throw new IllegalStateException("Expecting property hat.backend to name a Backend class");
        }
        String backendName = backend.getName();
        return (backendName.equals(requestedBackendName));
    };
    public static Predicate<Backend> FIRST = backend -> true;

    public static Backend getBackend(Predicate<Backend> backendPredicate) {
        return ServiceLoader.load(Backend.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .filter(backendPredicate)
                .findFirst().orElseThrow();
    }

    public abstract void computeContextHandoff(ComputeContext computeContext);

    public abstract void dispatchCompute(ComputeContext computeContext, Object... args);

    public abstract void dispatchKernel(KernelCallGraph kernelCallGraph, KernelContext kernelContext, Object... args);


    public static  CoreOp.FuncOp injectBufferTracking(Config config, MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        var transformer = Trxfmr.of(lookup,funcOp);
        if (config.minimizeCopies()) {
            var paramTable = new FuncOpParams(funcOp);
            return transformer
                    .when(config.showComputeModel(), trxfmr -> trxfmr.toText("COMPUTE before injecting buffer tracking..."))
                    .when(config.showComputeModelJavaCode(), trxfmr -> trxfmr.toJava("COMPUTE (Java) before injecting buffer tracking..."))
                    .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                        var invoke = invoke(lookup, c.op());
                        if (invoke.isMappableIface() && (invoke.returns(MappableIface.class) || invoke.returnsPrimitive())) {
                            Value computeContext = c.getValue(paramTable.list().getFirst().parameter);
                            Value ifaceMappedBuffer = c.mappedOperand(0);
                            c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.pre : ACCESS.pre, computeContext, ifaceMappedBuffer));
                            c.retain();
                            c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.post : ACCESS.post, computeContext, ifaceMappedBuffer));
                        } else if (!invoke.refIs(ComputeContext.class) && invoke.operandCount() > 0) {
                            List<AccessType.TypeAndAccess> typeAndAccesses = invoke.paramaterAccessList();
                            Value computeContext = c.getValue(paramTable.list().getFirst().parameter);
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> typeAndAccess.isIface(lookup))
                                    .forEach(typeAndAccess ->
                                            c.add(JavaOp.invoke(
                                                    typeAndAccess.ro() ? ACCESS.pre : MUTATE.pre,
                                                    computeContext, c.getValue(typeAndAccess.value()))
                                            )
                                    );
                            c.retain();
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> OpHelper.isAssignable(lookup, typeAndAccess.javaType(), MappableIface.class))
                                    .forEach(typeAndAccess ->
                                            c.add(JavaOp.invoke(
                                                    typeAndAccess.ro() ? ACCESS.post : MUTATE.post,
                                                    computeContext, c.getValue(typeAndAccess.value()))
                                            )
                                    );
                        }
                    })
                    .when(config.showComputeModel(), trxfmr -> trxfmr.toText("COMPUTE after injecting buffer tracking..."))
                    .funcOp();
        } else {
            return transformer.when(config.showComputeModel(), trxfmr -> trxfmr.toText("COMPUTE not injecting buffer tracking)")).funcOp();
        }
    }


}
