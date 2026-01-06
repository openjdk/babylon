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

package hat.backend.ffi;

import hat.ComputeContext;
import hat.Config;
import hat.callgraph.CallGraph;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;
import optkl.FuncOpParams;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.ifacemapper.MappableIface;

import java.lang.annotation.Annotation;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;
import static optkl.OpHelper.NamedOpHelper.Invoke.invokeOpHelper;

public abstract class FFIBackend extends FFIBackendDriver {

    public FFIBackend(Arena arena, MethodHandles.Lookup lookup, String libName, Config config) {
        super(arena, lookup, libName, config);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        if (computeContext.computeEntrypoint().lowered == null) {
            computeContext.computeEntrypoint().lowered =
                    computeContext.computeEntrypoint().funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER);
        }
        backendBridge.computeStart();
        if (config().interpret()) {
            Interpreter.invoke(computeContext.lookup(), computeContext.computeEntrypoint().lowered, args);
        } else {
            try {
                if (computeContext.computeEntrypoint().mh == null) {
                    computeContext.computeEntrypoint().mh = BytecodeGenerator.generate(computeContext.lookup(), computeContext.computeEntrypoint().lowered);
                }
                computeContext.computeEntrypoint().mh.invokeWithArguments(args);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
        backendBridge.computeEnd();
    }


    record TypeAndAccess(Annotation[] annotations, Value value, JavaType javaType) {
        static TypeAndAccess of(Annotation[] annotations, Value value) {
            return new TypeAndAccess(annotations, value, (JavaType) value.type());
        }

        boolean isIface(MethodHandles.Lookup lookup) {
            return OpHelper.isAssignable(lookup, javaType, MappableIface.class);
        }

        boolean ro() {
            for (Annotation annotation : annotations) {
                if (annotation instanceof MappableIface.RO) {
                    //System.out.println("MappableIface.RO");
                    return true;
                }
            }
            return false;
        }

        boolean rw() {
            for (Annotation annotation : annotations) {
                if (annotation instanceof MappableIface.RW) {
                    //System.out.println("MappableIface.RW");
                    return true;
                }
            }
            return false;
        }

        boolean wo() {
            for (Annotation annotation : annotations) {
                if (annotation instanceof MappableIface.WO) {
                   // System.out.println("MappableIface.WO");
                    return true;
                }
            }
            return false;
        }
    }


    // This code should be common with jextracted-shared probably should be pushed down into another lib?
    protected CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {

        var transformer =   Trxfmr.of(computeMethod.funcOp());
        if (config().minimizeCopies()) {
            var paramTable = new FuncOpParams(computeMethod.funcOp());
            transformer
                    .when(config().showComputeModel(), trxfmr -> trxfmr.toText("COMPUTE before injecting buffer tracking..."))
                    .when(config().showComputeModelJavaCode(), trxfmr -> trxfmr.toJavaSource(lookup(),"COMPUTE (Java) before injecting buffer tracking..."))
                    .transform(ce -> ce instanceof JavaOp.InvokeOp, c -> {
                        var invoke = invokeOpHelper(lookup(), c.op());
                        if (invoke.isMappableIface() && (invoke.returns(MappableIface.class) || invoke.returnsPrimitive())) {
                            Value computeContext = c.builder().context().getValue(paramTable.list().getFirst().parameter);
                            Value ifaceMappedBuffer = c.builder().context().getValue(invoke.op().operands().getFirst());
                            c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.pre : ACCESS.pre, computeContext, ifaceMappedBuffer));
                            c.retain();
                            c.add(JavaOp.invoke(invoke.returnsVoid() ? MUTATE.post : ACCESS.post, computeContext, ifaceMappedBuffer));
                        } else if (!invoke.refIs(ComputeContext.class) && invoke.operandCount()>0) {
                                Annotation[][] parameterAnnotations =  invoke.resolveMethodOrThrow().getParameterAnnotations();
                                int firstParam =invoke.isInstance()?1:0; // if virtual
                                List<TypeAndAccess> typeAndAccesses = new ArrayList<>();
                                for (int i = firstParam; i < invoke.operandCount(); i++) {
                                    typeAndAccesses.add(TypeAndAccess.of(parameterAnnotations[i - firstParam], invoke.op().operands().get(i)));
                                }
                                Value computeContext = c.builder().context().getValue(paramTable.list().getFirst().parameter);
                                typeAndAccesses.stream()
                                        .filter(typeAndAccess -> typeAndAccess.isIface(lookup()))
                                        .forEach(typeAndAccess ->
                                            c.add(JavaOp.invoke(
                                                    typeAndAccess.ro() ? ACCESS.pre : MUTATE.pre,
                                                    computeContext, c.builder().context().getValue(typeAndAccess.value))
                                            )
                                        );
                                c.retain();
                                typeAndAccesses.stream()
                                        .filter(typeAndAccess -> OpHelper.isAssignable(lookup(), typeAndAccess.javaType, MappableIface.class))
                                        .forEach(typeAndAccess ->
                                            c.add(JavaOp.invoke(
                                                    typeAndAccess.ro() ? ACCESS.post : MUTATE.post,
                                                    computeContext, c.builder().context().getValue(typeAndAccess.value))
                                            )
                                        );
                            }
                    })
                    .when(config().showComputeModel(), trxfmr -> trxfmr.toText("COMPUTE after injecting buffer tracking..."))
                    .run(trxfmr -> computeMethod.funcOp(trxfmr.funcOp()));
        } else {
            transformer.when(config().showComputeModel(),trxfmr -> trxfmr.toText("COMPUTE not injecting buffer tracking)"));
        }
        return computeMethod.funcOp();
    }
}
