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
import hat.KernelContext;
import jdk.incubator.code.CodeTransformer;
import optkl.Invoke;
import optkl.Trxfmr;
import optkl.util.CallSite;
import optkl.ifacemapper.Buffer;
import hat.callgraph.CallGraph;
import optkl.ifacemapper.MappableIface;
import optkl.FuncOpParams;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.annotation.Annotation;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;
import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.classTypeToTypeOrThrow;
import static optkl.OpTkl.isAssignable;

public abstract class FFIBackend extends FFIBackendDriver {

    public FFIBackend(Arena arena,MethodHandles.Lookup lookup,String libName, Config config) {
        super(arena, lookup,libName, config);
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
            return isAssignable(lookup, javaType,MappableIface.class);
        }
        boolean ro(){
            for (Annotation annotation : annotations) {
                if (  annotation instanceof MappableIface.RO){
                    System.out.println("MappableIface.RO");
                    return true;
                }
            }
            return false;
        }
        boolean rw(){
            for (Annotation annotation : annotations) {
                if (  annotation instanceof MappableIface.RW){
                    System.out.println("MappableIface.RW");
                    return true;
                }
            }
            return false;
        }
        boolean wo(){
            for (Annotation annotation : annotations) {
                if (  annotation instanceof MappableIface.WO){
                    System.out.println("MappableIface.WO");
                    return true;
                }
            }
            return false;
        }
    }



    // This code should be common with jextracted-shared probably should be pushed down into another lib?
    protected CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
        CoreOp.FuncOp transformedFuncOp = computeMethod.funcOp();
        var here = CallSite.of(FFIBackend.class,"injectBufferTracking");
        if (config().minimizeCopies()) {
            if (config().showComputeModel()) {
                System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
                System.out.println(transformedFuncOp.toText());
            }
            var paramTable = new FuncOpParams(computeMethod.funcOp());

            transformedFuncOp = new Trxfmr(computeMethod.funcOp()).transform(_->true,(bldr, op) -> {
                if (invokeOpHelper(lookup(),op) instanceof Invoke invoke ) {
                    Value cc = bldr.context().getValue(paramTable.list().getFirst().parameter);
                    if (invoke.isMappableIface() && invoke.returnsVoid()) {                    // iface.v(newV)
                        Value iface = bldr.context().getValue(invoke.op().operands().getFirst());
                        bldr.op(JavaOp.invoke(MUTATE.pre, cc, iface));                  // cc->preMutate(iface);
                        bldr.op(invoke.op());                                              // iface.v(newV);
                        bldr.op(JavaOp.invoke(MUTATE.post, cc, iface));                 // cc->postMutate(iface)
                    } else if (invoke.isMappableIface()
                            && (
                                    invoke.returnsClassType()
                                            && classTypeToTypeOrThrow(lookup(), (ClassType)invoke.returnType()) instanceof Class<?> type
                                            && Buffer.class.isAssignableFrom(type)
                                ||
                                            invoke.returnsPrimitive()
                               )
                    ) {
                        // if this is accessing a width if an array we don't want to force the buffer back from the GPU.
                        Value iface = bldr.context().getValue(invoke.op().operands().getFirst());
                        bldr.op(JavaOp.invoke(ACCESS.pre, cc, iface));                 // cc->preAccess(iface);
                        bldr.op(invoke.op());                                             // iface.v();
                        bldr.op(JavaOp.invoke(ACCESS.post, cc, iface));                // cc->postAccess(iface)
                    } else if (invoke.refIs(ComputeContext.class,KernelContext.class)) { //dispatchKernel
                        bldr.op(invoke.op());
                    } else {
                        List<Value> list = invoke.op().operands();
                        if (!list.isEmpty()) {
                          //  System.out.println("Escape! with args " +invokeOp.toText());
                            // We need to check

                            var method = invoke.resolveMethodOrThrow();

                            Annotation[][] parameterAnnotations = method.getParameterAnnotations();
                            boolean isVirtual = list.size() > parameterAnnotations.length;
                            List<TypeAndAccess> typeAndAccesses = new ArrayList<>();
                            for (int i = isVirtual ? 1 : 0; i < list.size(); i++) {
                                typeAndAccesses.add(TypeAndAccess.of(
                                        parameterAnnotations[i - (isVirtual ? 1 : 0)],
                                        list.get(i)));
                            }
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> typeAndAccess.isIface(lookup()))//InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, typeAndAccess.javaType))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.pre, cc, bldr.context().getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.pre, cc, bldr.context().getValue(typeAndAccess.value)));
                                        }
                                    });
                            bldr.op(invoke.op());
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> isAssignable(lookup(), typeAndAccess.javaType, MappableIface.class))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.post, cc, bldr.context().getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.post, cc, bldr.context().getValue(typeAndAccess.value)));
                                        }
                                    });
                        } else {
                            bldr.op(invoke.op());
                        }
                    }
                    return bldr;
                } else {
                    bldr.op(op);
                }
                return bldr;
            }).funcOp();
            if (config().showComputeModel()) {
                System.out.println("COMPUTE entrypoint after injecting buffer tracking...");
                System.out.println(transformedFuncOp.toText());
            }
        }else{
            if (config().showComputeModel()) {
                System.out.println("COMPUTE entrypoint (we will not be injecting buffer tracking...)...");
                System.out.println(transformedFuncOp.toText());
            }
        }
        computeMethod.funcOp(transformedFuncOp);
        return transformedFuncOp;
    }
}
