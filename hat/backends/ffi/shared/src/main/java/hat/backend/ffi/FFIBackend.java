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
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.SegmentMapper;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.annotation.Annotation;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;

public abstract class FFIBackend extends FFIBackendDriver {
    public final Arena arena = Arena.global();

    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> boundSchema) {
        return segmentMapper.allocate(arena, boundSchema);
    }

    public FFIBackend(String libName, Config config) {
        super(libName, config);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        var here = OpTk.CallSite.of(FFIBackend.class, "dispatchCompute");
        if (computeContext.computeCallGraph.entrypoint.lowered == null) {
            computeContext.computeCallGraph.entrypoint.lowered =
                    OpTk.lower(here, computeContext.computeCallGraph.entrypoint.funcOp());
        }

        backendBridge.computeStart();
        if (config().interpret()) {
            Interpreter.invoke(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered, args);
        } else {
            try {
                if (computeContext.computeCallGraph.entrypoint.mh == null) {
                    computeContext.computeCallGraph.entrypoint.mh = BytecodeGenerator.generate(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered);
                }
                computeContext.computeCallGraph.entrypoint.mh.invokeWithArguments(args);
            } catch (Throwable e) {
                System.out.println(computeContext.computeCallGraph.entrypoint.lowered.toText());
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
            return OpTk.isAssignable(lookup, javaType,MappableIface.class);
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
        var here = OpTk.CallSite.of(FFIBackend.class,"injectBufferTracking");
        if (config().minimizeCopies()) {
            if (config().showComputeModel()) {
                System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
                System.out.println(transformedFuncOp.toText());
            }
            var lookup = computeMethod.callGraph.computeContext.accelerator.lookup;
            var paramTable = new FuncOpParams(computeMethod.funcOp());

            transformedFuncOp = OpTk.transform(here, computeMethod.funcOp(),(bldr, op) -> {
                if (op instanceof JavaOp.InvokeOp invokeOp) {
                    Value cc = bldr.context().getValue(paramTable.list().getFirst().parameter);
                    if (OpTk.isIfaceBufferMethod(lookup, invokeOp)&& OpTk.javaReturnType(invokeOp).equals(JavaType.VOID)) {                    // iface.v(newV)
                        Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                        bldr.op(JavaOp.invoke(MUTATE.pre, cc, iface));                  // cc->preMutate(iface);
                        bldr.op(invokeOp);                                              // iface.v(newV);
                        bldr.op(JavaOp.invoke(MUTATE.post, cc, iface));                 // cc->postMutate(iface)
                    } else if (OpTk.isIfaceBufferMethod(lookup, invokeOp)
                            && (
                                    (OpTk.javaReturnType(invokeOp) instanceof ClassType returnClassType)
                                            && OpTk.classTypeToTypeOrThrow(lookup, returnClassType) instanceof Class<?> type
                                            && Buffer.class.isAssignableFrom(type)
                                ||
                                            (OpTk.javaReturnType(invokeOp) instanceof PrimitiveType primitiveType)
                               )
                    ) {
                        // if this is accessing a width if an array we don't want to force the buffer back from the GPU.
                        Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                        bldr.op(JavaOp.invoke(ACCESS.pre, cc, iface));                 // cc->preAccess(iface);
                        bldr.op(invokeOp);                                             // iface.v();
                        bldr.op(JavaOp.invoke(ACCESS.post, cc, iface));                // cc->postAccess(iface)
                    } else if (OpTk.isComputeContextMethod(lookup,invokeOp) || OpTk.isKernelContextMethod(lookup,invokeOp)) { //dispatchKernel
                        bldr.op(invokeOp);
                    } else {
                        List<Value> list = invokeOp.operands();
                        if (!list.isEmpty()) {
                            System.out.println("Escape! with args " +invokeOp.toText());
                            // We need to check
                            Annotation[][] parameterAnnotations = OpTk.methodOrThrow(lookup, invokeOp).getParameterAnnotations();
                            boolean isVirtual = list.size() > parameterAnnotations.length;
                            List<TypeAndAccess> typeAndAccesses = new ArrayList<>();
                            for (int i = isVirtual ? 1 : 0; i < list.size(); i++) {
                                typeAndAccesses.add(TypeAndAccess.of(
                                        parameterAnnotations[i - (isVirtual ? 1 : 0)],
                                        list.get(i)));
                            }
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> typeAndAccess.isIface(lookup))//InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, typeAndAccess.javaType))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.pre, cc, bldr.context().getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.pre, cc, bldr.context().getValue(typeAndAccess.value)));
                                        }
                                    });
                            bldr.op(invokeOp);
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> OpTk.isAssignable(lookup, typeAndAccess.javaType, MappableIface.class))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.post, cc, bldr.context().getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.post, cc, bldr.context().getValue(typeAndAccess.value)));
                                        }
                                    });
                        } else {
                            bldr.op(invokeOp);
                        }
                    }
                    return bldr;
                } else {
                    bldr.op(op);
                }
                return bldr;
            });
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
