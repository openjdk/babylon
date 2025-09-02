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
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.SegmentMapper;
import hat.optools.FuncOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.OpTk;
import jdk.incubator.code.Block;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.annotation.Annotation;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

import static hat.ComputeContext.WRAPPER.ACCESS;
//import static hat.ComputeContext.WRAPPER.ESCAPE;
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
        if (computeContext.computeCallGraph.entrypoint.lowered == null) {
            computeContext.computeCallGraph.entrypoint.lowered =
                    OpTk.lower(computeContext.accelerator.lookup,computeContext.computeCallGraph.entrypoint.funcOp());
        }

        boolean interpret = false;
     //   long ns = System.nanoTime();
        backendBridge.computeStart();
        if (config.isINTERPRET()) {
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
       // System.out.println("compute "+ ((System.nanoTime() - ns)/1000)+" us");
    }

    static void wrapInvoke(InvokeOpWrapper iow, Block.Builder bldr, ComputeContext.WRAPPER wrapper, Value cc, Value iface) {
        bldr.op(JavaOp.invoke(wrapper.pre, cc, iface));
        bldr.op(iow.op);
        bldr.op(JavaOp.invoke(wrapper.post, cc, iface));
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


        record PrePost(MethodRef pre, MethodRef post) {
            static PrePost access() {
                return new PrePost(ACCESS.pre, ACCESS.post);
            }

            static PrePost mutate() {
                return new PrePost(MUTATE.pre, MUTATE.post);
            }

            void apply(Block.Builder bldr, CopyContext bldrCntxt, Value computeContext, InvokeOpWrapper invokeOW) {
                if (InvokeOpWrapper.isIfaceMutator(invokeOW.lookup,invokeOW.op)) {                    // iface.v(newV)
                    Value iface = bldrCntxt.getValue(invokeOW.op.operands().getFirst());
                    bldr.op(JavaOp.invoke(MUTATE.pre, computeContext, iface));  // cc->preMutate(iface);
                    bldr.op(invokeOW.op);                         // iface.v(newV);
                    bldr.op(JavaOp.invoke(MUTATE.post, computeContext, iface));
                }
            }
        }

    protected CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
        CoreOp.FuncOp prevFO = computeMethod.funcOp();
        CoreOp.FuncOp returnFO = prevFO;
        if (config.isSHOW_COMPUTE_MODEL()) {
            if (config.isSHOW_COMPUTE_MODEL()) {
                System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
                System.out.println(returnFO.toText());
            }
            var lookup = computeMethod.callGraph.computeContext.accelerator.lookup;
            var paramTable = new OpTk.ParamTable(prevFO);
            returnFO = prevFO.transform((bldr, op) -> {
                if (op instanceof JavaOp.InvokeOp invokeO) {
                    CopyContext bldrCntxt = bldr.context();
                    //Map compute method's first param (computeContext) value to transformed model
                    Value cc = bldrCntxt.getValue(paramTable.list().getFirst().parameter);
                    if (InvokeOpWrapper.isIfaceBufferMethod(lookup, invokeO)) {                    // iface.v(newV)
                        Value iface = bldrCntxt.getValue(invokeO.operands().getFirst());
                        bldr.op(JavaOp.invoke(MUTATE.pre, cc, iface));  // cc->preMutate(iface);
                        bldr.op(invokeO);                         // iface.v(newV);
                        bldr.op(JavaOp.invoke(MUTATE.post, cc, iface)); // cc->postMutate(iface)
                    } else if (InvokeOpWrapper.isIfaceBufferMethod(lookup, invokeO)) {            // iface.v()
                        Value iface = bldrCntxt.getValue(invokeO.operands().getFirst());
                        bldr.op(JavaOp.invoke(ACCESS.pre, cc, iface));  // cc->preAccess(iface);
                        bldr.op(invokeO);                         // iface.v();
                        bldr.op(JavaOp.invoke(ACCESS.post, cc, iface)); // cc->postAccess(iface) } else {
                    } else if (InvokeOpWrapper.isComputeContextMethod(lookup,invokeO) || InvokeOpWrapper.isRawKernelCall(lookup,invokeO)) { //dispatchKernel
                        bldr.op(invokeO);
                    } else {
                        List<Value> list = invokeO.operands();
                        //   System.out.println("args "+list.size());
                        if (!list.isEmpty()) {
                            // System.out.println("method "+invokeOW.method());
                            Annotation[][] parameterAnnotations = InvokeOpWrapper.method(lookup, invokeO).getParameterAnnotations();
                            boolean isVirtual = list.size() > parameterAnnotations.length;
                            //   System.out.println("params length"+parameterAnnotations.length);
                            List<TypeAndAccess> typeAndAccesses = new ArrayList<>();

                            for (int i = isVirtual ? 1 : 0; i < list.size(); i++) {
                                typeAndAccesses.add(TypeAndAccess.of(
                                        parameterAnnotations[i - (isVirtual ? 1 : 0)],
                                        list.get(i)));
                            }
                            List<PrePost> prePosts = new ArrayList<>();
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> typeAndAccess.isIface(lookup))//InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, typeAndAccess.javaType))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.pre, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                            //   }else if (typeAndAccess.wo()||typeAndAccess.rw()) {
                                            //     bldr.op(CoreOp.invoke(MUTATE.pre, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.pre, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                        }
                                    });
                            //  invokeOW.op().operands().stream()
                            // .filter(value -> value.type() instanceof JavaType javaType && InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, javaType))
                            //  .forEach(value ->
                            //          bldr.op(CoreOp.invoke(ESCAPE.pre, cc, bldrCntxt.getValue(value)))
                            //  );
                            bldr.op(invokeO);
                            typeAndAccesses.stream()
                                    .filter(typeAndAccess -> OpTk.isAssignable(lookup, typeAndAccess.javaType, MappableIface.class))
                                    .forEach(typeAndAccess -> {
                                        if (typeAndAccess.ro()) {
                                            bldr.op(JavaOp.invoke(ACCESS.post, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                            //   }else if (typeAndAccess.rw() || typeAndAccess.wo()) {
                                            //       bldr.op(CoreOp.invoke(MUTATE.post, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                        } else {
                                            bldr.op(JavaOp.invoke(MUTATE.post, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                        }
                                    });
                        } else {
                            bldr.op(invokeO);
                        }
                    }
                    return bldr;
                } else {
                    bldr.op(op);
                }
                return bldr;
            });
            if (config.isSHOW_COMPUTE_MODEL()) {
                System.out.println("COMPUTE entrypoint after injecting buffer tracking...");
                System.out.println(returnFO.toText());
            }
        }else{
            if (config.isSHOW_COMPUTE_MODEL()) {
                System.out.println("COMPUTE entrypoint (we will not be injecting buffer tracking...)...");
                System.out.println(returnFO.toText());
            }
        }
        computeMethod.funcOp(returnFO);
        return returnFO;
    }
}
