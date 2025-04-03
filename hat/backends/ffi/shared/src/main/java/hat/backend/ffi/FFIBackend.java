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
import jdk.incubator.code.Block;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;

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

    public FFIBackend(String libName) {
        super(libName);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        if (computeContext.computeCallGraph.entrypoint.lowered == null) {
            computeContext.computeCallGraph.entrypoint.lowered = computeContext.computeCallGraph.entrypoint.funcOpWrapper().lower();
        }


        boolean interpret = false;
     //   long ns = System.nanoTime();
        backendBridge.computeStart();
        if (interpret) {
            Interpreter.invoke(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered.op(), args);
        } else {
            try {
                if (computeContext.computeCallGraph.entrypoint.mh == null) {
                    computeContext.computeCallGraph.entrypoint.mh = BytecodeGenerator.generate(computeContext.accelerator.lookup, computeContext.computeCallGraph.entrypoint.lowered.op());
                }
                computeContext.computeCallGraph.entrypoint.mh.invokeWithArguments(args);
            } catch (Throwable e) {
                computeContext.computeCallGraph.entrypoint.lowered.op().writeTo(System.out);
                throw new RuntimeException(e);
            }
        }
        backendBridge.computeEnd();
       // System.out.println("compute "+ ((System.nanoTime() - ns)/1000)+" us");
    }

    static void wrapInvoke(InvokeOpWrapper iow, Block.Builder bldr, ComputeContext.WRAPPER wrapper, Value cc, Value iface) {
        bldr.op(CoreOp.invoke(wrapper.pre, cc, iface));
        bldr.op(iow.op());
        bldr.op(CoreOp.invoke(wrapper.post, cc, iface));
    }

    record TypeAndAccess(Annotation[] annotations, Value value, JavaType javaType) {
        static TypeAndAccess of(Annotation[] annotations, Value value) {
            return new TypeAndAccess(annotations, value, (JavaType) value.type());
        }
        boolean isIface(MethodHandles.Lookup lookup) {
            return InvokeOpWrapper.isIfaceUsingLookup(lookup, javaType);
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


        record PrePost(MethodRef pre,MethodRef post) {
            static PrePost access() {
                return new PrePost(ACCESS.pre, ACCESS.post);
            }

            static PrePost mutate() {
                return new PrePost(MUTATE.pre, MUTATE.post);
            }

          //  static PrePost escape() {
            //    return new PrePost(ESCAPE.pre, ESCAPE.post);
           // }

            void apply(Block.Builder bldr, CopyContext bldrCntxt, Value computeContext, InvokeOpWrapper invokeOW) {
                if (invokeOW.isIfaceMutator()) {                    // iface.v(newV)
                    Value iface = bldrCntxt.getValue(invokeOW.operandNAsValue(0));
                    bldr.op(CoreOp.invoke(MUTATE.pre, computeContext, iface));  // cc->preMutate(iface);
                    bldr.op(invokeOW.op());                         // iface.v(newV);
                    bldr.op(CoreOp.invoke(MUTATE.post, computeContext, iface));
                }
            }
        }

    protected static FuncOpWrapper injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod, boolean show, boolean inject) {
        FuncOpWrapper prevFOW = computeMethod.funcOpWrapper();
        FuncOpWrapper returnFOW = prevFOW;
        if (inject) {
            if (show) {
                System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
                returnFOW.op().writeTo(System.out);
            }
            returnFOW = prevFOW.transformInvokes((bldr, invokeOW) -> {
                CopyContext bldrCntxt = bldr.context();
                //Map compute method's first param (computeContext) value to transformed model
                Value cc = bldrCntxt.getValue(prevFOW.parameter(0));
                if (invokeOW.isIfaceMutator()) {                    // iface.v(newV)
                    Value iface = bldrCntxt.getValue(invokeOW.operandNAsValue(0));
                    bldr.op(CoreOp.invoke(MUTATE.pre, cc, iface));  // cc->preMutate(iface);
                    bldr.op(invokeOW.op());                         // iface.v(newV);
                    bldr.op(CoreOp.invoke(MUTATE.post, cc, iface)); // cc->postMutate(iface)
                } else if (invokeOW.isIfaceAccessor()) {            // iface.v()
                    Value iface = bldrCntxt.getValue(invokeOW.operandNAsValue(0));
                    bldr.op(CoreOp.invoke(ACCESS.pre, cc, iface));  // cc->preAccess(iface);
                    bldr.op(invokeOW.op());                         // iface.v();
                    bldr.op(CoreOp.invoke(ACCESS.post, cc, iface)); // cc->postAccess(iface) } else {
                } else if (invokeOW.isComputeContextMethod() || invokeOW.isRawKernelCall()) { //dispatchKernel
                    bldr.op(invokeOW.op());
                } else {
                    List<Value> list = invokeOW.op().operands();
                 //   System.out.println("args "+list.size());
                    if (!list.isEmpty()) {
                       // System.out.println("method "+invokeOW.method());
                        Annotation[][] parameterAnnotations = invokeOW.method().getParameterAnnotations();
                        boolean isVirtual = list.size()>parameterAnnotations.length;
                     //   System.out.println("params length"+parameterAnnotations.length);
                        List<TypeAndAccess> typeAndAccesses = new ArrayList<>();

                            for (int i = isVirtual?1:0; i < list.size(); i++) {
                                typeAndAccesses.add(TypeAndAccess.of(
                                        parameterAnnotations[i-(isVirtual?1:0)],
                                        list.get(i)));
                            }
                        List<PrePost> prePosts = new ArrayList<>();
                        typeAndAccesses.stream()
                                .filter(typeAndAccess -> typeAndAccess.isIface(prevFOW.lookup))//InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, typeAndAccess.javaType))
                                .forEach(typeAndAccess -> {
                                     if (typeAndAccess.ro()) {
                                         bldr.op(CoreOp.invoke(ACCESS.pre, cc,  bldrCntxt.getValue(typeAndAccess.value)));
                                  //   }else if (typeAndAccess.wo()||typeAndAccess.rw()) {
                                    //     bldr.op(CoreOp.invoke(MUTATE.pre, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                     }else {
                                         bldr.op(CoreOp.invoke(MUTATE.pre, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                     }
                                });
                        //  invokeOW.op().operands().stream()
                        // .filter(value -> value.type() instanceof JavaType javaType && InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, javaType))
                        //  .forEach(value ->
                        //          bldr.op(CoreOp.invoke(ESCAPE.pre, cc, bldrCntxt.getValue(value)))
                        //  );
                        bldr.op(invokeOW.op());
                        typeAndAccesses.stream()
                                .filter(typeAndAccess -> InvokeOpWrapper.isIfaceUsingLookup(prevFOW.lookup, typeAndAccess.javaType))
                                .forEach(typeAndAccess -> {
                                    if (typeAndAccess.ro()) {
                                        bldr.op(CoreOp.invoke(ACCESS.post, cc,  bldrCntxt.getValue(typeAndAccess.value)));
                                 //   }else if (typeAndAccess.rw() || typeAndAccess.wo()) {
                                 //       bldr.op(CoreOp.invoke(MUTATE.post, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                    }else {
                                        bldr.op(CoreOp.invoke(MUTATE.post, cc, bldrCntxt.getValue(typeAndAccess.value)));
                                    }
                                });
                    }else{
                        bldr.op(invokeOW.op());
                    }
                }
                return bldr;
            });
            if (show) {
                System.out.println("COMPUTE entrypoint after injecting buffer tracking...");
                returnFOW.op().writeTo(System.out);
            }
        }else{
            if (show) {
                System.out.println("COMPUTE entrypoint (we will not be injecting buffer tracking...)...");
                returnFOW.op().writeTo(System.out);
            }
        }
        computeMethod.funcOpWrapper(returnFOW);
        return returnFOW;
    }
}
