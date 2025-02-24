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

package hat.backend.jextracted;

import hat.ComputeContext;
import hat.backend.ffi.FFIBackendDriver;
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.ifacemapper.BoundSchema;
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

import java.lang.foreign.Arena;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.ESCAPE;
import static hat.ComputeContext.WRAPPER.MUTATE;

public abstract class JExtractedBackend extends FFIBackendDriver {

    public final Arena arena = Arena.global();


    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper, BoundSchema<T> boundSchema) {
        return segmentMapper.allocate(arena, boundSchema);
    }

    public JExtractedBackend(String libName) {
        super(libName);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        if (computeContext.computeCallGraph.entrypoint.lowered == null) {
            computeContext.computeCallGraph.entrypoint.lowered = computeContext.computeCallGraph.entrypoint.funcOpWrapper().lower();
        }


        boolean interpret = false;
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
    }

    static void wrapInvoke(InvokeOpWrapper iow, Block.Builder bldr, ComputeContext.WRAPPER wrapper, Value cc, Value iface) {
        bldr.op(CoreOp.invoke(wrapper.pre, cc, iface));
        bldr.op(iow.op());
        bldr.op(CoreOp.invoke(wrapper.post, cc, iface));
    }

    protected static FuncOpWrapper injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
        FuncOpWrapper prevFOW = computeMethod.funcOpWrapper();
        FuncOpWrapper returnFOW = prevFOW;
        boolean transform = true;
        if (transform) {
            System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
            returnFOW.op().writeTo(System.out);
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
                    invokeOW.op().operands().stream()
                            .filter(value -> value.type() instanceof JavaType javaType && InvokeOpWrapper.isIface(javaType))
                            .forEach(value ->
                                    bldr.op(CoreOp.invoke(ESCAPE.pre, cc, bldrCntxt.getValue(value)))
                            );
                    bldr.op(invokeOW.op());
                    invokeOW.op().operands().stream()
                            .filter(value -> value.type() instanceof JavaType javaType && InvokeOpWrapper.isIface(javaType))
                            .forEach(value -> bldr.op(
                                    CoreOp.invoke(ESCAPE.post, cc, bldrCntxt.getValue(value)))
                            );
                }
                return bldr;
            });
            System.out.println("COMPUTE entrypoint after injecting buffer tracking...");
            returnFOW.op().writeTo(System.out);
        }
        computeMethod.funcOpWrapper(returnFOW);
        return returnFOW;
    }
}
