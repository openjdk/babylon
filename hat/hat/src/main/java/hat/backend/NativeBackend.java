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
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.ifacemapper.SegmentMapper;
import hat.optools.FuncOpWrapper;

import java.lang.foreign.Arena;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;

public abstract class NativeBackend extends NativeBackendDriver {

    public final Arena arena = Arena.global();


    @Override
    public <T extends Buffer> T allocate(SegmentMapper<T> segmentMapper){
        return segmentMapper.allocate(arena);
    }
    public NativeBackend(String libName) {
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

    static FuncOpWrapper injectBufferTracking(CallGraph.ResolvedMethodCall resolvedMethodCall) {
        FuncOpWrapper originalFuncOpWrapper = resolvedMethodCall.funcOpWrapper();

        var transformed = originalFuncOpWrapper.transformInvokes((builder, invokeOpWrapper) -> {
                    if (invokeOpWrapper.isIfaceBufferMethod()) {
                        CopyContext cc = builder.context();
                        Value computeContext = cc.getValue(originalFuncOpWrapper.parameter(0));
                        Value receiver = cc.getValue(invokeOpWrapper.operandNAsValue(0));

                        if (invokeOpWrapper.isIfaceMutator()) {
                            builder.op(CoreOp.invoke(ComputeContext.M_CC_PRE_MUTATE, computeContext, receiver));
                            builder.op(invokeOpWrapper.op());
                            builder.op(CoreOp.invoke(ComputeContext.M_CC_POST_MUTATE, computeContext, receiver));
                        } else if (invokeOpWrapper.isIfaceAccessor()) {
                            builder.op(CoreOp.invoke(ComputeContext.M_CC_PRE_ACCESS, computeContext, receiver));
                            builder.op(invokeOpWrapper.op());
                            builder.op(CoreOp.invoke(ComputeContext.M_CC_POST_ACCESS, computeContext, receiver));
                        } else {
                            builder.op(invokeOpWrapper.op());
                        }
                    } else {
                        builder.op(invokeOpWrapper.op());
                    }
                    return builder;
                }
        );
        // transformed.op().writeTo(System.out);
        resolvedMethodCall.funcOpWrapper(transformed);
        return transformed;
    }
}
