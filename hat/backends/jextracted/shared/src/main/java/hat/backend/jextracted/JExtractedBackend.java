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
import hat.buffer.Buffer;
import hat.callgraph.CallGraph;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.SegmentMapper;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.foreign.Arena;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;

public abstract class JExtractedBackend extends JExtractedBackendDriver {

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
            computeContext.computeCallGraph.entrypoint.lowered =
                    OpTk.lower(computeContext.computeCallGraph.entrypoint.funcOp());
        }
        boolean interpret = false;
        if (interpret) {
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
    }

    // This code should be common with ffi-shared probably should be pushed down into another lib?
    protected static CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
        System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
        System.out.println(computeMethod.funcOp().toText());
        var paramTable = new FuncOpParams(computeMethod.funcOp());
        var lookup = computeMethod.callGraph.computeContext.accelerator.lookup;
        var transformedFuncOp = computeMethod.funcOp().transform((bldr, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Value computeContext = bldr.context().getValue(paramTable.list().getFirst().parameter);
                if (OpTk.isIfaceBufferMethod(lookup, invokeOp) && OpTk.javaReturnType(invokeOp).equals(JavaType.VOID)) {                    // iface.v(newV)
                    Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                    bldr.op(JavaOp.invoke(MUTATE.pre, computeContext, iface));  // cc->preMutate(iface);
                    bldr.op(invokeOp);                                          // iface.v(newV);
                    bldr.op(JavaOp.invoke(MUTATE.post, computeContext, iface)); // cc->postMutate(iface)
                } else if (OpTk.isIfaceBufferMethod(lookup, invokeOp)
                        //&& !OpTk.javaReturnType(invokeOp).equals(JavaType.VOID) not sure we need this
                        && OpTk.javaReturnType(invokeOp) instanceof ClassType returnClassType
                        && OpTk.classTypeToTypeOrThrow(lookup, returnClassType) instanceof Class<?> type
                        && Buffer.class.isAssignableFrom(type)
                ) {            // iface.v()
                    Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                    bldr.op(JavaOp.invoke(ACCESS.pre, computeContext, iface));  // cc->preAccess(iface);
                    bldr.op(invokeOp);                                          // iface.v();
                    bldr.op(JavaOp.invoke(ACCESS.post, computeContext, iface)); // cc->postAccess(iface) } else {
                } else if (OpTk.isComputeContextMethod(lookup, invokeOp) || OpTk.isKernelContextMethod(lookup, invokeOp)) { //dispatchKernel
                    bldr.op(invokeOp);
                } else {
                    invokeOp.operands().stream()
                            .filter(val -> val.type() instanceof JavaType javaType && OpTk.isAssignable(lookup, javaType, MappableIface.class))
                            .forEach(val -> bldr.op(JavaOp.invoke(MUTATE.pre, computeContext, bldr.context().getValue(val))));
                    bldr.op(invokeOp);
                    invokeOp.operands().stream()
                            .filter(val -> val.type() instanceof JavaType javaType && OpTk.isAssignable(lookup, javaType, MappableIface.class))
                            .forEach(val -> bldr.op(JavaOp.invoke(MUTATE.post, computeContext, bldr.context().getValue(val))));
                }
                return bldr;
            } else {
                bldr.op(op);
            }
            return bldr;
        });
        System.out.println("COMPUTE entrypoint after injecting buffer tracking...");
        System.out.println(transformedFuncOp.toText());
        computeMethod.funcOp(transformedFuncOp);
        return transformedFuncOp;
    }
}
