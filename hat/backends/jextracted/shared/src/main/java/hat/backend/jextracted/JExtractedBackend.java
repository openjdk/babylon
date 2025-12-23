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
import hat.Config;
import optkl.CallSite;
import optkl.OpTkl;
import optkl.ifacemapper.Buffer;
import hat.callgraph.CallGraph;
import optkl.ifacemapper.MappableIface;
import optkl.FuncOpParams;
import hat.optools.OpTk;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

import static hat.ComputeContext.WRAPPER.ACCESS;
import static hat.ComputeContext.WRAPPER.MUTATE;
import static hat.optools.OpTk.isComputeContextMethod;
import static hat.optools.OpTk.isIfaceBufferMethod;
import static optkl.OpTkl.classTypeToTypeOrThrow;
import static optkl.OpTkl.isAssignable;
import static optkl.OpTkl.javaReturnType;
import static optkl.OpTkl.lower;
import static optkl.OpTkl.transform;

public abstract class JExtractedBackend extends JExtractedBackendDriver {

    public JExtractedBackend(Arena arena, MethodHandles.Lookup lookup,Config config, String libName) {
        super(arena,lookup,config,libName);
    }

    public void dispatchCompute(ComputeContext computeContext, Object... args) {
        var here = CallSite.of(JExtractedBackend.class, "dispatchCompuet");
        if (computeContext.computeEntrypoint().lowered == null) {
            computeContext.computeEntrypoint().lowered =
                    lower(here, computeContext.computeEntrypoint().funcOp());
        }
        boolean interpret = false;
        if (interpret) {
            Interpreter.invoke(computeContext.lookup(), computeContext.computeEntrypoint().lowered, args);
        } else {
            try {
                if (computeContext.computeEntrypoint().mh == null) {
                    computeContext.computeEntrypoint().mh = BytecodeGenerator.generate(computeContext.lookup(), computeContext.computeEntrypoint().lowered);
                }
                computeContext.computeEntrypoint().mh.invokeWithArguments(args);
            } catch (Throwable e) {
                System.out.println(computeContext.computeEntrypoint().lowered.toText());
                throw new RuntimeException(e);
            }
        }
    }

    // This code should be common with ffi-shared probably should be pushed down into another lib?
    protected static CoreOp.FuncOp injectBufferTracking(CallGraph.ResolvedMethodCall computeMethod) {
      //  System.out.println("COMPUTE entrypoint before injecting buffer tracking...");
       // System.out.println(computeMethod.funcOp().toText());
        var lookup = computeMethod.callGraph.lookup();
        // TODO : can't we get this from somewhere maybe it should be capturein the compute method?
        var paramTable = new FuncOpParams(computeMethod.funcOp());
        var here = CallSite.of(JExtractedBackend.class, "injectBufferTracking");
        var transformedFuncOp = transform(here,computeMethod.funcOp(),(bldr, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Value computeContext = bldr.context().getValue(paramTable.list().getFirst().parameter);
                if (isIfaceBufferMethod(lookup, invokeOp) && javaReturnType(invokeOp).equals(JavaType.VOID)) {                    // iface.v(newV)
                    Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                    bldr.op(JavaOp.invoke(MUTATE.pre, computeContext, iface));  // cc->preMutate(iface);
                    bldr.op(invokeOp);                                          // iface.v(newV);
                    bldr.op(JavaOp.invoke(MUTATE.post, computeContext, iface)); // cc->postMutate(iface)
                } else if (isIfaceBufferMethod(lookup, invokeOp)
                        //&& !OpTk.javaReturnType(invokeOp).equals(JavaType.VOID) not sure we need this
                        && javaReturnType(invokeOp) instanceof ClassType returnClassType
                        && classTypeToTypeOrThrow(lookup, returnClassType) instanceof Class<?> type
                        && Buffer.class.isAssignableFrom(type)
                ) {            // iface.v()
                    Value iface = bldr.context().getValue(invokeOp.operands().getFirst());
                    bldr.op(JavaOp.invoke(ACCESS.pre, computeContext, iface));  // cc->preAccess(iface);
                    bldr.op(invokeOp);                                          // iface.v();
                    bldr.op(JavaOp.invoke(ACCESS.post, computeContext, iface)); // cc->postAccess(iface) } else {
                } else if (isComputeContextMethod(lookup, invokeOp) || OpTk.isKernelContextInvokeOp(lookup, invokeOp,OpTkl.AnyInvoke)) { //dispatchKernel
                    bldr.op(invokeOp);
                } else {
                    invokeOp.operands().stream()
                            .filter(val -> val.type() instanceof JavaType javaType && isAssignable(lookup, javaType, MappableIface.class))
                            .forEach(val -> bldr.op(JavaOp.invoke(MUTATE.pre, computeContext, bldr.context().getValue(val))));
                    bldr.op(invokeOp);
                    invokeOp.operands().stream()
                            .filter(val -> val.type() instanceof JavaType javaType && isAssignable(lookup, javaType, MappableIface.class))
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
