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
package hat.callgraph;

import hat.buffer.Buffer;
import hat.optools.FuncOpWrapper;
import hat.optools.OpWrapper;

import java.lang.reflect.Method;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.MethodRef;
import java.util.Optional;
import java.util.stream.Stream;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
            super(callGraph, targetMethodRef, method, funcOpWrapper);
        }
    }

    public static class KernelReachableUnresolvedMethodCall extends UnresolvedMethodCall implements KernelReachable {
        KernelReachableUnresolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }


    public static class KernelReachableUnresolvedIfaceMappedMethodCall extends KernelReachableUnresolvedMethodCall {
        KernelReachableUnresolvedIfaceMappedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public static class KidAccessor extends MethodCall {
        KidAccessor(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }

    public final ComputeCallGraph computeCallGraph;

    public Stream<KernelReachableResolvedMethodCall> kernelReachableResolvedStream() {
        return methodRefToMethodCallMap.values().stream()
                .filter(call -> call instanceof KernelReachableResolvedMethodCall)
                .map(kernelReachable -> (KernelReachableResolvedMethodCall) kernelReachable);
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, MethodRef methodRef, Method method, FuncOpWrapper funcOpWrapper) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(null, methodRef, method, funcOpWrapper));
        entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
    }

    void updateDag(KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {
        /*
         * A ResolvedKernelMethodCall (entrypoint or java  method reachable from a compute entrypojnt)  has the following calls
         * <p>
         * 1) java calls to compute class static functions provided they follow the kernel restrictions
         *    a) we must have the code model available for these and must extend the dag
         * 2) calls to buffer based interface mappings
         *    a) getters (return non void)
         *    b) setters (return void)
         * 3) calls on the NDRange id
         */

        kernelReachableResolvedMethodCall.funcOpWrapper().selectCalls(invokeOpWrapper -> {
            MethodRef methodRef = invokeOpWrapper.methodRef();
            Class<?> javaRefTypeClass = invokeOpWrapper.javaRefClass().orElseThrow();
            Method invokeOpCalledMethod = invokeOpWrapper.method(this.computeContext.accelerator.lookup);
            if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
                //System.out.println("kernel reachable iface mapped buffer call  -> " + methodRef);
                kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new KernelReachableUnresolvedIfaceMappedMethodCall(this, methodRef, invokeOpCalledMethod)
                ));
            } else if (entrypoint.method.getDeclaringClass().equals(javaRefTypeClass)) {
                Optional<CoreOp.FuncOp> optionalFuncOp = invokeOpCalledMethod.getCodeModel();
                if (optionalFuncOp.isPresent()) {
                    //System.out.println("A call to a method on the kernel class which we have code model for " + methodRef);
                    kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                            new KernelReachableResolvedMethodCall(this, methodRef, invokeOpCalledMethod, OpWrapper.wrap(optionalFuncOp.get())
                            )));
                } else {
                    // System.out.println("A call to a method on the compute class which we DO NOT have code model for " + methodRef);
                    kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                            new KernelReachableUnresolvedMethodCall(this, methodRef, invokeOpCalledMethod)
                    ));
                }
            } else {
                //  System.out.println("A call to a method on the compute class which we DO NOT have code model for " + methodRef);
                kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(methodRef, _ ->
                        new KernelReachableUnresolvedMethodCall(this, methodRef, invokeOpCalledMethod)
                ));
                // System.out.println("Were we expecting " + methodRef + " here ");
            }

        });

        boolean updated = true;
        kernelReachableResolvedMethodCall.closed = true;
        while (updated) {
            updated = false;
            var unclosed = callStream().filter(m -> !m.closed).findFirst();
            if (unclosed.isPresent()) {
                if (unclosed.get() instanceof KernelReachableResolvedMethodCall reachableResolvedMethodCall) {
                    updateDag(reachableResolvedMethodCall);
                } else {
                    unclosed.get().closed = true;
                }
                updated = true;
            }
        }
    }

    KernelCallGraph close() {
        updateDag(entrypoint);
        // now lets sort the MethodCalls into a dependency list
        calls.forEach(m -> m.rank = 0);
        entrypoint.rankRecurse();
        return this;
    }
}
