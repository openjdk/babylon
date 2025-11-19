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

import hat.BufferTagger;
import hat.buffer.Buffer;
import hat.optools.OpTk;
import hat.phases.HATDialectifyTier;
import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Stream;

public class KernelCallGraph extends CallGraph<KernelEntrypoint> {
    public final ComputeCallGraph computeCallGraph;
    public final Map<MethodRef, MethodCall> bufferAccessToMethodCallMap = new LinkedHashMap<>();
    public final List<BufferTagger.AccessType> bufferAccessList;
    public boolean usesArrayView;

    public interface KernelReachable {
    }

    public static class KernelReachableResolvedMethodCall extends ResolvedMethodCall implements KernelReachable {
        public KernelReachableResolvedMethodCall(CallGraph<KernelEntrypoint> callGraph, MethodRef targetMethodRef, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method, funcOp);
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

    public Stream<KernelReachableResolvedMethodCall> kernelReachableResolvedStream() {
        return methodRefToMethodCallMap.values().stream()
                .filter(call -> call instanceof KernelReachableResolvedMethodCall)
                .map(kernelReachable -> (KernelReachableResolvedMethodCall) kernelReachable);
    }

    KernelCallGraph(ComputeCallGraph computeCallGraph, MethodRef methodRef, Method method, CoreOp.FuncOp funcOp) {
        super(computeCallGraph.computeContext, new KernelEntrypoint(null, methodRef, method, funcOp));
        entrypoint.callGraph = this;
        this.computeCallGraph = computeCallGraph;
        bufferAccessList = BufferTagger.getAccessList(computeContext.accelerator.lookup, entrypoint.funcOp());
        usesArrayView = false;
        CoreOp.ModuleOp initialModuleOp = OpTk.createTransitiveInvokeModule(computeContext.accelerator.lookup, entrypoint.funcOp(), this);
        HATDialectifyTier tier = new HATDialectifyTier(computeContext.accelerator);
        CoreOp.FuncOp initialEntrypointFuncOp = tier.apply(entrypoint.funcOp());
        entrypoint.funcOp(initialEntrypointFuncOp);
        List<CoreOp.FuncOp> initialFuncOps = new ArrayList<>();
        initialModuleOp.functionTable().forEach((_, accessableFuncOp) ->
                initialFuncOps.add( tier.apply(accessableFuncOp))
        );
        setModuleOp(CoreOp.module(initialFuncOps));
    }
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
    void oldUpdateDag(KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {

        var here = OpTk.CallSite.of(KernelCallGraph.class,"updateDag");
        OpTk.traverse(here, kernelReachableResolvedMethodCall.funcOp(), (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
              //  MethodRef methodRef = invokeOp.invokeDescriptor();
                Class<?> javaRefTypeClass = OpTk.javaRefClassOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                Method invokeOpCalledMethod = OpTk.methodOrThrow(kernelReachableResolvedMethodCall.callGraph.computeContext.accelerator.lookup,invokeOp);
                if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
                        kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedIfaceMappedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                } else if (entrypoint.method.getDeclaringClass().equals(javaRefTypeClass)) {
                    Optional<CoreOp.FuncOp> optionalFuncOp = Op.ofMethod(invokeOpCalledMethod);
                    if (optionalFuncOp.isPresent()) {
                             kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableResolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod, optionalFuncOp.get()
                                )));
                    } else {
                           kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                                new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                        ));
                    }
                } else {
                       kernelReachableResolvedMethodCall.addCall(methodRefToMethodCallMap.computeIfAbsent(invokeOp.invokeDescriptor(), _ ->
                            new KernelReachableUnresolvedMethodCall(this, invokeOp.invokeDescriptor(), invokeOpCalledMethod)
                    ));
                    // System.out.println("Were we expecting " + methodRef + " here ");
                }
            }
            return map;
        });

        boolean updated = true;
        kernelReachableResolvedMethodCall.closed = true;
        while (updated) {
            updated = false;
            var unclosed = callStream().filter(m -> !m.closed).findFirst();
            if (unclosed.isPresent()) {
                if (unclosed.get() instanceof KernelReachableResolvedMethodCall reachableResolvedMethodCall) {
                    oldUpdateDag(reachableResolvedMethodCall);
                } else {
                    unclosed.get().closed = true;
                }
                updated = true;
            }
        }
    }


    @Override
    public boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass) {
        if (Buffer.class.isAssignableFrom(javaRefTypeClass)) {
            // TODO this side effect seems scary
            bufferAccessToMethodCallMap.computeIfAbsent(methodRef, _ ->
                    new KernelReachableUnresolvedIfaceMappedMethodCall(this, methodRef, method)
            );
        } else {
            return false;
        }
        return true;
    }
/*
    public void nodialectifyToHat() {
        // Analysis Phases to transform the Java Code Model to a HAT Code Model

        // Main kernel
        // TODO we should not need the entrypoint handles seprately. !
        //{
            HATDialectifyTier tier = new HATDialectifyTier(computeContext.accelerator);
            CoreOp.FuncOp f = tier.run(entrypoint.funcOp());
            entrypoint.funcOp(f);
       // }
        // Reachable functions
      //  if (moduleOp != null) {
            List<CoreOp.FuncOp> funcs = new ArrayList<>();
            getModuleOp().functionTable().forEach((_, funcOp) -> {
                // ModuleOp is an Immutable Collection, thus, we need to create a new one from a
                // new list of methods
         //       HATDialectifyTier tier = new HATDialectifyTier(computeContext.accelerator);
                CoreOp.FuncOp fn = tier.run(funcOp);
                funcs.add(fn);
            });
            // TODO: can we just replaced moduleOp here.  What if another side table has a prev reference with non transformed funcOps?
             setModuleOp(CoreOp.module(funcs));
        //} else {
          //  throw new IllegalStateException("moduleOp is null");
           kernelReachableResolvedStream().forEach((kernel) -> {
                HatDialectifyTier tier = new HatDialectifyTier(computeContext.accelerator);
                CoreOp.FuncOp f = tier.run(kernel.funcOp());
                kernel.funcOp(f);
            });
        //}
    }

    public void noconvertArrayView() {
        CoreOp.FuncOp entry = convertArrayViewForFunc(computeContext.accelerator.lookup, entrypoint.funcOp());
        entrypoint.funcOp(entry);

       // if (moduleOp != null) {
            List<CoreOp.FuncOp> funcs = new ArrayList<>();
            getModuleOp().functionTable().forEach((_, kernelOp) -> {
                CoreOp.FuncOp f = convertArrayViewForFunc(computeContext.accelerator.lookup, kernelOp);
                funcs.add(f);
            });
            setModuleOp(CoreOp.module(funcs));
       // } else {
         //   kernelReachableResolvedStream().forEach((method) -> {
           //     CoreOp.FuncOp f = convertArrayViewForFunc(computeContext.accelerator.lookup, method.funcOp());
             //   method.funcOp(f);
            //});
       // }
    }
*/
}
