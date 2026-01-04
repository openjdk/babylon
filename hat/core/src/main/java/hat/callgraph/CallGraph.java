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

import hat.ComputeContext;
import hat.Config;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.Invoke;
import optkl.util.CallSite;
import optkl.util.carriers.LookupCarrier;
import optkl.OpTkl;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static optkl.Invoke.invokeOpHelper;

public abstract class CallGraph<E extends Entrypoint> implements LookupCarrier {

    @Override
    public final MethodHandles.Lookup lookup() {
        return computeContext.lookup();
    }


    public final ComputeContext computeContext;
    public final E entrypoint;
    public final Set<MethodCall> calls = new HashSet<>();
    public final Map<MethodRef, MethodCall> methodRefToMethodCallMap = new LinkedHashMap<>();
    private CoreOp.ModuleOp moduleOp;
    public CoreOp.ModuleOp getModuleOp(){
        return this.moduleOp;
    }

    public void setModuleOp(CoreOp.ModuleOp moduleOp){
        this.moduleOp = moduleOp;
    }


     CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup lookup,
                                                        CoreOp.FuncOp entry) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, CoreOp.FuncOp f) {}

        Deque<RefAndFunc> work = new ArrayDeque<>();

        Invoke.stream(lookup,entry)
                .forEach(invoke -> {
                Class<?> javaRefTypeClass = invoke.classOrThrow();
                try {
                    var method = invoke.op().invokeDescriptor().resolveToMethod(lookup);
                    CoreOp.FuncOp f = Op.ofMethod(method).orElse(null);
                    // TODO filter calls has side effects we may need another call. We might just check the map.

                    if (f != null && !filterCalls(f, invoke.op(), method, invoke.op().invokeDescriptor(), javaRefTypeClass)) {
                        work.push(new RefAndFunc(invoke.op().invokeDescriptor(),  f));
                    }
                } catch (ReflectiveOperationException _) {
                    throw new IllegalStateException("Could not resolve invokeWrapper to method");
                }
        });

        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (funcsVisited.add(rf.r)) {
                // TODO:is this really transforming? it seems to be creating a new funcop.. Oh I guess for the new ModuleOp?
                CoreOp.FuncOp tf = rf.f.transform(rf.r.name(), (blockBuilder, op) -> {
                    if (op instanceof JavaOp.InvokeOp iop) {
                        try {
                            Method invokeOpCalledMethod = iop.invokeDescriptor().resolveToMethod(lookup);
                            if (invokeOpCalledMethod instanceof Method m) {
                                CoreOp.FuncOp f = Op.ofMethod(m).orElse(null);
                                if (f != null) {
                                    RefAndFunc call = new RefAndFunc(iop.invokeDescriptor(), f);
                                    work.push(call);
                                    Op.Result result = blockBuilder.op(CoreOp.funcCall(
                                            call.r.name(),
                                            call.f.invokableType(),
                                            blockBuilder.context().getValues(iop.operands())));
                                    blockBuilder.context().mapValue(op.result(), result);
                                    return blockBuilder;
                                }
                            }
                        } catch (ReflectiveOperationException _) {
                            throw new IllegalStateException("Could not resolve invokeWrapper to method");
                        }
                    }
                    blockBuilder.op(op);
                    return blockBuilder;
                });

                funcs.addFirst(tf);
            }
        }

        return CoreOp.module(funcs);
    }


    public abstract boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass);

    public Config config() {
        return computeContext.config();
    }

    public interface Resolved {
        CoreOp.FuncOp funcOp();
        void funcOp(CoreOp.FuncOp funcOp);
    }

    public interface Unresolved {
    }

    public abstract static class MethodCall {
        public CallGraph<?> callGraph;
        public final Method method;
        public final MethodRef targetMethodRef;

        MethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method) {
            this.callGraph = callGraph;
            this.targetMethodRef = targetMethodRef;
            this.method = method;
        }
    }

    public abstract static class ResolvedMethodCall extends MethodCall implements Resolved {
        private CoreOp.FuncOp funcOp;

        ResolvedMethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method,  CoreOp.FuncOp funcOp) {
            super(callGraph, targetMethodRef, method);
            this.funcOp = funcOp;
        }

        @Override
        public CoreOp.FuncOp funcOp() {
            return funcOp;
        }

        @Override
        public void funcOp(CoreOp.FuncOp funcOp) {
            this.funcOp = funcOp;
        }
    }


    public abstract static class UnresolvedMethodCall extends MethodCall implements Unresolved {
        UnresolvedMethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method) {
            super(callGraph, targetMethodRef, method);
        }
    }


    CallGraph(ComputeContext computeContext, E entrypoint) {
        this.computeContext = computeContext;
        this.entrypoint = entrypoint;
    }
}
