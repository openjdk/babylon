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
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.OpHelper.Invoke;
import optkl.util.carriers.LookupCarrier;

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

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public abstract class CallGraph<E extends Entrypoint> implements LookupCarrier {
    @Override
    public final MethodHandles.Lookup lookup() {
        return computeContext.lookup();
    }

    public final ComputeContext computeContext;
    public final E entrypoint;
    public final Set<AbstractMethodCall> calls = new HashSet<>();
    public final Map<MethodRef, AbstractMethodCall> methodRefToMethodCallMap = new LinkedHashMap<>();

    private CoreOp.ModuleOp moduleOp;

    public CoreOp.ModuleOp getModuleOp() {
        return this.moduleOp;
    }

    public void setModuleOp(CoreOp.ModuleOp moduleOp) {
        this.moduleOp = moduleOp;
    }

    CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup lookup, Method entryMethod, CoreOp.FuncOp entry) {
        record RefAndFunc(MethodRef methodRef, CoreOp.FuncOp funcOp) {
        }

        Deque<RefAndFunc> work = new ArrayDeque<>();

        Invoke.stream(lookup, entry).forEach(invoke -> {
            if (invoke.targetMethodModelOrNull() instanceof CoreOp.FuncOp funcOp && !filterCalls(funcOp, invoke)) {
                work.push(new RefAndFunc(invoke.op().invokeDescriptor(), funcOp));
            }
        });

        List<CoreOp.FuncOp> moduleFuncOps = new ArrayList<>();
        LinkedHashSet<MethodRef> setOfVisitedMethodRefs = new LinkedHashSet<>();
        while (!work.isEmpty() && work.pop() instanceof RefAndFunc refAndFunc && setOfVisitedMethodRefs.add(refAndFunc.methodRef)) {
            CoreOp.FuncOp tf = refAndFunc.funcOp.transform(refAndFunc.methodRef.name(), (blockBuilder, op) -> {
                if (invoke(lookup, op) instanceof Invoke iop && iop.targetMethodModelOrNull() instanceof CoreOp.FuncOp funcOp) {
                    RefAndFunc call = new RefAndFunc(iop.op().invokeDescriptor(), funcOp);
                    work.push(call);
                    blockBuilder.context().mapValue(op.result(), blockBuilder.op(copyLocation(funcOp, CoreOp.funcCall(
                            call.methodRef.name(),
                            call.funcOp.invokableType(),
                            blockBuilder.context().getValues(iop.op().operands())))));
                } else {
                    assert op != null;
                    blockBuilder.op(op);
                }
                return blockBuilder;
            });
            moduleFuncOps.addFirst(tf);
        }

        return CoreOp.module(moduleFuncOps);
    }


    public abstract boolean filterCalls(CoreOp.FuncOp f, Invoke invoke);

    public Config config() {
        return computeContext.config();
    }

    public interface Resolved {
        CoreOp.FuncOp funcOp();

        void funcOp(CoreOp.FuncOp funcOp);
    }

    public interface Unresolved {
    }

    public abstract static class AbstractMethodCall implements MethodCall {
        public CallGraph<?> callGraph;
        private final Method method;

        AbstractMethodCall(CallGraph<?> callGraph, Method method) {
            this.callGraph = callGraph;
            this.method = method;
        }


        public Method method() {
            return this.method;
        }

    }

    public abstract static class ResolvedMethodCall extends AbstractMethodCall implements Resolved {
        private CoreOp.FuncOp funcOp;

        ResolvedMethodCall(CallGraph<?> callGraph, Method method, CoreOp.FuncOp funcOp) {
            super(callGraph, method);
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


    public abstract static class UnresolvedMethodCall extends AbstractMethodCall implements Unresolved {
        UnresolvedMethodCall(CallGraph<?> callGraph, Method method) {
            super(callGraph, method);
        }
    }


    CallGraph(ComputeContext computeContext, E entrypoint) {
        this.computeContext = computeContext;
        this.entrypoint = entrypoint;
    }
}
