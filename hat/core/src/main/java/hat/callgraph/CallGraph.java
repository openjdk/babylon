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
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.lang.reflect.Method;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

public abstract class CallGraph<E extends Entrypoint> {
    public final ComputeContext computeContext;
    public final E entrypoint;
    public final Set<MethodCall> calls = new HashSet<>();
    public final Map<MethodRef, MethodCall> methodRefToMethodCallMap = new LinkedHashMap<>();
    public CoreOp.ModuleOp moduleOp;

    // Todo: We should phase these out. We can also use Config.....
    public final static boolean noModuleOp = Boolean.getBoolean("noModuleOp");
    public final static boolean bufferTagging = Boolean.getBoolean("bufferTagging");
    public Stream<MethodCall> callStream() {
        return methodRefToMethodCallMap.values().stream();
    }

    public abstract boolean filterCalls(CoreOp.FuncOp f, JavaOp.InvokeOp invokeOp, Method method, MethodRef methodRef, Class<?> javaRefTypeClass);

    public interface Resolved {
        CoreOp.FuncOp funcOp();
        void funcOp(CoreOp.FuncOp funcOp);
    }

    public interface Unresolved {
    }

    public abstract static class MethodCall {
        public CallGraph<?> callGraph;
        public final Method method;
        public final Class<?> declaringClass;
        public final Set<MethodCall> calls = new HashSet<>();
        public final Set<MethodCall> callers = new HashSet<>();
        public final MethodRef targetMethodRef;
        public boolean closed = false;
        public int rank = 0;

        MethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method) {
            this.callGraph = callGraph;
            this.targetMethodRef = targetMethodRef;
            this.method = method;
            this.declaringClass = method.getDeclaringClass();
        }


        public void dump(String indent) {
            System.out.println(indent + ((targetMethodRef == null ? "EntryPoint" : targetMethodRef)));
            calls.forEach(call -> call.dump(indent + " -> "));
        }


        public void addCall(MethodCall methodCall) {
            callGraph.calls.add(methodCall);
            methodCall.callers.add(this);
            this.calls.add(methodCall);
        }

        protected void rankRecurse(int value) {
            calls.forEach(c -> c.rankRecurse(value + 1));
            if (value > this.rank) {
                this.rank = value;
            }
        }

        public void rankRecurse() {
            rankRecurse(0);
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
