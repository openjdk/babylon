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
import hat.optools.FuncOpWrapper;

import java.lang.reflect.Method;
import jdk.incubator.code.java.lang.reflect.code.type.MethodRef;
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

    public Stream<MethodCall> callStream() {
        return methodRefToMethodCallMap.values().stream();
    }

    public interface Resolved {
        FuncOpWrapper funcOpWrapper();

        void funcOpWrapper(FuncOpWrapper funcOpWrapper);
    }

    public interface Unresolved {
    }

    public abstract static class MethodCall {

        public final Method method;
        public final Class<?> declaringClass;
        public CallGraph<?> callGraph;
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
        private FuncOpWrapper funcOpWrapper;

        ResolvedMethodCall(CallGraph<?> callGraph, MethodRef targetMethodRef, Method method, FuncOpWrapper funcOpWrapper) {
            super(callGraph, targetMethodRef, method);
            this.funcOpWrapper = funcOpWrapper;
        }

        @Override
        public FuncOpWrapper funcOpWrapper() {
            return funcOpWrapper;
        }

        @Override
        public void funcOpWrapper(FuncOpWrapper funcOpWrapper) {
            this.funcOpWrapper = funcOpWrapper;
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
