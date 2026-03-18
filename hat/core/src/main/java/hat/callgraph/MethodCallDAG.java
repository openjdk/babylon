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

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.OpHelper;
import optkl.ifacemapper.Buffer;
import optkl.jdot.ui.JDot;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public class MethodCallDAG implements LookupCarrier {

    public MethodHandles.Lookup lookup;

    @Override
    public MethodHandles.Lookup lookup() {
        return lookup;
    }

    public void view() {
        JDot.digraph("dag", $ ->
                edges.forEach((l, r) ->
                        r.forEach(e ->
                                $.edge(l.funcOp.funcName(), e.funcOp.funcName())
                        )
                ));
    }

    public boolean isDag() {
        return edges.size()>1;
    }

    static public class MethodInfo{
        public  CoreOp.FuncOp funcOp;
        public int rank;
        public final MethodRef methodRef;
        public final Method method;
        MethodInfo(CoreOp.FuncOp funcOp, MethodRef methodRef, Method method){
            this.funcOp = funcOp;
            this.methodRef = methodRef;
            this.method = method;
        }

        @Override
        public int hashCode() {
            return Objects.hash(methodRef,method);
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            return o instanceof MethodInfo that &&
                    Objects.equals(methodRef, that.methodRef) && Objects.equals(method, that.method);
        }

        static MethodInfo of(CoreOp.FuncOp funcOp, MethodRef methodRef, Method method) {
            return new MethodInfo(funcOp, methodRef, method);
        }
    }

    final MethodInfo entryPoint;
    final CoreOp.FuncOp inlined;
    final Set<MethodInfo> set = new HashSet<>();
    final Map<MethodInfo, Set<MethodInfo>> edges = new HashMap<>();

    MethodCallDAG(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp funcOp, CoreOp.FuncOp inlined) {
        this.lookup = lookup;
        this.inlined = inlined;
        this.entryPoint = MethodInfo.of(funcOp, null, method);// we dont have a methodRef for the root
        set.add(this.entryPoint);
    }


    void addEdge(MethodInfo methodInfo, OpHelper.Invoke invoke) {
        var edge = MethodInfo.of(invoke.targetMethodModelOrNull(), invoke.op().invokeReference(), invoke.resolveMethodOrThrow());
        var edgeSet = edges.computeIfAbsent(methodInfo, _ -> new HashSet<>());
        if (edgeSet.add(edge)) {
            OpHelper.Invoke.stream(invoke.lookup(), edge.funcOp)
                    .filter(i -> i.targetMethodModelOrNull() != null)
                    .forEach(i -> addEdge(edge, i));
        }
    }

    static public MethodCallDAG of(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp entry, CoreOp.FuncOp inlined) {
        var dag = new MethodCallDAG(lookup, method, entry, inlined);
        OpHelper.Invoke.stream(lookup, entry)
                .filter(invoke -> invoke.targetMethodModelOrNull() != null)
                .forEach(i -> dag.addEdge(dag.entryPoint, i));
        return dag;
    }

    public void traverseDeclarationOrder(Consumer<MethodInfo> consumer) {
        Map<MethodInfo, Integer> outDegree = new HashMap<>();
        Map<MethodInfo, List<MethodInfo>> reverseEdges = new HashMap<>();
        Queue<MethodInfo> queue = new LinkedList<>();

        for (MethodInfo parent : edges.keySet()) {
            outDegree.put(parent, edges.get(parent).size());
            for (MethodInfo child : edges.get(parent)) {
                reverseEdges.computeIfAbsent(child, k -> new ArrayList<>()).add(parent);
                outDegree.putIfAbsent(child, 0);
            }
        }

        for (Map.Entry<MethodInfo, Integer> entry : outDegree.entrySet()) {
            if (entry.getValue() == 0) {
                queue.add(entry.getKey());
            }
        }

        while (!queue.isEmpty()) {
            MethodInfo current = queue.poll();
            consumer.accept(current);
            List<MethodInfo> parents = reverseEdges.getOrDefault(current, Collections.emptyList());
            for (MethodInfo parent : parents) {
                int remainingChildren = outDegree.get(parent) - 1;
                outDegree.put(parent, remainingChildren);
                if (remainingChildren == 0) {
                    queue.add(parent);
                }
            }
        }
    }

    public List<MethodInfo> declarationOrder() {
        List<MethodInfo> methodInfos = new ArrayList<>();
        traverseDeclarationOrder(methodInfos::add);
        return methodInfos;
    }


    public CoreOp.ModuleOp toModuleOp() {
        List<CoreOp.FuncOp> moduleFuncOps = new ArrayList<>();
        declarationOrder().forEach(methodInfo -> {
                    if (methodInfo.methodRef != null) {
                      //  String methodName = methodInfo.methodRef.name();
                       // FunctionType functionType = methodInfo.funcOp.invokableType();
                        CoreOp.FuncOp tf = methodInfo.funcOp.transform(methodInfo.methodRef.name(), (blockBuilder, op) -> {
                            if (invoke(lookup, op) instanceof OpHelper.Invoke invoke && invoke.targetMethodModelOrNull() instanceof CoreOp.FuncOp funcOp) {
                                var funcCall = copyLocation(funcOp,
                                        CoreOp.funcCall(funcOp.funcName(), funcOp.invokableType(), blockBuilder.context().getValues(invoke.op().operands()))
                                );
                                blockBuilder.context().mapValue(op.result(), blockBuilder.op(funcCall));
                            } else {
                                blockBuilder.op(op);
                            }
                            return blockBuilder;
                        });
                        moduleFuncOps.add(tf);
                    }
                }
        );
        return CoreOp.module(moduleFuncOps);
    }

}
