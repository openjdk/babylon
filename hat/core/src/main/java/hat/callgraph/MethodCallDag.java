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
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.OpHelper;
import optkl.util.Dag;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public class MethodCallDag extends Dag<MethodCallDag.MethodInfo> {
    static public class MethodInfo{
        public  CoreOp.FuncOp funcOp;

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

    MethodCallDag(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp funcOp, CoreOp.FuncOp inlined) {
        super(lookup);
        this.inlined = inlined;
        this.entryPoint = MethodInfo.of(funcOp, null, method);// we dont have a methodRef for the root
        nodeSet.add(this.entryPoint);
        fromToNodes.put(this.entryPoint,new HashSet<>());
    }

    // recursive
    void addEdge(MethodInfo methodInfo, OpHelper.Invoke invoke) {
        computeIfAbsent(methodInfo,MethodInfo.of(invoke.targetMethodModelOrNull(), invoke.op().invokeReference(), invoke.resolveMethodOrThrow()), n->
            OpHelper.Invoke.stream(invoke.lookup(), n.funcOp)
                    .filter(i -> i.targetMethodModelOrNull() != null)
                    .forEach(i -> addEdge(n, i))
        );
    }

    static public MethodCallDag of(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp entry, CoreOp.FuncOp inlined) {
        var dag = new MethodCallDag(lookup, method, entry, inlined);
        OpHelper.Invoke.stream(lookup, entry)
                .filter(invoke -> invoke.targetMethodModelOrNull() != null)
                .forEach(i -> dag.addEdge(dag.entryPoint, i));
        dag.closeRanks();
        return dag;
    }



    public CoreOp.ModuleOp toModuleOp() {
        List<CoreOp.FuncOp> moduleFuncOps = new ArrayList<>();
        rankOrdered.forEach(methodInfo -> {
                    if (methodInfo.methodRef != null) {
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
