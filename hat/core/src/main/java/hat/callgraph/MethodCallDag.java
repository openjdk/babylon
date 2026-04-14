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
import optkl.util.carriers.FuncOpCarrier;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Objects;
import java.util.stream.Stream;

public class MethodCallDag extends Dag<MethodCallDag.Call> {
    static abstract public class Call implements FuncOpCarrier {
        private final MethodRef methodRef;
        private final Method method;
        private CoreOp.FuncOp funcOp;
        Call(MethodRef methodRef, Method method, CoreOp.FuncOp funcOp){
            this.methodRef = methodRef;
            this.method = method;
            this.funcOp = funcOp;
        }

        @Override
        public int hashCode() {
            return Objects.hash(methodRef,method); // We exclude the funcOp!
        }

        @Override
        public boolean equals(Object o) {
            return (this == o)
                    || ( o instanceof Call that
                       && Objects.equals(methodRef, that.methodRef) && Objects.equals(method, that.method) // we exclude the funcOp
            );
        }

        @Override
        public CoreOp.FuncOp funcOp(){
            return this.funcOp;
        }
        @Override
        public void funcOp(CoreOp.FuncOp funcOp){
            this.funcOp = funcOp;
        }
        public Method method(){return method;}
        public MethodRef methodRef(){
            return this.methodRef;
        }
    }

    public static class EntryMethodCall extends Call {
        EntryMethodCall( Method method, CoreOp.FuncOp funcOp) {
            super(null, method, funcOp); // we dont have a methodRef for the root
        }
    }
    public static class OtherMethodCall extends Call {
        OtherMethodCall(OpHelper.Invoke invoke) {
            super(invoke.op().invokeReference(), invoke.resolveMethodOrThrow(),invoke.targetMethodModelOrNull()); // we dont have a methodRef for the root
        }
    }


    public final EntryMethodCall entryPoint;
    public final CoreOp.FuncOp inlined;

    // recursive
    void addEdge(Call from, OpHelper.Invoke invoke) {
        var to = new OtherMethodCall(invoke);
        add(from,to, _-> //only called if from->to is a new 'edge'
                    OpHelper.Invoke.stream(invoke.lookup(), to.funcOp()).filter((inv) -> inv.targetMethodModelOrNull() != null).forEach(i ->
                            addEdge(to, i) // recurses here
                    )
        );
    }

    MethodCallDag(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp entry, CoreOp.FuncOp inlined) {
        this.inlined = inlined;
        this.entryPoint = new EntryMethodCall(method,entry);
        OpHelper.Invoke.stream(lookup, entry)
                .filter((inv)-> inv.targetMethodModelOrNull() != null)
                .forEach(i ->
                        addEdge(entryPoint, i)
                );
        closeRanks();
    }
}
