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
import java.util.Objects;
import java.util.stream.Stream;

public class MethodCallDag extends Dag<MethodCallDag.MethodCall> {


    static public class MethodCall {
        public enum MethodType{Func,Entry};
        public final  MethodType methodType;
        public  CoreOp.FuncOp funcOp;
        public final MethodRef methodRef;
        public final Method method;
        MethodCall(MethodType methodType, CoreOp.FuncOp funcOp, MethodRef methodRef, Method method){
            this.methodType = methodType;
            this.funcOp = funcOp;
            this.methodRef = methodRef;
            this.method = method;
        }

        @Override
        public int hashCode() {
            return Objects.hash(methodType,methodRef,method);
        }

        @Override
        public boolean equals(Object o) {
            return (this == o)
                    || ( o instanceof MethodCall that
                       && Objects.equals(methodType,that.methodType)&&Objects.equals(methodRef, that.methodRef) && Objects.equals(method, that.method)
            );
        }
    }

    public final MethodCall entryPoint;
    public final CoreOp.FuncOp inlined;

    // recursive
    void addEdge(MethodCall methodCall, OpHelper.Invoke invoke) {
        add(methodCall, new MethodCall(MethodCall.MethodType.Func,invoke.targetMethodModelOrNull(), invoke.op().invokeReference(), invoke.resolveMethodOrThrow()), n->
                OpHelper.Invoke.stream(invoke.lookup(), n.funcOp).filter((inv)-> inv.targetMethodModelOrNull() != null).forEach(i ->
                        addEdge(n, i) // recurse
                )
        );
    }

    MethodCallDag(MethodHandles.Lookup lookup, Method method, CoreOp.FuncOp entry, CoreOp.FuncOp inlined) {
        this.inlined = inlined;
        this.entryPoint = new MethodCall(MethodCall.MethodType.Entry,entry, null, method);// we dont have a methodRef for the root
        OpHelper.Invoke.stream(lookup, entry).filter((inv)-> inv.targetMethodModelOrNull() != null).forEach(i ->
                addEdge(entryPoint, i)
        );
        closeRanks();
    }

    public Stream<MethodCall> rankOrderedFunctions() {
        return rankOrdered.stream().filter(f->f.methodType.equals(MethodCallDag.MethodCall.MethodType.Func));
    }
}
