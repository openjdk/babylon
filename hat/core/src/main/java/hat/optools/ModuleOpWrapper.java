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
package hat.optools;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

import hat.callgraph.CallGraph;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import java.util.*;

public class ModuleOpWrapper extends OpWrapper<CoreOp.ModuleOp> {
    public ModuleOpWrapper(MethodHandles.Lookup lookup, CoreOp.ModuleOp op) {
        super(lookup,op);
    }

    public SequencedMap<String, CoreOp.FuncOp> functionTable() {
        return op().functionTable();
    }

//    public static ModuleOpWrapper createTransitiveInvokeModule(MethodHandles.Lookup lookup,
//                                                               CallGraph.ResolvedMethodCall resolvedMethodCall) {
//        Optional<CoreOp.FuncOp> codeModel = Op.ofMethod(entryPoint);
//        if (codeModel.isPresent()) {
//            return OpWrapper.wrap(lookup, createTransitiveInvokeModule(lookup, resolvedMethodCall.targetMethodRef, resolvedMethodCall.funcOpWrapper()));
//        } else {
//            return OpWrapper.wrap(lookup, CoreOp.module(List.of()));
//        }
//    }

   /*  Method resolveToMethod(MethodHandles.Lookup lookup, MethodRef invokedMethodRef){
        Method invokedMethod = null;
        try {
            invokedMethod = invokedMethodRef.resolveToMethod(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        return invokedMethod;
    } */

    public static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup l,
                                                        FuncOpWrapper entry, CallGraph<?> callGraph) {
        LinkedHashSet<MethodRef> funcsVisited = new LinkedHashSet<>();
        List<CoreOp.FuncOp> funcs = new ArrayList<>();
        record RefAndFunc(MethodRef r, FuncOpWrapper f) {}

        Deque<RefAndFunc> work = new ArrayDeque<>();

        entry.selectCalls((invokeOpWrapper) -> {
            MethodRef methodRef = invokeOpWrapper.methodRef();
            Method method = null;
            Class<?> javaRefTypeClass = invokeOpWrapper.javaRefClass().orElseThrow();
            try {
                method = methodRef.resolveToMethod(l, invokeOpWrapper.op().invokeKind());
            } catch (ReflectiveOperationException _) {}
            Optional<CoreOp.FuncOp> f = Op.ofMethod(method);
            if (f.isPresent() && !callGraph.filterCalls(f.get(), invokeOpWrapper, method, methodRef, javaRefTypeClass)) {
                work.push(new RefAndFunc(methodRef, new FuncOpWrapper(l, f.get())));
            }
        });

        while (!work.isEmpty()) {
            RefAndFunc rf = work.pop();
            if (!funcsVisited.add(rf.r)) {
                continue;
            }

            CoreOp.FuncOp tf = rf.f.transform(rf.r.name(), (blockBuilder, op) -> {
                if (op instanceof JavaOp.InvokeOp iop) {
                    InvokeOpWrapper iopWrapper = OpWrapper.wrap(entry.lookup, iop);
                    MethodRef methodRef = iopWrapper.methodRef();
                    Method invokeOpCalledMethod = null;
                    try {
                        invokeOpCalledMethod = methodRef.resolveToMethod(l, iop.invokeKind());
                    } catch (ReflectiveOperationException _) {}
                    if (invokeOpCalledMethod instanceof Method m) {
                        Optional<CoreOp.FuncOp> f = Op.ofMethod(m);
                        if (f.isPresent()) {
                            RefAndFunc call = new RefAndFunc(methodRef, new FuncOpWrapper(l, f.get()));
                            work.push(call);

                            Op.Result result = blockBuilder.op(CoreOp.funcCall(
                                    call.r.name(),
                                    call.f.op().invokableType(),
                                    blockBuilder.context().getValues(iop.operands())));
                            blockBuilder.context().mapValue(op.result(), result);
                            return blockBuilder;
                        }
                    }
                }
                blockBuilder.op(op);
                return blockBuilder;
            });
            funcs.addFirst(tf);
        }

        return CoreOp.module(funcs);
    }
}
