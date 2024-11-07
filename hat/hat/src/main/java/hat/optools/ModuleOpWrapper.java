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
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.MethodRef;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;

public class ModuleOpWrapper extends OpWrapper<CoreOp.ModuleOp> {
    ModuleOpWrapper(CoreOp.ModuleOp op) {
        super(op);
    }

    record MethodRefToEntryFuncOpCall(MethodRef methodRef, CoreOp.FuncOp funcOp) {
    }

    record Closure(Deque<MethodRefToEntryFuncOpCall> work, LinkedHashSet<MethodRef> funcsVisited,
                   List<CoreOp.FuncOp> moduleFuncOps) {
    }

    public static ModuleOpWrapper createTransitiveInvokeModule(MethodHandles.Lookup lookup,
                                                               Method entryPoint) {
        Optional<CoreOp.FuncOp> codeModel = entryPoint.getCodeModel();
        if (codeModel.isPresent()) {
            return OpWrapper.wrap(createTransitiveInvokeModule(lookup, MethodRef.method(entryPoint), codeModel.get()));
        } else {
            return OpWrapper.wrap(CoreOp.module(List.of()));
        }
    }
   /* static Method resolveToMethod(MethodHandles.Lookup lookup, MethodRef invokedMethodRef){
        Method invokedMethod = null;
        try {
            invokedMethod = invokedMethodRef.resolveToMethod(lookup);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        return invokedMethod;
    } */

    static CoreOp.ModuleOp createTransitiveInvokeModule(MethodHandles.Lookup lookup,
                                                        MethodRef methodRef, CoreOp.FuncOp entryFuncOp) {
        Closure closure = new Closure(new ArrayDeque<>(), new LinkedHashSet<>(), new ArrayList<>());
        closure.work.push(new MethodRefToEntryFuncOpCall(methodRef, entryFuncOp));
        while (!closure.work.isEmpty()) {
            MethodRefToEntryFuncOpCall methodRefToEntryFuncOpCall = closure.work.pop();
            if (closure.funcsVisited.add(methodRefToEntryFuncOpCall.methodRef)) {
                CoreOp.FuncOp tf = methodRefToEntryFuncOpCall.funcOp.transform(
                        methodRefToEntryFuncOpCall.methodRef.toString(), (blockBuilder, op) -> {
                            if (op instanceof CoreOp.InvokeOp invokeOp && OpWrapper.wrap(invokeOp) instanceof InvokeOpWrapper invokeOpWrapper) {
                                Method invokedMethod = invokeOpWrapper.method(lookup);
                                Optional<CoreOp.FuncOp> optionalInvokedFuncOp = invokedMethod.getCodeModel();
                                if (optionalInvokedFuncOp.isPresent() && OpWrapper.wrap(optionalInvokedFuncOp.get()) instanceof FuncOpWrapper funcOpWrapper) {
                                    MethodRefToEntryFuncOpCall call =
                                            new MethodRefToEntryFuncOpCall(invokeOpWrapper.methodRef(), funcOpWrapper.op());
                                    closure.work.push(call);
                                    CopyContext copyContext = blockBuilder.context();
                                    List<Value> operands = copyContext.getValues(invokeOp.operands());
                                    CoreOp.FuncCallOp replacementCall = CoreOp.funcCall(
                                            call.methodRef.toString(),
                                            call.funcOp.invokableType(),
                                            operands);
                                    Op.Result replacementResult = blockBuilder.op(replacementCall);
                                    copyContext.mapValue(invokeOp.result(), replacementResult);
                                    // System.out.println("replaced " + call);
                                } else {
                                    // System.out.println("We have no code model for " + invokeOpWrapper.methodRef());
                                    blockBuilder.op(invokeOp);
                                }
                            } else {
                                blockBuilder.op(op);
                            }
                            return blockBuilder;
                        });
                closure.moduleFuncOps.add(tf);
            }
        }

        return CoreOp.module(closure.moduleFuncOps);
    }
}
