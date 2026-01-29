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

import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

public class ComputeEntrypoint extends ComputeCallGraph.ComputeReachableResolvedMethodCall implements Entrypoint {
    private final MethodHandles.Lookup lookup;
    @Override
    public MethodHandles.Lookup lookup() {
        return lookup;
    }
    private CoreOp.FuncOp lowered;
    private MethodHandle bytecodeGeneratedMethodHandle;

    public ComputeEntrypoint(MethodHandles.Lookup lookup,CallGraph<ComputeEntrypoint> callGraph, Method method, CoreOp.FuncOp funcOp) {
        super(callGraph, null, method, funcOp);
        this.lookup = lookup;
    }



    public CoreOp.FuncOp lazyLower(){
        if (lowered == null) {
            lowered =funcOp().transform(CodeTransformer.LOWERING_TRANSFORMER);
        }
        return lowered;
    }

    public void interpretWithArgs( Object[] args) {
        Interpreter.invoke(lookup, lazyLower(), args);
    }

    public void invokeWithArgs(Object[] args) {
        try {
            if (bytecodeGeneratedMethodHandle == null) {
                bytecodeGeneratedMethodHandle = BytecodeGenerator.generate(lookup,lazyLower());
            }
            bytecodeGeneratedMethodHandle.invokeWithArguments(args);
        }catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

}
