/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package oracle.code.samples;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Example of inlining based on one of the example from the code-reflection unit-tests.
 * This example illustrates how to embed a value into a function and propagate that
 * change in the code model.
 */
public class InlineExample {

    // We are going to inline this function and specialize one of the parameters
    // (e.g., parameter b) to a constant value.
    // The new function will contain two parameters to perform a * CONSTANT + c
    @CodeReflection
    private static float fma(float a, float b, float c) {
        return a * b + c;
    }

    private static CoreOp.FuncOp buildCodeModelForMethod(Class<?> klass, String methodName) {
        Optional<Method> function = Stream.of(klass.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();
        Method method = function.get();
        CoreOp.FuncOp funcOp = Op.ofMethod(method).get();
        return funcOp;
    }

    public static void main(String[] args) {

        // 1. Build a new function
        CoreOp.FuncOp fmaCodeModel = buildCodeModelForMethod(InlineExample.class, "fma");

        CoreOp.FuncOp f = CoreOp.func("myFunction", CoreType.functionType(JavaType.FLOAT, // return type
                                                                                    JavaType.FLOAT, // param 1
                                                                                    JavaType.FLOAT  // param 2 (the new function has 2 params
                                        ))
                .body(blockBuilder -> {
                    // Get parameters
                    Block.Parameter parameter1 = blockBuilder.parameters().get(0);
                    Block.Parameter parameter2 = blockBuilder.parameters().get(1);

                    // Build a new op with a pre-defined constant. We will place this constant
                    // as one of the parameters of the function and then inline with the new values.
                    Op.Result myConstantValue = blockBuilder.op(CoreOp.ConstantOp.constant(JavaType.FLOAT, 50.f));

                    // Inline the function with the new values
                    Inliner.inline(blockBuilder,
                            fmaCodeModel,      // inline the fmaCodeModel
                            List.of(parameter1, myConstantValue, parameter2),  // apply the 3 parameters to the function to inline
                            Inliner.INLINE_RETURN);
                });

        System.out.println(f.toText());

        var result = Interpreter.invoke(MethodHandles.lookup(), f, 10.f, 20.f);
        // 10 * 50 + 20 => 520.0f
        System.out.println("Result: " + result);
    }
}
