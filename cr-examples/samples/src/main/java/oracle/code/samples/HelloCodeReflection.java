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
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Simple example of how to use the code reflection API, showing how to
 * build a code model, lower it to an SSA representation and run it
 * in an interpreter.
 *
 * <p>
 *     Babylon repository: {@see <a href="https://github.com/openjdk/babylon/tree/code-reflection">link</a>}
 * </p>
 *
 * <p>
 *     How to run?
 *     <code>
 *          java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.HelloCodeReflection
 *     </code>
 * </p>
 */
public class HelloCodeReflection {

    private int value;

    private HelloCodeReflection(int value) {
        this.value = value;
    }

    // Code Reflection methods are annotated with @CodeReflection.
    // When javac sees an annotated method with @CodeReflection, it stores
    // metadata in the ClassFile to be able to query and manipulate
    // code models at runtime.
    @CodeReflection
    private double myFunction(int value) {
        return Math.pow(value, 2);
    }

    // Example of an instance method using a field
    @CodeReflection
    private double myFunctionWithFieldAccess() {
        return Math.pow(this.value, 2);
    }

    static void main(String[] args) {

        System.out.println("Hello Code Reflection!");

        HelloCodeReflection obj = new HelloCodeReflection(5);

        // 1. Build the code model of the annotated method
        // 1.1 We use reflection to obtain the list of methods declared within a function and f
        //     filter for the one we want to build the code model.
        Optional<Method> myFunction = Stream.of(HelloCodeReflection.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("myFunction"))
                .findFirst();

        // 1.2 Obtain the method object
        Method m = myFunction.get();

        // 1.3 Obtain the code model for the annotated method
        CoreOp.FuncOp codeModel = Op.ofMethod(m).get();

        // 2. Print the code model of the annotated method
        String codeModelString = codeModel.toText();
        System.out.println(codeModelString);

        // 3. Transform the code model to an SSA representation
        CoreOp.FuncOp ssaCodeModel = SSA.transform(codeModel);
        System.out.println("SSA Representation of a code model");
        System.out.println(ssaCodeModel.toText());

        // 4. Evaluate a code model
        // Note: because it is an instance method, the first parameter refers to `this`.
        var result = Interpreter.invoke(MethodHandles.lookup(), ssaCodeModel, obj, 10);
        System.out.println("Evaluate a code model");
        System.out.println(result);

        // 5. We can obtain parameters to the method
        Block.Parameter _this = ssaCodeModel.body().entryBlock().parameters().get(0);
        System.out.println("First parameter: " + _this);
        Block.Parameter _second = ssaCodeModel.body().entryBlock().parameters().get(1);
        System.out.println("Second parameter: " + _second);

        // 6. Generate bytecodes from the lowered code model.
        // Note: The BytecodeGenerator.generate method receives a code model, and returns
        // a method handle to be able to invoke the code.
        MethodHandle methodHandle = BytecodeGenerator.generate(MethodHandles.lookup(), ssaCodeModel);
        try {
            var res = methodHandle.invoke(obj, 10);
            System.out.println("Result from bytecode generation: " + res);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        // 7. AST Printer
        // Just for illustration purposes, this is another way to print a code model,
        // traversing each element until we reach the parent
        codeModel.traverse(null, (acc, codeElement) -> {
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) {
                depth++;
            }
            System.out.println(" ".repeat(depth) + codeElement.getClass());
            return acc;
        });
    }
}