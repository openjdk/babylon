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
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

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

    // instance method with no accessors to any field in the function body
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

        Optional<Method> myFunction = Stream.of(HelloCodeReflection.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("myFunction"))
                .findFirst();

        Method m = myFunction.get();

        // Obtain the code model for the annotated method
        CoreOp.FuncOp codeModel = Op.ofMethod(m).get();

        // Print the code model of the annotated method
        String codeModelString = codeModel.toText();
        System.out.println(codeModelString);

        // Transform the code model to an SSA representation
        CoreOp.FuncOp ssaCodeModel = SSA.transform(codeModel);
        System.out.println("SSA Representation of a code model");
        System.out.println(ssaCodeModel.toText());

        // Evaluate a code model
        // because it is an instance method, the first parameter refers to `this`.
        var result = Interpreter.invoke(MethodHandles.lookup(), ssaCodeModel, obj, 10);
        System.out.println("Evaluate a code model");
        System.out.println(result);

        // Obtain parameters to the method
        Block.Parameter _this = ssaCodeModel.body().entryBlock().parameters().get(0);
        System.out.println("First parameter: " + _this);
        Block.Parameter _second = ssaCodeModel.body().entryBlock().parameters().get(1);
        System.out.println("Second parameter: " + _second);

        // Another way to print a code model, traversing each element until we reach the parent
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