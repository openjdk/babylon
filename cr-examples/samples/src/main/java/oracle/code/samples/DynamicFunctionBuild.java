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
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;

/**
 * Demonstrates how to dynamically build a new function using the code reflection API.
 * <p>
 * This example creates an <code>rsqrt</code> function, which computes the inverse of a square root.
 * The function takes one argument of type <code>double</code> and returns a <code>double</code>.
 * The implementation uses {@link Math#sqrt(double)} for the square root calculation.
 * </p>
 *
 * <p>
 * In this example, you will learn how to:
 * <ol>
 *   <li>Create a function dynamically</li>
 *   <li>Append new Op nodes in the builder</li>
 *   <li>Compose operations in the code tree</li>
 *   <li>Create nodes to call static methods</li>
 *   <li>Evaluate the composed method in the interpreter</li>
 * </ol>
 * </p>
 *
 * <p>
 * After building the code model for the function, it will be executed both in the code reflection interpreter and in the bytecode interpreter.
 * </p>
 *
 * <p>
 * <b>How to run:</b><br>
 * <code>
 * java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DynamicFunctionBuild
 * </code>
 * </p>
 */
public class DynamicFunctionBuild {

    static void main(String[] args) {

        CoreOp.FuncOp myFunction = CoreOp.func("rsqrt",
                // Define the signature of our new method
                // The function will be called `rsqrt`:
                // rsqrt(double):double
                CoreType.functionType(JavaType.DOUBLE, JavaType.DOUBLE))
                .body(builder -> {

                    // Obtain the first parameter
                    Block.Parameter inputParameter = builder.parameters().get(0);

                    // Create an op to represent the constant 1
                    CoreOp.ConstantOp constant1 = CoreOp.constant(JavaType.DOUBLE, 1.0);
                    // Add the Op into the builder
                    Op.Result constantResult = builder.op(constant1);

                    // Create a MethodRef to point to Math.sqrt
                    MethodRef sqrtMethodRef = MethodRef.method(Math.class, "sqrt", double.class, double.class);

                    // Prepare the list of arguments for the Math.sqrt invoke
                    List<Value> arguments = new ArrayList<>();
                    arguments.add(inputParameter);

                    // Create an invoke Op
                    JavaOp.InvokeOp invokeMathOp = JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, sqrtMethodRef, arguments);

                    // Add the invoke op into the builder
                    Op.Result invokeResult = builder.op(invokeMathOp);

                    // Create a division node and add it to the code builder
                    JavaOp.BinaryOp divOp = JavaOp.div(constantResult, invokeResult);
                    Op.Result divResult = builder.op(divOp);

                    // Finally, add a return and add it to the code builder
                    CoreOp.ReturnOp retOp = CoreOp.return_(divResult);
                    builder.op(retOp);
                });

        // Print the code model for the function we have just created
        System.out.println(myFunction.toText());

        // Run the new function in the Code Reflection's interpreter
        Object result = Interpreter.invoke(MethodHandles.lookup(), myFunction, 100);
        System.out.println("Evaluation in the Code Reflection's Interpreter: 1/sqrt(100) = " + result);

        // Run in the Java Bytecode interpreter
        MethodHandle generate = BytecodeGenerator.generate(MethodHandles.lookup(), myFunction);
        try {
            Object resultFromBCInterpreter = generate.invoke(100);
            System.out.println("Evaluation in the Java Bytecode Interpreter: 1/sqrt(100) = " + resultFromBCInterpreter);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}
