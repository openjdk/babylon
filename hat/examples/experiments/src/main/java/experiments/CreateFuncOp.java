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

package experiments;


import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;

import static optkl.OpTkl.transform;


import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;

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
     * java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DynamicFunctionBuild
     * </code>
     * </p>
     */
    public class CreateFuncOp {
        @Reflect
        public  static void main(String[] args) {
            MethodRef MathSqrt = MethodRef.method(Math.class, "sqrt", double.class, double.class);
            CoreOp.FuncOp rsqrtFuncOp = CoreOp.func("rsqrt", CoreType.functionType(JavaType.DOUBLE, JavaType.DOUBLE))
                    .body(builder -> {
                                // double rsqrt(double arg){
                                //      return 1 / Math.sqrt(qrg)
                                //}

                        var block = builder.block(JavaType.INT);
                        var c=block.context();


                      //  builder.op(block);
                        var arg = builder.parameters().getFirst();
                        // Add an invoke of Math.sqrt
                        Op.Result invokeResult = builder.op(
                                JavaOp.invoke(InvokeKind.STATIC, false, JavaType.DOUBLE, MathSqrt, arg)// we pass the first param of our rsqrt to Math.sqrt
                        );

                        // Add divisionOp (1f,invokeResult)
                        Op.Result divResult = builder.op(JavaOp.div(builder.op(CoreOp.constant(JavaType.DOUBLE, 1.0)), invokeResult));

                        builder.op(CoreOp.return_(divResult));
                    });

            // Print the code model for the function we have just created
            System.out.println(rsqrtFuncOp.toText());

            // Run in the Java Bytecode interpreter
            MethodHandle rsqrt = BytecodeGenerator.generate(MethodHandles.lookup(), rsqrtFuncOp);
            try {
                System.out.println("Evaluation in the Java Bytecode Interpreter: 1/sqrt(100) = " +  rsqrt.invoke(100));
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

