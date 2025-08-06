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
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Location;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Simple example of how to use the code reflection API.
 *
 * <p>
 * This example replaces a math function Math.pow with an optimized function using code transforms
 * from the code-reflection API. The optimized function can be applied only under certain conditions.
 * </p>
 *
 * <p>
 * Optimizations:
 * 1) Replace Pow(x, y) when x == 1 to 1 << y, if only if the parameter y is an integer.
 * 2) Replace Pow(x, y) when y == 2 to x * x.
 * </p>
 *
 * <p>
 *     Babylon repository: {@see <a href="https://github.com/openjdk/babylon/tree/code-reflection">link</a>}
 * </p>
 *
 * <p>
 *     How to run?
 *     <code>
 *         java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizer
 *     </code>
 * </p>:
 */
public class MathOptimizer {

    @CodeReflection
    private static double myFunction(int value) {
        return Math.pow(2, value);
    }

    // if pow(2, x), then substitute for this function
    // We could apply this function if, at runtime, user pass int values to the pow function
    // Thus, we narrow the result type from 8 bytes (double) to 4 bytes (INT).
    private static int functionShift(int val) {
        return 1 << val;
    }

    // if pow(x, 2) then substitute for this function
    private static double functionMult(double x) {
        return x * x;
    }

    private static final MethodRef MY_SHIFT_FUNCTION = MethodRef.method(MathOptimizer.class, "functionShift", int.class, int.class);

    private static final MethodRef MY_MULT_FUNCTION = MethodRef.method(MathOptimizer.class, "functionMult", double.class, double.class);

    // Analyze type methods: taken from example of String Concat Transformer to traverse the tree.
    static boolean analyseType(Block.Builder block, JavaOp.ConvOp cz, JavaType typeToMatch) {
        return analyseType(block, cz.operands().get(0), typeToMatch);
    }

    static boolean analyseType(Block.Builder block, Value v, JavaType typeToMatch) {
        // Maybe there is an utility already to do tree traversal
        if (v instanceof Op.Result r && r.op() instanceof JavaOp.ConvOp conv) {
            // Node of tree, recursively traverse the operands
            return analyseType(block, conv, typeToMatch);
        } else {
            // Leaf of tree: analyze type
            TypeElement type = v.type();
            return type.equals(typeToMatch);
        }
    }

    static void main(String[] args) {

        Optional<Method> myFunction = Stream.of(MathOptimizer.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("myFunction"))
                .findFirst();

        Method myMathMethod = myFunction.get();

        // Obtain the code model for the annotated method
        CoreOp.FuncOp codeModel = Op.ofMethod(myMathMethod).get();
        System.out.println(codeModel.toText());

        System.out.println("\nLet's transform the code");
        codeModel = codeModel.transform(CopyContext.create(), (blockBuilder, op) -> {
            switch (op) {
                case JavaOp.InvokeOp invokeOp when whenIsMathPowFunction(invokeOp) -> {
                    // The idea here is to create a new JavaOp.invoke with the optimization and replace it.
                    List<Value> operands = blockBuilder.context().getValues(op.operands());

                    // Analyse second operand of the Math.pow(x, y).
                    // if the x == 2, and both are integers, then we can optimize the function using bitwise operations
                    // pow(2, y) replace with (1 << y)
                    Value operand = operands.getFirst();  // obtain the first parameter
                    // inspect if the base (as in pow(base, exp) is value 2
                    boolean canApplyBitShift = inspectParameter(operand, 2);
                    if (canApplyBitShift) {
                        // We also need to inspect types. We can apply this optimization
                        // if the exp type is also an integer.
                        boolean isIntType = analyseType(blockBuilder, operands.get(1), JavaType.INT);
                        if (!isIntType) {
                            canApplyBitShift = false;
                        }
                    }

                    // If the conditions to apply the first optimization failed, we try the second optimization
                    // if types are not int, and base is not 2.
                    // pow(x, 2) => replace with x * x
                    boolean canApplyMultiplication = false;
                    if (!canApplyBitShift) {
                        // inspect if exp (as in pow(base, exp) is value 2
                        canApplyMultiplication = inspectParameter(operands.get(1), 2);
                    }

                    if (canApplyBitShift) {
                        // Narrow type from DOUBLE to INT for the input parameter of the new function.
                        Op.Result op2 = blockBuilder.op(JavaOp.conv(JavaType.INT, operands.get(1)));
                        List<Value> newOperandList = new ArrayList<>();
                        newOperandList.add(op2);

                        // Create a new invoke with the optimised method
                        JavaOp.InvokeOp newInvoke = JavaOp.invoke(MY_SHIFT_FUNCTION, newOperandList);
                        // Copy the original location info to the new invoke
                        newInvoke.setLocation(invokeOp.location());

                        // Replace the invoke node with the new optimized invoke
                        Op.Result newResult = blockBuilder.op(newInvoke);
                        blockBuilder.context().mapValue(invokeOp.result(), newResult);

                    } else if (canApplyMultiplication) {
                        // Adapt the parameters to the new function. We only need the first
                        // parameter from the initial parameter list  - pow(x, 2) -
                        // Thus, we copy parameter x into a new list and pass it to the new
                        // invoke function.
                        List<Value> newOperandList2 = new ArrayList<>();
                        newOperandList2.add(operands.get(0));

                        // Create a new invoke function with the optimised method
                        JavaOp.InvokeOp newInvoke = JavaOp.invoke(MY_MULT_FUNCTION, newOperandList2);
                        // Copy the location info to the new invoke
                        newInvoke.setLocation(invokeOp.location());

                        // Replace the invoke node with the new optimized invoke
                        Op.Result newResult = blockBuilder.op(newInvoke);
                        blockBuilder.context().mapValue(invokeOp.result(), newResult);

                    } else {
                        // ignore the transformation
                        blockBuilder.op(op);
                    }
                }
                default -> blockBuilder.op(op);
            }
            return blockBuilder;
        });

        System.out.println("AFTER TRANSFORM: ");
        System.out.println(codeModel.toText());
        codeModel = codeModel.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println("After Lowering: ");
        System.out.println(codeModel.toText());

        System.out.println("\nEvaluate");
        // The Interpreter Invoke should launch new exceptions
        var result = Interpreter.invoke(MethodHandles.lookup(), codeModel, 10);
        System.out.println(result);

        // Select invocation calls and display the lines
        System.out.println("\nPlaying with Traverse");
        codeModel.traverse(null, (map, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                System.out.println("Function Name: " + invokeOp.invokeDescriptor().name());

                // Maybe Location should throw a new exception instead of the NPE,
                // since it is possible we don't have a location after a transformation has been done.
                Location location = invokeOp.location();
                if (location != null) {
                    int line = location.line();
                    System.out.println("Line " + line);
                    System.out.println("Class: " + invokeOp.getClass());
                    // Detect Math::pow
                    boolean contains = invokeOp.invokeDescriptor().equals(JAVA_LANG_MATH_POW);
                    if (contains) {
                        System.out.println("Method: " + invokeOp.invokeDescriptor().name());
                    }
                } else {
                    System.out.println("[WARNING] Location is null");
                }
            }
            return map;
        });
    }

    // It could be a better way of doing the following.
    // Goal: obtain and check the value of the function parameters.
    private static boolean inspectParameter(Value operand, final int value) {
        final Boolean[] isMultipliedByTwo = new Boolean[] { false };
        if (operand instanceof Op.Result res) {
            if (res.op() instanceof JavaOp.ConvOp convOp) {
                convOp.operands().forEach(v -> {
                    if (v instanceof Op.Result res2) {
                        if (res2.op() instanceof CoreOp.ConstantOp constantOp) {
                            if (constantOp.value() instanceof Integer parameter) {
                                if (parameter.intValue() == value) {
                                    // Transformation is valid
                                    isMultipliedByTwo[0] = true;
                                }
                            }
                        }
                    }
                });
            }
        }
        return isMultipliedByTwo[0];
    }

    static final MethodRef JAVA_LANG_MATH_POW = MethodRef.method(Math.class, "pow", double.class, double.class, double.class);

    private static boolean whenIsMathPowFunction(JavaOp.InvokeOp invokeOp) {
        return invokeOp.invokeDescriptor().equals(JAVA_LANG_MATH_POW);
    }
}