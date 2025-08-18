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

import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.Inliner;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.OpWriter;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

/**
 * Simple example of how to use the code reflection API.
 *
 * <p>
 * This example is based on the {@link MathOptimizer} example to add inlining for the new replaced
 * methods. In this example, we focus on explaining in detail the inlining component.
 * </p>
 *
 * <p>
 * Optimizations:
 *     <ol>Replace Pow(x, y) when x == 2 to (1 << y), if only if the parameter y is an integer.</ol>
 *     <ol>Replace Pow(x, y) when y == 2 to (x * x).</ol>
 *     <ol>After a replacement has been done, we inline the new invoke into the main code model.</ol>
 * </p>
 *
 * <p>
 *     In a nutshell, we apply a second transform to perform the inlining. Note that the inlining could be done
 *     also in within the first transform.
 *     To be able to inline, we need to also annotate the new invoke nodes that were replaced during the first
 *     transform with the <code>@CodeReflection</code> annotation. In this way, we can build the code models for
 *     each of the methods and apply the inlining directly using the <code>Inliner.inline</code> from the code
 *     reflection API.
 * </p>
 *
 * <p>
 *     Babylon repository: {@see <a href="https://github.com/openjdk/babylon/tree/code-reflection">link</a>}
 * </p>
 *
 * <p>
 *     How to run?
 *     <code>
 *         java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.MathOptimizerWithInlining
 *     </code>
 * </p>:
 */
public class MathOptimizerWithInlining {

    // New functions are also annotated with code reflection
    @CodeReflection
    private static double myFunction(int value) {
        return Math.pow(2, value);
    }

    @CodeReflection
    private static int functionShift(int val) {
        return 1 << val;
    }

    @CodeReflection
    private static double functionMult(double x) {
        return x * x;
    }

    private static final MethodRef MY_SHIFT_FUNCTION = MethodRef.method(MathOptimizerWithInlining.class, "functionShift", int.class, int.class);

    private static final MethodRef MY_MULT_FUNCTION = MethodRef.method(MathOptimizerWithInlining.class, "functionMult", double.class, double.class);

    private static CoreOp.FuncOp buildCodeModelForMethod(Class<?> klass, String methodName) {
        Optional<Method> function = Stream.of(klass.getDeclaredMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();
        Method method = function.get();
        CoreOp.FuncOp funcOp = Op.ofMethod(method).get();
        return funcOp;
    }

    static void main(String[] args) {

        // Obtain the code model for the annotated method
        CoreOp.FuncOp codeModel = buildCodeModelForMethod(MathOptimizerWithInlining.class, "myFunction");
        System.out.println(codeModel.toText());

        enum FunctionToUse {
            SHIFT,
            MULT,
            GENERIC;
        }

        AtomicReference<FunctionToUse> replace = new AtomicReference<>(FunctionToUse.GENERIC);

        codeModel = codeModel.transform(CopyContext.create(), (blockBuilder, op) -> {
            // The idea here is to create a new JavaOp.invoke with the optimization and replace it.
            if (Objects.requireNonNull(op) instanceof JavaOp.InvokeOp invokeOp && whenIsMathPowFunction(invokeOp)) {
                List<Value> operands = blockBuilder.context().getValues(op.operands());

                // Analyse second operand of the Math.pow(x, y).
                // if the x == 2, and both are integers, then we can optimize the function using bitwise operations
                // pow(2, y) replace with (1 << y)
                Value operand = operands.getFirst();  // obtain the first parameter
                // inspect if the base (as in pow(base, exp) is value 2
                boolean canApplyBitShift = inspectParameterRecursive(operand, 2);
                if (canApplyBitShift) {
                    // We also need to inspect types. We can apply this optimization
                    // if the exp type is also an integer.
                    boolean isIntType = analyseType(operands.get(1), JavaType.INT);
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
                    canApplyMultiplication = inspectParameterRecursive(operands.get(1), 2);
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

                    replace.set(FunctionToUse.SHIFT);

                } else if (canApplyMultiplication) {
                    // Adapt the parameters to the new function. We only need the first
                    // parameter from the initial parameter list  - pow(x, 2) -
                    // Create a new invoke function with the optimised method
                    JavaOp.InvokeOp newInvoke = JavaOp.invoke(MY_MULT_FUNCTION, operands.get(0));
                    // Copy the location info to the new invoke
                    newInvoke.setLocation(invokeOp.location());

                    // Replace the invoke node with the new optimized invoke
                    Op.Result newResult = blockBuilder.op(newInvoke);
                    blockBuilder.context().mapValue(invokeOp.result(), newResult);
                    replace.set(FunctionToUse.MULT);

                } else {
                    // ignore the transformation
                    blockBuilder.op(op);
                }
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });

        System.out.println("Code Model after the first transform (replace with a new method): ");
        System.out.println(codeModel.toText());

        // Let's now apply a second transformation
        // We want to inline the functions. Note that we can apply this transformation if any of the new functions
        // have been replaced.
        System.out.println("Second transform: apply inlining for the new methods into the main code model");
        if (replace.get() != FunctionToUse.GENERIC) {

            // Build code model for the functions we want to inline.
            // Since we apply two replacements (depending on the values of the input code), we can apply two different
            // inline functions.
            CoreOp.FuncOp shiftCodeModel = buildCodeModelForMethod(MathOptimizerWithInlining.class, "functionShift");
            CoreOp.FuncOp multCodeModel = buildCodeModelForMethod(MathOptimizerWithInlining.class, "functionMult");

            // Apply inlining
            codeModel = codeModel.transform(codeModel.funcName(),
                                            (blockBuilder, op) -> {

                if (op instanceof JavaOp.InvokeOp invokeOp && isMethodWeWantToInline(invokeOp)) {
                    // Let's inline the function

                    // 1. Select the function we want to inline.
                    // Since we have two possible replacements, depending on the input code, we need to
                    // apply the corresponding replacement function
                    CoreOp.FuncOp codeModelToInline = isShiftFunction(invokeOp) ? shiftCodeModel : multCodeModel;

                    // 2. Apply the inlining
                    Inliner.inline(
                            blockBuilder,   // the current block builder
                            codeModelToInline,  // the method to inline which we obtained using code reflection too
                            blockBuilder.context().getValues(invokeOp.operands()),  // operands to this call. Since we already replace the function,
                            // we can use the same operands as the invoke call
                            (builder, val) -> blockBuilder.context().mapValue(invokeOp.result(), val)); // Propagate the new result
                } else {
                    // copy the op into the builder if it is not the invoke node we are looking for
                    blockBuilder.op(op);
                }

                // return new transformed block builder
                return blockBuilder;
            });

            System.out.println("After inlining: " + codeModel.toText());
        }

        codeModel = codeModel.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println("After Lowering: ");
        System.out.println(codeModel.toText());

        System.out.println("\nEvaluate with Interpreter.invoke");
        // The Interpreter Invoke should launch new exceptions
        var result = Interpreter.invoke(MethodHandles.lookup(), codeModel, 10);
        System.out.println(result);
    }

    // Utility methods

    // Analyze type methods: taken from example of String Concat Transformer to traverse the tree.
    static boolean analyseType(JavaOp.ConvOp convOp, JavaType typeToMatch) {
        return analyseType(convOp.operands().get(0), typeToMatch);
    }

    static boolean analyseType(Value v, JavaType typeToMatch) {
        // Maybe there is a utility already to do tree traversal
        if (v instanceof Op.Result r && r.op() instanceof JavaOp.ConvOp convOp) {
            // Node of tree, recursively traverse the operands
            return analyseType(convOp, typeToMatch);
        } else {
            // Leaf of tree: analyze type
            TypeElement type = v.type();
            return type.equals(typeToMatch);
        }
    }

    // Inspect a value for a parameter
    static boolean inspectParameterRecursive(JavaOp.ConvOp convOp, int valToMatch) {
        return inspectParameterRecursive(convOp.operands().get(0), valToMatch);
    }

    static boolean inspectParameterRecursive(Value v, int valToMatch) {
        if (v instanceof Op.Result r && r.op() instanceof JavaOp.ConvOp convOp) {
            return inspectParameterRecursive(convOp, valToMatch);
        } else {
            // Leaf of tree - we want to obtain the actual value of the parameter and check
            if (v instanceof CoreOp.Result r && r.op() instanceof CoreOp.ConstantOp constant) {
                return constant.value().equals(valToMatch);
            }
            return false;
        }
    }

    static final MethodRef JAVA_LANG_MATH_POW = MethodRef.method(Math.class, "pow", double.class, double.class, double.class);

    private static boolean whenIsMathPowFunction(JavaOp.InvokeOp invokeOp) {
        return invokeOp.invokeDescriptor().equals(JAVA_LANG_MATH_POW);
    }

    private static boolean isMethodWeWantToInline(JavaOp.InvokeOp invokeOp) {
        return (invokeOp.invokeDescriptor().toString().startsWith("oracle.code.samples.MathOptimizerWithInlining::functionShift")
                || invokeOp.invokeDescriptor().toString().startsWith("oracle.code.samples.MathOptimizerWithInlining::functionMult"));
    }

    private static boolean isShiftFunction(JavaOp.InvokeOp invokeOp) {
        return invokeOp.invokeDescriptor().toString().contains("functionShift");
    }
}