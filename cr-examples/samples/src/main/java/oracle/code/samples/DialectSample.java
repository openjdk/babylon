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

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Simple example demonstrating how to create new nodes to build a new dialect.
 * <p>
 * This example customizes two operations: addition and invoke. Specifically, it
 * replaces the standard {@link JavaOp.AddOp} node with a custom node
 * {@link MyAdd}, and the {@link JavaOp.InvokeOp} node with {@link MyInvoke}.
 * </p>
 * <p>
 * After constructing the new code-reflection tree, you can traverse it and
 * perform new operations with the custom nodes. One potential use case is to
 * assign specific semantics to these new nodes and handle them accordingly.
 * </p>
 *
 * <p>
 *     How to run from the terminal?
 *     <code>
 *         $ java --enable-preview -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DialectSample
 *     </code>
 * </p>
 */
public class DialectSample {

    // Part A: Create a new Op for specializing additions
    @CodeReflection
    public static int mySum(int a, int b) {
        return a + b;
    }

    // Custom Node inherits from Op
    private static class MyAdd extends Op {

        private TypeElement typeElement;

        MyAdd(String name, List<Value> operands, TypeElement typeElement) {
            super(name, operands);
            this.typeElement = typeElement;
        }

        MyAdd(Op that, CopyContext cc) {
            super(that, cc);
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            return new MyAdd(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeElement;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", this.typeElement);
        }
    }

    /**
     * Utility method to print the code tree in levels. We should see the new instances
     * of our custom nodes.
     * @param dialectFuncOp
     */
    private static void printCodeTree(CoreOp.FuncOp dialectFuncOp) {
        dialectFuncOp.traverse(null, (acc, codeElement) -> {
            int depth = 0;
            CodeElement<?, ?> parent = codeElement;
            while ((parent = parent.parent()) != null) {
                depth++;
            }
            System.out.println(" ".repeat(depth) + codeElement.getClass());
            return acc;
        });
    }

    /**
     * Builds the code model for the {@link DialectSample#mySum(int, int)} method and replaces
     * the {@link JavaOp.AddOp} op with our custom op.
     */
    private static void customAdd() {

        // 1. Obtain the method instance for the reflected code.
        Optional<Method> myFunction = Stream.of(DialectSample.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("mySum"))
                .findFirst();
        Method m = myFunction.get();

        // 2. Obtain the code model for the function
        CoreOp.FuncOp functionModel = Op.ofMethod(m).get();

        // 3. Print the original code model
        String codeModelString = functionModel.toText();
        System.out.println(codeModelString);

        // 3. Transform the code model to include our custom Op
        CoreOp.FuncOp dialectModel = functionModel.transform((builder, op) -> {
            CopyContext context = builder.context();
            // 4 Find the JavaOp.Op node
            if (op instanceof JavaOp.AddOp addOp) {
                // 5. Obtain the operands for the node
                List<Value> inputOperands = addOp.operands();
                List<Value> outputOperands = context.getValues(inputOperands);

                // 6. Create a new Op with the new operation
                MyAdd myAddOp = new MyAdd("myAdd", outputOperands, JavaType.INT);

                // 7. Attach the new Op to the builder
                Op.Result result = builder.op(myAddOp);

                // 8. Propagate the location of the new op
                myAddOp.setLocation(addOp.location());

                // 9. Map the values from input -> output for the new Op
                context.mapValue(addOp.result(), result);
            } else {
                builder.op(op);
            }
            return builder;
        });

        // 10. Print the transformed code model
        System.out.println("Model with new OpNodes for Dialect: ");
        System.out.println(dialectModel.toText());

        System.out.println("Print Code Tree: ");
        printCodeTree(dialectModel);

        // Currently, we can't interpreter a code model with dialect ops
        //var result = Interpreter.invoke(MethodHandles.lookup(), dialectModel,  10, 20);
        //System.out.println("Result: ");
    }


    // Part B: Create a new Op for specializing the invoke Op.
    private static int myIntrinsic(int a, int b) {
        return a + b;
    }

    @CodeReflection
    public static int myFunction(int a, int b) {
        return myIntrinsic(a, b);
    }

    public static class MyInvoke extends Op { // externalized

        private final TypeElement typeDescriptor;

        MyInvoke(String opName, TypeElement typeDescriptor, List<Value> operands) {
            super(opName, operands);
            this.typeDescriptor = typeDescriptor;
        }

        MyInvoke(Op that, CopyContext cc) {
            super(that, cc);
            this.typeDescriptor = that.resultType();
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            return new MyInvoke(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeDescriptor;
        }

        @Override
        public List<Value> operands() {
            return super.operands();
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", this.typeDescriptor);
        }
    }

    private static void customInvoke() {
        Optional<Method> myFunction = Stream.of(DialectSample.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("myFunction"))
                .findFirst();

        Method m = myFunction.get();

        CoreOp.FuncOp functionModel = Op.ofMethod(m).get();

        String codeModelString = functionModel.toText();
        System.out.println(codeModelString);

        CoreOp.FuncOp dialectModel = functionModel.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(inputOperands);

                // Create new node
                Op.Result inputResult = invokeOp.result();
                MyInvoke myCustomFunction = new MyInvoke("myCustomFunction", JavaType.INT, outputOperands);
                Op.Result outputResult = blockBuilder.op(myCustomFunction);

                // Preserve the location from the original invoke
                myCustomFunction.setLocation(invokeOp.location());

                // Map input-> new output
                context.mapValue(inputResult, outputResult);
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        });

        System.out.println("Model with new OpNodes for Dialect: ");
        System.out.println(dialectModel.toText());

        CoreOp.FuncOp ssaDialect = SSA.transform(dialectModel);
        System.out.println("Model with new OpNodes for SsaDialect: ");
        System.out.println(ssaDialect.toText());

        System.out.println("Printing the code tree. Is the new Op present? ");
        printCodeTree(ssaDialect);

        // Currently, we can't interpreter a code model with dialect ops
        //var result = Interpreter.invoke(MethodHandles.lookup(), ssaDialect,  10, 20);
        //System.out.println("Result: " + result);
    }

    static void main() {
        System.out.println("Create new Integer Add Op: ");
        customAdd();

       System.out.println("Create new Invoke Op: ");
       customInvoke();
    }
}
