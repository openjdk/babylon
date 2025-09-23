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

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Simple example demonstrating how to create new nodes to build a new dialect with
 * code reflection.
 * <p>
 * This example customizes an operation: it inspects the code model to find
 * Fused-Multiple-Add operations (FMA). This means and Add(Mult(x, y), z).
 * </p>
 *
 * <p>
 * If the detection is successful, we replace the Add and Mult ops with a new
 * Op called {@FMA}.
 *
 * <p>
 *     How to run from the terminal?
 *     <code>
 *         $ java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DialectFMAOp
 *     </code>
 * </p>
 */
public class DialectFMAOp {

    @CodeReflection
    public static float compute(float a, float b, float c) {
        return a * b + c;
    }

    // Custom Node inherits from Op
    private static class FMA extends Op {

        private final TypeElement typeElement;
        private static final String NAME = "fma";

        FMA(List<Value> operands, TypeElement typeElement) {
            super(NAME, operands);
            this.typeElement = typeElement;
        }

        FMA(Op that, CopyContext cc) {
            super(that, cc);
            this.typeElement = that.resultType();
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            return new FMA(this, copyContext);
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

    private static void customFMA() {

        // 1. Obtain the method instance for the reflected code.
        Optional<Method> myFunction = Stream.of(DialectFMAOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("compute"))
                .findFirst();
        Method m = myFunction.get();

        // 2. Obtain the code model for the function
        CoreOp.FuncOp functionModel = Op.ofMethod(m).get();

        // 3. Print the original code model
        String codeModelString = functionModel.toText();
        System.out.println(codeModelString);

        // 4. Detect if FMA can be applied.
        // To do so, we traverse the original code model and find
        // all AddOp(MultOp)) patterns.

        // data structure used to store all Ops involved, so it will be easier later
        // to transform/replace/eliminate pending the nodes involved in this transformation.
        Stream<CodeElement<?, ?>> elements = functionModel.elements()
                .mapMulti( (codeElement, consumer) -> {
                        // Obtain dependency list of dependencies and check if any of the
                        // input parameters comes from a multiply operation
                    if (codeElement instanceof JavaOp.AddOp addOp) {
                        List<Value> inputOperandsAdd = addOp.operands();
                        Value addDep = inputOperandsAdd.getFirst();
                        if (addDep instanceof Op.Result result) {
                            if (result.op() instanceof JavaOp.MulOp multOp) {
                                // At this point, we know AddOp uses a value from a
                                // result from a multiplication. Thus, we add them
                                // in the processing list
                                consumer.accept(multOp);
                                consumer.accept(addOp);
                            }
                        }
                    }
                });

        // Collect the stream to a HashSet
        Set<CodeElement<?, ?>> nodesInvolved = elements.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            System.out.println("No fma found");
            return;
        }

        // 5. Transform the code model to include the FMA op
        CoreOp.FuncOp dialectModel = functionModel.transform((builder, op) -> {
            CopyContext context = builder.context();
            if (!nodesInvolved.contains(op)) {
                builder.op(op);
            } else if (op instanceof JavaOp.MulOp  mulOp) {
                // If it is only used by one operation, we know it is the one we are replacing.
                // In this case, we can eliminate the node (we don't insert it into the builder)
                if (mulOp.result().uses().size() == 1) {
                    context.mapValue(mulOp.result(), context.getValue(mulOp.operands().getFirst()));
                } else {
                    // We need to insert it into the tree because another non-FMA operation also uses this
                    // operand
                    builder.op(op);
                }
            } else if (op instanceof JavaOp.AddOp addOp) {
                // 6. Obtain the operands for the node
                List<Value> inputOperandsAdd = addOp.operands();

                // Obtain the fist operand and check if the value comes from an Mult Op.
                // In that case, we check if the mult node is contained in the set of
                // involved nodes. If all of this is true, then we replace it with an
                // FMA operation.
                if (addOp.operands().get(0) instanceof Op.Result r &&
                        r.op() instanceof JavaOp.MulOp mulOp
                        && nodesInvolved.contains(mulOp)) {

                    // 7. Create a new Op with the new operation
                    List<Value> inputOperandsMult = mulOp.operands();
                    List<Value> outputAdd = context.getValues(inputOperandsAdd);
                    List<Value> outputMul = context.getValues(inputOperandsMult);
                    List<Value> outFMA = new ArrayList<>();

                    // Build the new parameters list
                    outFMA.addAll(outputMul);      // First two parameters comes from the multiplication.
                    outFMA.add(outputAdd.get(1));  // the last parameter is the second arg to the AddOp

                    FMA myFMA = new FMA(outFMA, addOp.resultType());

                    // 8. Attach the new Op to the builder
                    Op.Result resultFMA = builder.op(myFMA);

                    // 9. Propagate the location of the new op
                    myFMA.setLocation(addOp.location());

                    // 10. Map the values from input -> output for the new Op
                    context.mapValue(addOp.result(), resultFMA);
                }
            }
            return builder;
        });

        // 11. Print the transformed code model
        System.out.println("Model with new OpNodes for Dialect: ");
        System.out.println(dialectModel.toText());

        // 12. Transform to SSA and print the code model
        System.out.println(SSA.transform(dialectModel).toText());

        // Currently, we can't interpreter a code model with dialect ops
        //var result = Interpreter.invoke(MethodHandles.lookup(), dialectModel,  10, 20);
        //System.out.println("Result: ");
    }

    static void main() {
        System.out.println("Testing Dialects in Code-Reflection");
        customFMA();
    }
}
