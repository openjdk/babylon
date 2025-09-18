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
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Simple example demonstrating how to create new nodes to build a new dialect.
 * <p>
 * This example customizes invoke ops to handle them as intrinsics operations.
 * The example showcase how to search for Java methods with specific signature (in this case
 * FMA), and replace them with Op that performs the FMA operation.
 * </p>
 *
 * <p>
 *     How to run from the terminal?
 *     <code>
 *         $ java --add-modules jdk.incubator.code -cp target/crsamples-1.0-SNAPSHOT.jar oracle.code.samples.DialectWithInvoke
 *     </code>
 * </p>
 */
public class DialectWithInvoke {

    private static float intrinsicsFMA(float a, float b, float c) {
        return Math.fma(a, b, c);
    }

    @CodeReflection
    public static float myFunction(float a, float b, float c) {
        return intrinsicsFMA(a, b, c);
    }

    // Custom/Dialect Nodes extends from Op
    public static class FMAIntrinsicOp extends Op { // externalized

        private final TypeElement typeDescriptor;

        FMAIntrinsicOp(String opName, TypeElement typeDescriptor, List<Value> operands) {
            super(opName, operands);
            this.typeDescriptor = typeDescriptor;
        }

        FMAIntrinsicOp(Op that, CopyContext cc) {
            super(that, cc);
            this.typeDescriptor = that.resultType();
        }

        @Override
        public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
            return new FMAIntrinsicOp(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeDescriptor;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("", "dialect." + this.opName());
        }
    }

    private static void customInvoke() {

        Optional<Method> myFunction = Stream.of(DialectWithInvoke.class.getDeclaredMethods())
                .filter(m -> m.getName().equals("myFunction"))
                .findFirst();
        Method m = myFunction.get();

        // Original Code Mode.
        CoreOp.FuncOp functionModel = Op.ofMethod(m).get();
        System.out.println(functionModel.toText());

        // Transform the code model to search for all InvokeOp and check if the
        // method name matches with the one we want to replace. We could also check
        // parameters and their types. For simplication, this example does not check this.
        CoreOp.FuncOp dialectModel = functionModel.transform((blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (op instanceof JavaOp.InvokeOp invokeOp && invokeOp.invokeDescriptor().name().equals("intrinsicsFMA")) {
                // The Op is the one we are looking for.
                // We obtain the input values to this Op and use them to build the new FMA op.
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(inputOperands);

                // Create new node
                FMAIntrinsicOp myCustomFunction = new FMAIntrinsicOp("intrinsicsFMA", invokeOp.resultType(), outputOperands);

                // Add the new node to the code builder
                Op.Result outputResult = blockBuilder.op(myCustomFunction);

                // Preserve the location from the original invoke
                myCustomFunction.setLocation(invokeOp.location());

                // Map input-> new output
                context.mapValue(invokeOp.result(), outputResult);
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

        // Currently, we can't interpreter a code model with dialect ops
        //var result = Interpreter.invoke(MethodHandles.lookup(), ssaDialect,  10, 20);
        //System.out.println("Result: " + result);
    }

    static void main() {
        System.out.println("Testing Dialects in Code-Reflection");
        customInvoke();
    }
}
