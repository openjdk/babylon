/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.PrintStream;
import java.lang.reflect.code.*;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.FieldRef;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.lang.reflect.code.op.CoreOp.arrayStoreOp;
import static java.lang.reflect.code.op.CoreOp.constant;
import static java.lang.reflect.code.op.CoreOp.fieldLoad;
import static java.lang.reflect.code.op.CoreOp.newArray;
import static java.lang.reflect.code.type.MethodRef.method;
import static java.lang.reflect.code.type.JavaType.*;

/*
 * @test
 * @enablePreview
 * @run testng TestLocalTransformationsAdaption
 */

public class TestLocalTransformationsAdaption {

    @CodeReflection
    static int f(int i) {
        IntBinaryOperator add = (a, b) -> {
            return add(a, b);
        };

        try {
            IntUnaryOperator add42 = (a) -> {
                return add.applyAsInt(a, 42);
            };

            int j = add42.applyAsInt(i);

            IntBinaryOperator f = (a, b) -> {
                if (i < 0) {
                    throw new RuntimeException();
                }

                IntUnaryOperator g = (c) -> {
                    return add(a, c);
                };

                return g.applyAsInt(b);
            };

            return f.applyAsInt(j, j);
        } catch (RuntimeException e) {
            throw new IndexOutOfBoundsException(i);
        }
    }

    @Test
    public void testInvocation() {
        CoreOp.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        f.writeTo(System.out);

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), f, 2);
        Assert.assertEquals(x, f(2));

        try {
            Interpreter.invoke(MethodHandles.lookup(), f, -10);
            Assert.fail();
        } catch (Throwable e) {
            Assert.assertEquals(IndexOutOfBoundsException.class, e.getClass());
        }
    }

    @Test
    public void testFuncEntryExit() {
        CoreOp.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        AtomicBoolean first = new AtomicBoolean(true);
        CoreOp.FuncOp fc = f.transform((block, op) -> {
            if (first.get()) {
                printConstantString(block, "ENTRY");
                first.set(false);
            }

            switch (op) {
                case CoreOp.ReturnOp returnOp when getNearestInvokeableAncestorOp(returnOp) instanceof CoreOp.FuncOp: {
                    printConstantString(block, "EXIT");
                    break;
                }
                case CoreOp.ThrowOp throwOp: {
                    printConstantString(block, "EXIT");
                    break;
                }
                default:
            }

            block.apply(op);

            return block;
        });
        fc.writeTo(System.out);

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        fc.writeTo(System.out);

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assert.assertEquals(x, f(2));

        try {
            Interpreter.invoke(MethodHandles.lookup(), fc, -10);
            Assert.fail();
        } catch (Throwable e) {
            Assert.assertEquals(IndexOutOfBoundsException.class, e.getClass());
        }
    }

    static void printConstantString(Function<Op, Op.Result> opBuilder, String s) {
        Op.Result c = opBuilder.apply(constant(J_L_STRING, s));
        Value System_out = opBuilder.apply(fieldLoad(FieldRef.field(System.class, "out", PrintStream.class)));
        opBuilder.apply(CoreOp.invoke(method(PrintStream.class, "println", void.class, String.class), System_out, c));
    }

    static Op getNearestInvokeableAncestorOp(Op op) {
        do {
            op = op.ancestorBody().parentOp();
        } while (!(op instanceof Op.Invokable));
        return op;
    }


    @Test
    public void testReplaceCall() {
        CoreOp.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        CoreOp.FuncOp fc = f.transform((block, op) -> {
            switch (op) {
                case CoreOp.InvokeOp invokeOp when invokeOp.invokeDescriptor().equals(ADD_METHOD): {
                    // Get the adapted operands, and pass those to the new call method
                    List<Value> adaptedOperands = block.context().getValues(op.operands());
                    Op.Result adaptedResult = block.apply(CoreOp.invoke(ADD_WITH_PRINT_METHOD, adaptedOperands));
                    // Map the old call result to the new call result, so existing operations can be
                    // adapted to use the new result
                    block.context().mapValue(invokeOp.result(), adaptedResult);
                    break;
                }
                default: {
                    block.apply(op);
                }
            }
            return block;
        });
        fc.writeTo(System.out);

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        fc.writeTo(System.out);

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assert.assertEquals(x, f(2));
    }


    @Test
    public void testCallEntryExit() {
        CoreOp.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        CoreOp.FuncOp fc = f.transform((block, op) -> {
            switch (op) {
                case CoreOp.InvokeOp invokeOp: {
                    printCall(block.context(), invokeOp, block);
                    break;
                }
                default: {
                    block.apply(op);
                }
            }
            return block;
        });
        fc.writeTo(System.out);

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        fc.writeTo(System.out);

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assert.assertEquals(x, f(2));
    }

    static void printCall(CopyContext cc, CoreOp.InvokeOp invokeOp, Function<Op, Op.Result> opBuilder) {
        List<Value> adaptedInvokeOperands = cc.getValues(invokeOp.operands());

        String prefix = "ENTER";

        Value arrayLength = opBuilder.apply(
                constant(INT, adaptedInvokeOperands.size()));
        Value formatArray = opBuilder.apply(
                newArray(type(Object[].class), arrayLength));

        Value indexZero = null;
        for (int i = 0; i < adaptedInvokeOperands.size(); i++) {
            Value operand = adaptedInvokeOperands.get(i);

            Value index = opBuilder.apply(
                    constant(INT, i));
            if (i == 0) {
                indexZero = index;
            }

            if (operand.type().equals(INT)) {
                operand = opBuilder.apply(
                        CoreOp.invoke(method(Integer.class, "valueOf", Integer.class, int.class), operand));
                // @@@ Other primitive types
            }
            opBuilder.apply(
                    arrayStoreOp(formatArray, index, operand));
        }

        Op.Result formatString = opBuilder.apply(
                constant(J_L_STRING,
                        prefix + ": " + invokeOp.invokeDescriptor() + "(" + formatString(adaptedInvokeOperands) + ")%n"));
        Value System_out = opBuilder.apply(fieldLoad(FieldRef.field(System.class, "out", PrintStream.class)));
        opBuilder.apply(
                CoreOp.invoke(method(PrintStream.class, "printf", PrintStream.class, String.class, Object[].class),
                        System_out, formatString, formatArray));

        // Method call

        Op.Result adaptedInvokeResult = opBuilder.apply(invokeOp);

        // After method call

        prefix = "EXIT";

        if (adaptedInvokeResult.type().equals(INT)) {
            adaptedInvokeResult = opBuilder.apply(
                    CoreOp.invoke(method(Integer.class, "valueOf", Integer.class, int.class), adaptedInvokeResult));
            // @@@ Other primitive types
        }
        opBuilder.apply(
                arrayStoreOp(formatArray, indexZero, adaptedInvokeResult));

        formatString = opBuilder.apply(
                constant(J_L_STRING,
                        prefix + ": " + invokeOp.invokeDescriptor() + " -> " + formatString(adaptedInvokeResult.type()) + "%n"));
        opBuilder.apply(
                CoreOp.invoke(method(PrintStream.class, "printf", PrintStream.class, String.class, Object[].class),
                        System_out, formatString, formatArray));
    }

    static String formatString(List<Value> vs) {
        return vs.stream().map(v -> formatString(v.type())).collect(Collectors.joining(","));
    }

    static String formatString(TypeElement t) {
        if (t.equals(INT)) {
            return "%d";
        } else {
            return "%s";
        }
    }


    static final MethodRef ADD_METHOD = MethodRef.method(
            TestLocalTransformationsAdaption.class, "add",
            int.class, int.class, int.class);

    static int add(int a, int b) {
        return a + b;
    }

    static final MethodRef ADD_WITH_PRINT_METHOD = MethodRef.method(
            TestLocalTransformationsAdaption.class, "addWithPrint",
            int.class, int.class, int.class);

    static int addWithPrint(int a, int b) {
        System.out.printf("Adding %d + %d%n", a, b);
        return a + b;
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestLocalTransformationsAdaption.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
