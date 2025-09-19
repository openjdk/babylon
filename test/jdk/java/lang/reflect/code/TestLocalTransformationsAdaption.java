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

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.io.PrintStream;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.IntBinaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.constant;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;
import static jdk.incubator.code.dialect.java.MethodRef.method;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestLocalTransformationsAdaption
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
        System.out.println(f.toText());

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(f.toText());

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), f, 2);
        Assertions.assertEquals(f(2), x);

        try {
            Interpreter.invoke(MethodHandles.lookup(), f, -10);
            Assertions.fail();
        } catch (Throwable e) {
            Assertions.assertEquals(e.getClass(), IndexOutOfBoundsException.class);
        }
    }

    @Test
    public void testFuncEntryExit() {
        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

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
                case JavaOp.ThrowOp throwOp: {
                    printConstantString(block, "EXIT");
                    break;
                }
                default:
            }

            block.op(op);

            return block;
        });
        System.out.println(fc.toText());

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(fc.toText());

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assertions.assertEquals(f(2), x);

        try {
            Interpreter.invoke(MethodHandles.lookup(), fc, -10);
            Assertions.fail();
        } catch (Throwable e) {
            Assertions.assertEquals(e.getClass(), IndexOutOfBoundsException.class);
        }
    }

    static void printConstantString(Block.Builder opBuilder, String s) {
        Op.Result c = opBuilder.op(constant(J_L_STRING, s));
        Value System_out = opBuilder.op(fieldLoad(FieldRef.field(System.class, "out", PrintStream.class)));
        opBuilder.op(JavaOp.invoke(method(PrintStream.class, "println", void.class, String.class), System_out, c));
    }

    static Op getNearestInvokeableAncestorOp(Op op) {
        do {
            op = op.ancestorOp();
        } while (!(op instanceof Op.Invokable));
        return op;
    }


    @Test
    public void testReplaceCall() {
        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

        CoreOp.FuncOp fc = f.transform((block, op) -> {
            switch (op) {
                case JavaOp.InvokeOp invokeOp when invokeOp.invokeDescriptor().equals(ADD_METHOD): {
                    // Get the adapted operands, and pass those to the new call method
                    List<Value> adaptedOperands = block.context().getValues(op.operands());
                    Op.Result adaptedResult = block.op(JavaOp.invoke(ADD_WITH_PRINT_METHOD, adaptedOperands));
                    // Map the old call result to the new call result, so existing operations can be
                    // adapted to use the new result
                    block.context().mapValue(invokeOp.result(), adaptedResult);
                    break;
                }
                default: {
                    block.op(op);
                }
            }
            return block;
        });
        System.out.println(fc.toText());

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(fc.toText());

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assertions.assertEquals(f(2), x);
    }


    @Test
    public void testCallEntryExit() {
        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

        CoreOp.FuncOp fc = f.transform((block, op) -> {
            switch (op) {
                case JavaOp.InvokeOp invokeOp: {
                    printCall(block.context(), invokeOp, block);
                    break;
                }
                default: {
                    block.op(op);
                }
            }
            return block;
        });
        System.out.println(fc.toText());

        fc = fc.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(fc.toText());

        int x = (int) Interpreter.invoke(MethodHandles.lookup(), fc, 2);
        Assertions.assertEquals(f(2), x);
    }

    static void printCall(CopyContext cc, JavaOp.InvokeOp invokeOp, Block.Builder opBuilder) {
        List<Value> adaptedInvokeOperands = cc.getValues(invokeOp.operands());

        String prefix = "ENTER";

        Value arrayLength = opBuilder.op(
                constant(INT, adaptedInvokeOperands.size()));
        Value formatArray = opBuilder.op(
                newArray(type(Object[].class), arrayLength));

        Value indexZero = null;
        for (int i = 0; i < adaptedInvokeOperands.size(); i++) {
            Value operand = adaptedInvokeOperands.get(i);

            Value index = opBuilder.op(
                    constant(INT, i));
            if (i == 0) {
                indexZero = index;
            }

            if (operand.type().equals(INT)) {
                operand = opBuilder.op(
                        JavaOp.invoke(method(Integer.class, "valueOf", Integer.class, int.class), operand));
                // @@@ Other primitive types
            }
            opBuilder.op(
                    arrayStoreOp(formatArray, index, operand));
        }

        Op.Result formatString = opBuilder.op(
                constant(J_L_STRING,
                        prefix + ": " + invokeOp.invokeDescriptor() + "(" + formatString(adaptedInvokeOperands) + ")%n"));
        Value System_out = opBuilder.op(fieldLoad(FieldRef.field(System.class, "out", PrintStream.class)));
        opBuilder.op(
                JavaOp.invoke(method(PrintStream.class, "printf", PrintStream.class, String.class, Object[].class),
                        System_out, formatString, formatArray));

        // Method call

        Op.Result adaptedInvokeResult = opBuilder.op(invokeOp);

        // After method call

        prefix = "EXIT";

        if (adaptedInvokeResult.type().equals(INT)) {
            adaptedInvokeResult = opBuilder.op(
                    JavaOp.invoke(method(Integer.class, "valueOf", Integer.class, int.class), adaptedInvokeResult));
            // @@@ Other primitive types
        }
        opBuilder.op(
                arrayStoreOp(formatArray, indexZero, adaptedInvokeResult));

        formatString = opBuilder.op(
                constant(J_L_STRING,
                        prefix + ": " + invokeOp.invokeDescriptor() + " -> " + formatString(adaptedInvokeResult.type()) + "%n"));
        opBuilder.op(
                JavaOp.invoke(method(PrintStream.class, "printf", PrintStream.class, String.class, Object[].class),
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
        return Op.ofMethod(m).get();
    }
}
