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

/*
 * @test
 * @run junit CoreBinaryOps2Test
 */

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;
import org.junit.jupiter.api.function.Executable;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.code.*;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

public class CoreBinaryOps2Test {

    record TestInput(Object left, Object right) {
    }

    private static Stream<Method> codeReflectionMethods() {
        return Arrays.stream(CoreBinaryOps2Test.class.getDeclaredMethods())
                .filter(method -> method.accessFlags().contains(AccessFlag.STATIC))
                .filter(method -> method.getCodeModel().isPresent());
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @interface SupportedTypes {
        Class<?>[] types();
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, boolean.class})
    static int and(int left, int right) {
        return left & right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, float.class, double.class})
    static int add(int left, int right) {
        return left + right;
    }


    private static Object interpret(TestInput testInput, CoreOps.FuncOp op) {
        return Interpreter.invoke(MethodHandles.lookup(), op, testInput.left, testInput.right);
    }

    private static Object bytecode(TestInput testInput, CoreOps.FuncOp op) {
        CoreOps.FuncOp func = SSA.transform(op.transform((block, o) -> {
            if (o instanceof Op.Lowerable lowerable) {
                return lowerable.lower(block);
            } else {
                block.op(o);
                return block;
            }
        }));
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(), func);
        try {
            return handle.invoke(testInput.left, testInput.right);
        } catch (Throwable e) {
            return fail(e);
        }
    }

    @Test
    void ensureAllBinOpsCovered() {
        Set<Class<?>> allBinaryOps = Arrays.stream(CoreOps.class.getDeclaredClasses())
                .filter(clazz -> !clazz.accessFlags().contains(AccessFlag.ABSTRACT) && CoreOps.BinaryOp.class.isAssignableFrom(clazz))
                .collect(Collectors.toCollection(HashSet::new));
        List<List<Op>> binaryOpsPerMethod = codeReflectionMethods()
                .flatMap(method -> method.getCodeModel().stream())
                .map(CoreOps.FuncOp::bodies)
                .map(l -> l.stream().flatMap(body -> body.blocks().stream()).flatMap(block -> block.ops().stream()))
                .map(s -> s.filter(op -> op instanceof CoreOps.BinaryOp))
                .map(Stream::toList)
                .toList();
        Executable[] tests = new Executable[binaryOpsPerMethod.size()];
        for (int i = 0; i < binaryOpsPerMethod.size(); i++) {
            List<Op> ops = binaryOpsPerMethod.get(i);
            tests[i] = () -> {
                assertEquals(1, ops.size(), () -> "Expects exactly one binary op per method, got " + ops);
                FunctionType type = ops.getFirst().parent().parent().bodyType();
                assertEquals(2, type.parameterTypes().size(), "Binary op method should have two parameters");
                // only same types
                assertEquals(type.returnType(), type.parameterTypes().getFirst());
                assertEquals(type.parameterTypes().getFirst(), type.parameterTypes().getLast());
            };
        }
        assertAll(tests);

        for (List<Op> ops : binaryOpsPerMethod) {
            allBinaryOps.remove(ops.getFirst().getClass());
        }

        if (!allBinaryOps.isEmpty()) {
            // fail("Not all binary ops are covered by test cases: " + allBinaryOps);
        }
    }

    @TestFactory
    Stream<DynamicTest> runAllTests() {
        List<Method> list = codeReflectionMethods().toList();
        Map<CoreOps.FuncOp, Class<?>[]> types = new HashMap<>();
        Stream.Builder<DynamicTest> compareToReflection = Stream.builder();
        for (Method method : list) {
            Class<?>[] classes = Optional.ofNullable(method.getAnnotation(SupportedTypes.class))
                    .map(SupportedTypes::types)
                    .orElseGet(() -> new Class[0]);
            CoreOps.FuncOp funcOp = method.getCodeModel().orElseThrow();
            types.put(funcOp, classes);
            FunctionType type = funcOp.invokableType();
            TestInput input = new TestInput(valueForType(type.returnType()), valueForType(type.returnType()));
            compareToReflection.add(dynamicTest("reflection " + method, () -> {
                Result reflection = runCatching(() -> {
                    try {
                        return method.invoke(null, input.left, input.right);
                    } catch (IllegalAccessException | InvocationTargetException e) {
                        throw new RuntimeException(e);
                    }
                });
                Result interpret = runCatching(() -> interpret(input, funcOp));
                assertResults(reflection, interpret);
            }));
        }
        Stream<DynamicTest> compareInterpreterBytecode = list.stream().map(Method::getCodeModel)
                .flatMap(Optional::stream)
                .flatMap(func -> Arrays.stream(types.get(func)).map(type -> retype(func, type)))
                .map(func -> {
                    FunctionType type = func.invokableType();
                    TestInput input = new TestInput(valueForType(type.returnType()), valueForType(type.returnType()));

                    return dynamicTest(input.left + " " + func.funcName() + " " + input.right,
                            () -> {
                                System.out.println("Running test " + input.left + " " + func.funcName() + " " + input.right);
                                Result interpret = runCatching(() -> interpret(input, func));
                                Result bytecode = runCatching(() -> bytecode(input, func));
                                assertResults(interpret, bytecode);
                            }
                    );
                });
        return Stream.concat(compareToReflection.build(), compareInterpreterBytecode);
    }

    private static void assertResults(Result first, Result second) {
        // either the same error occurred on both or no error occurred
        if (first.throwable != null || second.throwable != null) {
            assertNotNull(first.throwable);
            assertNotNull(second.throwable);
            assertEquals(first.throwable.getClass(), second.throwable.getClass());
        }
        // otherwise, both results should be non-null and equals
        assertNotNull(first.onSuccess);
        assertEquals(first.onSuccess, second.onSuccess);
    }

    private static <T> Result runCatching(Supplier<T> supplier) {
        Object value = null;
        Throwable interpretThrowable = null;
        try {
            value = supplier.get();
        } catch (Throwable t) {
            interpretThrowable = t;
        }
        return new Result(value, interpretThrowable);
    }

    record Result(Object onSuccess, Throwable throwable) {
    }

    static CoreOps.FuncOp retype(CoreOps.FuncOp original, Class<?> newType) {
        JavaType type = JavaType.type(newType);
        if (original.resultType().equals(type)) {
            return original; // already expected type
        }
        return CoreOps.func(original.funcName(), FunctionType.functionType(type, type, type))
                .body(builder -> builder.transformBody(original.body(), builder.parameters(), (block, op) -> {
                            block.context().mapValue(op.result(), block.op(retype(block.context(), op)));
                            return block;
                        })
                );
    }

    static Op retype(CopyContext context, Op op) {
        return switch (op) {
            case CoreOps.VarOp varOp ->
                    CoreOps.var(varOp.varName(), context.getValueOrDefault(varOp.operands().getFirst(), varOp.operands().getFirst()));
            default -> op;
        };
    }

    // can be improved to return random values (?)
    static Object valueForType(TypeElement type) {
        if (type.equals(JavaType.INT)) {
            return 1;
        }
        if (type.equals(JavaType.LONG)) {
            return 1L;
        }
        if (type.equals(JavaType.DOUBLE)) {
            return 1d;
        }
        if (type.equals(JavaType.FLOAT)) {
            return 1f;
        }
        if (type.equals(JavaType.BOOLEAN)) {
            return true;
        }
        throw new IllegalArgumentException(type + " is not supported");
    }
}
