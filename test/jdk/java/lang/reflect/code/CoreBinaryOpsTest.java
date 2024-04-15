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
 * @run junit CoreBinaryOpsTest
 */

import org.junit.jupiter.api.Named;
import org.junit.jupiter.api.extension.ExtensionContext;
import org.junit.jupiter.api.function.ThrowingSupplier;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.ArgumentsProvider;
import org.junit.jupiter.params.provider.ArgumentsSource;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

public class CoreBinaryOpsTest {

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

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, float.class, double.class})
    static int div(int left, int right) {
        return left / right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class})
    static int leftShift(int left, int right) {
        return left << right;
    }

    @CodeReflection
    @Direct
    static int leftShiftIL(int left, long right) {
        return left << right;
    }

    @CodeReflection
    @Direct
    static long leftShiftLI(long left, int right) {
        return left << right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, float.class, double.class})
    static int mod(int left, int right) {
        return left % right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, float.class, double.class})
    static int mul(int left, int right) {
        return left * right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, boolean.class})
    static int or(int left, int right) {
        return left | right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class})
    static int signedRightShift(int left, int right) {
        return left >> right;
    }

    @CodeReflection
    @Direct
    static int signedRightShiftIL(int left, long right) {
        return left >> right;
    }

    @CodeReflection
    @Direct
    static long signedRightShiftLI(long left, int right) {
        return left >> right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, float.class, double.class})
    static int sub(int left, int right) {
        return left - right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class})
    static int unsignedRightShift(int left, int right) {
        return left >>> right;
    }

    @CodeReflection
    @Direct
    static int unsignedRightShiftIL(int left, long right) {
        return left >>> right;
    }

    @CodeReflection
    @Direct
    static long unsignedRightShiftLI(long left, int right) {
        return left >>> right;
    }

    @CodeReflection
    @SupportedTypes(types = {int.class, long.class, boolean.class})
    static int xor(int left, int right) {
        return left ^ right;
    }

    @ParameterizedTest
    @CodeReflectionExecutionSource
    void test(CoreOps.FuncOp funcOp, Object left, Object right) {
        Result interpret = runCatching(() -> interpret(left, right, funcOp));
        Result bytecode = runCatching(() -> bytecode(left, right, funcOp));
        assertResults(interpret, bytecode);
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @interface SupportedTypes {
        Class<?>[] types();
    }

    // mark as "do not transform"
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @interface Direct {
    }

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @ArgumentsSource(CodeReflectionSourceProvider.class)
    @interface CodeReflectionExecutionSource {
    }

    static class CodeReflectionSourceProvider implements ArgumentsProvider {
        private static final Map<JavaType, List<Object>> INTERESTING_INPUTS = Map.of(
                JavaType.INT, List.of(Integer.MIN_VALUE, Integer.MAX_VALUE, 1, 0, -1),
                JavaType.LONG, List.of(Long.MIN_VALUE, Long.MAX_VALUE, 1, 0, -1),
                JavaType.DOUBLE, List.of(Double.MIN_VALUE, Double.MAX_VALUE, Double.NaN, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.MIN_NORMAL, 1, 0, -1),
                JavaType.FLOAT, List.of(Float.MIN_VALUE, Float.MAX_VALUE, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.MIN_NORMAL, 1, 0, -1),
                JavaType.BOOLEAN, List.of(true, false)
        );

        @Override
        public Stream<? extends Arguments> provideArguments(ExtensionContext extensionContext) {
            Method testMethod = extensionContext.getRequiredTestMethod();
            return codeReflectionMethods(extensionContext.getRequiredTestClass())
                    .flatMap(method -> {
                        CoreOps.FuncOp funcOp = method.getCodeModel().orElseThrow(
                                () -> new IllegalStateException("Expected code model to be present for method " + method)
                        );
                        SupportedTypes supportedTypes = method.getAnnotation(SupportedTypes.class);
                        if (method.isAnnotationPresent(Direct.class)) {
                            if (supportedTypes != null) {
                                throw new IllegalArgumentException("Direct should not be combined with SupportedTypes");
                            }
                            return Stream.of(funcOp);
                        }
                        if (supportedTypes == null || supportedTypes.types().length == 0) {
                            throw new IllegalArgumentException("Missing supported types");
                        }
                        return Arrays.stream(supportedTypes.types())
                                .map(type -> retype(funcOp, type));
                    })
                    .flatMap(transformedFunc -> argumentsForMethod(transformedFunc, testMethod));
        }

        private static <T> Stream<List<T>> cartesianProduct(List<List<T>> source) {
            if (source.isEmpty()) {
                return Stream.of(new ArrayList<>());
            }
            return source.getFirst().stream()
                    .flatMap(e -> cartesianProduct(source.subList(1, source.size())).map(l -> {
                        ArrayList<T> newList = new ArrayList<>(l);
                        newList.add(e);
                        return newList;
                    }));
        }

        private static CoreOps.FuncOp retype(CoreOps.FuncOp original, Class<?> newType) {
            JavaType type = JavaType.type(newType);
            FunctionType functionType = original.invokableType();
            if (functionType.parameterTypes().stream().allMatch(t -> t.equals(type))) {
                return original; // already expected type
            }
            if (functionType.parameterTypes().stream().distinct().count() != 1) {
                original.writeTo(System.err);
                throw new IllegalArgumentException("Only FuncOps with exactly one distinct parameter type are supported");
            }
            // if the return type does not match the input types, we keep it
            TypeElement retType = functionType.returnType().equals(functionType.parameterTypes().getFirst())
                    ? type
                    : functionType.returnType();
            return CoreOps.func(original.funcName(), FunctionType.functionType(retType, type, type))
                    .body(builder -> builder.transformBody(original.body(), builder.parameters(), (block, op) -> {
                                block.context().mapValue(op.result(), block.op(retype(block.context(), op)));
                                return block;
                            })
                    );
        }

        private static Op retype(CopyContext context, Op op) {
            return switch (op) {
                case CoreOps.VarOp varOp ->
                        CoreOps.var(varOp.varName(), context.getValueOrDefault(varOp.operands().getFirst(), varOp.operands().getFirst()));
                default -> op;
            };
        }

        private static Stream<Arguments> argumentsForMethod(CoreOps.FuncOp funcOp, Method testMethod) {
            Parameter[] testMethodParameters = testMethod.getParameters();
            List<TypeElement> funcParameters = funcOp.invokableType().parameterTypes();
            if (testMethodParameters.length - 1 != funcParameters.size()) {
                throw new IllegalArgumentException("method " + testMethod + " does not take the correct number of parameters");
            }
            if (testMethodParameters[0].getType() != CoreOps.FuncOp.class) {
                throw new IllegalArgumentException("method " + testMethod + " does not take a leading FuncOp argument");
            }
            Named<CoreOps.FuncOp> opNamed = Named.of(funcOp.funcName() + "{" + funcOp.invokableType() + "}", funcOp);
            MethodHandles.Lookup lookup = MethodHandles.lookup();
            for (int i = 1; i < testMethodParameters.length; i++) {
                Class<?> resolved = resolveParameter(funcParameters.get(i - 1), lookup);
                if (!isCompatible(resolved, testMethodParameters[i].getType())) {
                    System.out.println(testMethod + " does not accept inputs of type " + resolved + " at index " + i);
                    return Stream.empty();
                }
            }
            List<List<Object>> allInputs = new ArrayList<>();
            for (TypeElement parameterType : funcParameters) {
                allInputs.add(INTERESTING_INPUTS.get((JavaType) parameterType));
            }
            return cartesianProduct(allInputs)
                    .map(objects -> {
                        objects.add(opNamed);
                        return objects.reversed().toArray(); // reverse so FuncOp is at the beginning
                    })
                    .map(Arguments::of);
        }

        private static Class<?> resolveParameter(TypeElement typeElement, MethodHandles.Lookup lookup) {
            try {
                return ((JavaType) typeElement).resolve(lookup);
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        }

        // check whether elements of type sourceType can be passed to a parameter of parameterType
        private static boolean isCompatible(Class<?> sourceType, Class<?> parameterType) {
            return wrapped(parameterType).isAssignableFrom(wrapped(sourceType));
        }

        private static Class<?> wrapped(Class<?> target) {
            return MethodType.methodType(target).wrap().returnType();
        }

        private static Stream<Method> codeReflectionMethods(Class<?> testClass) {
            return Arrays.stream(testClass.getDeclaredMethods())
                    .filter(method -> method.accessFlags().contains(AccessFlag.STATIC))
                    .filter(method -> method.isAnnotationPresent(CodeReflection.class));
        }

    }

    private static Object interpret(Object left, Object right, CoreOps.FuncOp op) {
        return Interpreter.invoke(MethodHandles.lookup(), op, left, right);
    }

    private static Object bytecode(Object left, Object right, CoreOps.FuncOp op) throws Throwable {
        CoreOps.FuncOp func = SSA.transform(op.transform((block, o) -> {
            if (o instanceof Op.Lowerable lowerable) {
                return lowerable.lower(block);
            } else {
                block.op(o);
                return block;
            }
        }));
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(), func);
        return handle.invoke(left, right);
    }

    private static void assertResults(Result first, Result second) {
        System.out.println("first: " + first);
        System.out.println("second: " + second);
        // either the same error occurred on both or no error occurred
        if (first.throwable != null || second.throwable != null) {
            assertNotNull(first.throwable, "only second threw an exception");
            assertNotNull(second.throwable, "only first threw an exception");
            if (first.throwable.getClass() != second.throwable.getClass()) {
                first.throwable.printStackTrace();
                second.throwable.printStackTrace();
                fail("Different exceptions were thrown");
            }
            return;
        }
        // otherwise, both results should be non-null and equals
        assertNotNull(first.onSuccess);
        assertEquals(first.onSuccess, second.onSuccess);
    }

    private static <T> Result runCatching(ThrowingSupplier<T> supplier) {
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

}
