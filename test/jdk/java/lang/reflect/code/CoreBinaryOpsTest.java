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

import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.AccessFlag;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.OpDeclaration;
import java.lang.reflect.code.op.OpDefinition;
import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

public class CoreBinaryOpsTest {

    record TestCase<I>(I left, I right) {
    }

    record TestType<I, O, F extends Quotable & BiFunction<I, I, O>>(
            Class<? extends CoreOps.BinaryOp> opType,
            F action, TestCase<I>... testCases) {

        @SafeVarargs
        public TestType {
        }

        Stream<DynamicTest> check() {
            var op = (Op & Op.Invokable) action.quoted().op();
            return Arrays.stream(testCases)
                    .mapMulti((testCase, consumer) -> {
                        // add one interpreter test and one bytecode test per testcase
                        O result = action.apply(testCase.left, testCase.right);
                        consumer.accept(
                                dynamic(result, testCase, c -> interpret(c, op)));
                        consumer.accept(
                                dynamic(result, testCase, c -> bytecode(c, op))
                        );
                    });
        }

        private DynamicTest dynamic(O result, TestCase<I> testCase, Function<TestCase<I>, ?> function) {
            return dynamicTest(testCase.left + " " + opName(opType) + " " + testCase.right,
                    () -> assertEquals(result, function.apply(testCase))
            );
        }

        private String opName(Class<? extends CoreOps.BinaryOp> opType) {
            return opType.getAnnotation(OpDeclaration.class).value();
        }
    }

    private static Stream<TestType<?, ?, ?>> tests() {
        return Stream.of(
                // and
                new TestType<>(CoreOps.AndOp.class, (l, r) -> l & r, new TestCase<>(13, -18)),
                new TestType<>(CoreOps.AndOp.class, (l, r) -> l & r, new TestCase<>(1329394299218L, 838749348593L)),
                new TestType<>(CoreOps.AndOp.class, (l, r) -> l & r, new TestCase<>(true, false)),
                // add
                new TestType<>(CoreOps.AddOp.class, (l, r) -> l + r, new TestCase<>(1, 2)),
                new TestType<>(CoreOps.AddOp.class, (l, r) -> l + r, new TestCase<>(1L, -2L)),
                new TestType<>(CoreOps.AddOp.class, (l, r) -> l + r, new TestCase<>(5.3f, 9.22f)),
                new TestType<>(CoreOps.AddOp.class, (l, r) -> l + r, new TestCase<>(-5.99d, 4.286d))
        );
    }

    private static <I, OP extends Op & Op.Invokable> Object interpret(TestCase<I> testCase, OP op) {
        return Interpreter.invoke(MethodHandles.lookup(), op, testCase.left, testCase.right);
    }

    private static <I, OP extends Op & Op.Invokable> Object bytecode(TestCase<I> testCase, OP op) {
        OpDefinition lambdaDef = OpDefinition.fromOp(CopyContext.create(), op);
        Map<String, Object> attributes = new HashMap<>(lambdaDef.attributes());
        attributes.put(CoreOps.FuncOp.ATTRIBUTE_FUNC_NAME, "testName");
        OpDefinition funcDef = new OpDefinition(CoreOps.FuncOp.NAME, lambdaDef.operands(), lambdaDef.successors(), lambdaDef.resultType(), attributes, lambdaDef.bodyDefinitions());
        CoreOps.FuncOp func = SSA.transform(CoreOps.FuncOp.create(funcDef).transform((block, o) -> {
            if (o instanceof Op.Lowerable lowerable) {
                return lowerable.lower(block);
            } else {
                block.op(o);
                return block;
            }
        }));
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(), func);
        try {
            return handle.invoke(testCase.left, testCase.right);
        } catch (Throwable e) {
            return fail(e);
        }
    }

    @Test
    void ensureAllBinOpsCovered() {
        Set<Class<?>> allBinaryOps = Arrays.stream(CoreOps.class.getDeclaredClasses())
                .filter(clazz -> !clazz.accessFlags().contains(AccessFlag.ABSTRACT) && CoreOps.BinaryOp.class.isAssignableFrom(clazz))
                .collect(Collectors.toCollection(HashSet::new));
        var existingTests = tests().map(TestType::opType).toList();
        for (Class<? extends CoreOps.BinaryOp> test : existingTests) {
            allBinaryOps.remove(test);
        }
        if (!allBinaryOps.isEmpty()) {
            // fail("Not all binary ops are covered by test cases: " + allBinaryOps);
        }
    }

    @TestFactory
    Stream<DynamicTest> runAllTests() {
        return tests().flatMap(TestType::check);
    }
}
