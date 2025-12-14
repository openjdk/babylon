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
 * @modules jdk.incubator.code
 * @run junit TestModuleOp
 * @run junit/othervm -Dbabylon.ssa=cytron TestModuleOp
 */

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

import static jdk.incubator.code.dialect.core.CoreOp.*;
import static jdk.incubator.code.dialect.java.JavaOp.*;
import static jdk.incubator.code.dialect.java.JavaType.*;

public class TestModuleOp {

    @Reflect
    public static void a() {
    }

    @Reflect
    public static int b(int i) {
        return i;
    }

    @Reflect
    public static int c(int i) {
        return (i * 2) % 5;
    }

    @Reflect
    public static int d(int i, int j) {
        return (c(i) / 4) + j;
    }

    @Reflect
    public static int e(int i) {if (i == 0) return i; return e(--i);}

    @Test
    public void testEmptyLambda() {
        Runnable runnable = (@Reflect Runnable) () -> {};
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(runnable).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("rootLambda_0")));
        Assertions.assertTrue(moduleOp.functionTable().get("rootLambda_0").elements().allMatch(e -> e instanceof ReturnOp || e instanceof FuncOp || !(e instanceof Op)));
    }

    @Test
    public void testSingleInvoke() {
        Runnable runnable = (@Reflect Runnable) () -> a();
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(runnable).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("a_0")));
    }

    @Test
    public void testMultipleCapture() {
        int i = 0;
        int j = 0;
        Runnable runnable = (@Reflect Runnable) () -> d(i, j);
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(runnable).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("d_0", "c_1")));
    }

    @Test
    public void testArray() throws NoSuchMethodException {
        int i = 10;
        int[] array = new int[i];
        Consumer<Integer> lambda = (@Reflect Consumer<Integer>) (j) -> c(b(i) + array[j]);
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(lambda).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("rootLambda_0", "b_1", "c_2")));
    }

    @Test
    public void testTernary() throws NoSuchMethodException {
        int i = 0;
        int j = 0;
        Consumer<Integer> lambda = (@Reflect Consumer<Integer>) (k) -> c(i > k ? b(i) : c(d(j, k)));
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(lambda).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("rootLambda_0", "b_1", "d_2", "c_3")));
    }

    @Test
    public void testRepeatLambdaName() {
        @Reflect IntUnaryOperator runnable = (@Reflect IntUnaryOperator) (int j) -> {return b(1);};
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(runnable).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "b");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("b_0", "b_1")));
    }

    @Test
    public void testMultipleStatements() throws NoSuchMethodException {
        int i = 0;
        int j = 0;
        Consumer<Integer> lambda = (@Reflect Consumer<Integer>) (k) -> {
           int temp = c(i);
           a();
           b(k * d(temp, j));
        };
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(lambda).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "rootLambda");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("rootLambda_0", "c_1", "a_2", "d_3", "b_4")));
    }

    @Test
    public void testRecursion() throws ReflectiveOperationException {
        Runnable runnable = (@Reflect Runnable) () -> e(1);
        LambdaOp lambdaOp = (LambdaOp) Op.ofQuotable(runnable).get().op();
        ModuleOp moduleOp = ModuleOp.ofLambdaOp(lambdaOp, MethodHandles.lookup(), "e");
        Assertions.assertTrue(moduleOp.functionTable().keySet().stream().toList().equals(List.of("e_0", "e_1")));
    }
}
