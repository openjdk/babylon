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

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.testng.annotations.DataProvider;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.CodeReflection;
import java.util.*;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestOverloads
 */

public class TestOverloads {

    @CodeReflection
    static int f() {
       return 0;
    }

    @CodeReflection
    static int f(int i) {
       return 1;
    }

    @CodeReflection
    static int f(Integer i) {
       return 2;
    }

    @CodeReflection
    static int f(Object o) {
       return 3;
    }

    @CodeReflection
    static int f(List<Integer> l) {
       return 4;
    }

    @CodeReflection
    static int f(List<Integer> l, Object o) {
       return 5;
    }

    @DataProvider(name = "testData")
    public static Object[][]  testData() {
        return new Object[][]{
            new Object[] {new Class[]{}, new Object[]{}},
            new Object[] {new Class[]{int.class}, new Object[]{-1}},
            new Object[] {new Class[]{Integer.class}, new Object[]{-1}},
            new Object[] {new Class[]{Object.class}, new Object[]{"hello"}},
            new Object[] {new Class[]{List.class}, new Object[]{List.of()}},
            new Object[] {new Class[]{List.class, Object.class}, new Object[]{List.of(), -1}}
        };
    }

    @Test(dataProvider = "testData")
    public static void testOverloads(Class<?>[] paramTypes, Object[] params) {
        try {
            Class<TestOverloads> clazz = TestOverloads.class;
            Method method = clazz.getDeclaredMethod("f", paramTypes);
            CoreOp.FuncOp f = Op.ofMethod(method).orElseThrow();
            var res1 = Interpreter.invoke(MethodHandles.lookup(), f, params);
            var res2 = method.invoke(null, params);

            Assert.assertEquals(res1, res2);

        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }
}
