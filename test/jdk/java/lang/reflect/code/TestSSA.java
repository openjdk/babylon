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

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.function.IntSupplier;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestSSA
 * @run testng/othervm -Dbabylon.ssa=cytron TestSSA
 */

public class TestSSA {

    @CodeReflection
    static int ifelse(int a, int b, int n) {
        if (n < 10) {
            a += 1;
        } else {
            b += 2;
        }
        return a + b;
    }

    @Test
    public void testIfelse() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("ifelse");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 0, 0, 1), ifelse(0, 0, 1));
        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 0, 0, 11), ifelse(0, 0, 11));
    }

    @CodeReflection
    static int ifelseNested(int a, int b, int c, int d, int n) {
        if (n < 20) {
            if (n < 10) {
                a += 1;
            } else {
                b += 2;
            }
            c += 3;
        } else {
            if (n > 20) {
                a += 4;
            } else {
                b += 5;
            }
            d += 6;
        }
        return a + b + c + d;
    }

    @Test
    public void testIfelseNested() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("ifelseNested");

        CoreOp.FuncOp lf = generate(f);

        for (int i : new int[]{1, 11, 20, 21}) {
            Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 0, 0, 0, 0, i), ifelseNested(0, 0, 0, 0, i));
        }
    }

    @CodeReflection
    static int loop(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum = sum + i;
        }
        return sum;
    }

    @Test
    public void testLoop() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("loop");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 10), loop(10));
    }

    @CodeReflection
    static int nestedLoop(int n) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum = sum + i + j;
            }
        }
        return sum;
    }

    @Test
    public void testNestedLoop() {
        CoreOp.FuncOp f = getFuncOp("nestedLoop");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 10), nestedLoop(10));
    }

    @CodeReflection
    static int nestedLambdaCapture(int i) {
        IntSupplier s = () -> {
            int j = i + 1;
            IntSupplier s2 = () -> i + j;
            return s2.getAsInt() + i;
        };
        return s.getAsInt();
    }

    @Test
    public void testNestedLambdaCapture() {
        CoreOp.FuncOp f = getFuncOp("nestedLambdaCapture");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 10), nestedLambdaCapture(10));
    }

    static CoreOp.FuncOp generate(CoreOp.FuncOp f) {
        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(lf.toText());

        lf = SSA.transform(lf);
        System.out.println(lf.toText());
        return lf;
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSSA.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
