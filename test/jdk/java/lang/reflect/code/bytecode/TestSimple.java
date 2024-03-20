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

import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.function.Function;
import java.util.function.IntBinaryOperator;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestSimple
 */

public class TestSimple {

    @CodeReflection
    static int f(int i, int j) {
        i = i + j;
        return i;
    }

    @Test
    public void testF() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f");

        MethodHandle mh = generate(f);

        Assert.assertEquals(f(1, 2), (int) mh.invoke(1, 2));
    }

    @Test
    public void testQuoted() throws Throwable {
        Quoted q = (int i, int j) -> {
            i = i + j;
            return i;
        };
        CoreOps.ClosureOp cop = (CoreOps.ClosureOp) q.op();

        MethodHandle mh = generate(cop);

        Assert.assertEquals(f(1, 2), (int) mh.invoke(1, 2));
    }

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
        CoreOps.FuncOp f = getFuncOp("ifelse");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int) mh.invoke(0, 0, 1), ifelse(0, 0, 1));
        Assert.assertEquals((int) mh.invoke(0, 0, 11), ifelse(0, 0, 11));
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
        CoreOps.FuncOp f = getFuncOp("loop");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int) mh.invoke(10), loop(10));
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
        CoreOps.FuncOp f = getFuncOp("ifelseNested");

        MethodHandle mh = generate(f);

        for (int i : new int[]{1, 11, 20, 21}) {
            Assert.assertEquals((int) mh.invoke(0, 0, 0, 0, i), ifelseNested(0, 0, 0, 0, i));
        }
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
    public void testNestedLoop() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("nestedLoop");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int) mh.invoke(10), nestedLoop(10));
    }

    @CodeReflection
    static int methodCallFieldAccess(int a, int b, IntBinaryOperator o) {
        int i = o.applyAsInt(a, b);
        System.out.println(i);
        return i;
    }

    @Test
    public void testMethodCall() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("methodCallFieldAccess");

        MethodHandle mh = generate(f);

        int a = 1;
        int b = 1;
        IntBinaryOperator o = Integer::sum;
        Assert.assertEquals((int) mh.invoke(a, b, o), methodCallFieldAccess(a, b, o));
    }

    @CodeReflection
    static Class<?>[] arrayCreationAndAccess() {
        Class<?>[] ca = new Class<?>[1];
        ca[0] = Function.class;
        return ca;
    }

    @Test
    public void testArrayCreationAndAccess() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("arrayCreationAndAccess");

        MethodHandle mh = generate(f);

        Assert.assertEquals((Class<?>[]) mh.invoke(), arrayCreationAndAccess());
    }

    @CodeReflection
    static int[] primitiveArrayCreationAndAccess() {
        int[] ia = new int[1];
        ia[0] = 42;
        return ia;
    }

    @Test
    public void testPrimitiveArrayCreationAndAccess() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("primitiveArrayCreationAndAccess");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int[]) mh.invoke(), primitiveArrayCreationAndAccess());
    }

    @CodeReflection
    public static boolean not(boolean b) {
        return !b;
    }

    @Test
    public void testNot() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("not");

        MethodHandle mh = generate(f);

        Assert.assertEquals((boolean) mh.invoke(true), not(true));
        Assert.assertEquals((boolean) mh.invoke(false), not(false));
    }

    @CodeReflection
    public static int mod(int a, int b) {
        return a % b;
    }

    @Test
    public void testMod() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("mod");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int) mh.invoke(10, 3), mod(10, 3));
    }

    @CodeReflection
    public static boolean xor(boolean a, boolean b) {
        return a ^ b;
    }

    @Test
    public void testXor() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("xor");

        MethodHandle mh = generate(f);

        Assert.assertEquals((boolean) mh.invoke(true, false), xor(true, false));
    }

    static <O extends Op & Op.Invokable> MethodHandle generate(O f) {
        f.writeTo(System.out);

        @SuppressWarnings("unchecked")
        O lf = (O) f.transform(CopyContext.create(), (block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lf.writeTo(System.out);

        lf = SSA.transform(lf);
        lf.writeTo(System.out);

        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSimple.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
