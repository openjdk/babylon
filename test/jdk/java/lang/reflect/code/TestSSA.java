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

    @CodeReflection
    static int deadCode(int n) {
        int factorial = 1;
        int unused = factorial;
        int unusedLoop = 0;
        for (int i = 0; i < 4; i++) {
            unusedLoop++;
        }
        while (n > 0) {
            factorial *= n;
            n--;
            if (factorial == 0) {
                factorial = -1;
                int unusedNested = factorial;
            }
        }
        return factorial;
    }

    @Test
    public void testDeadCode() {
        CoreOp.FuncOp f = getFuncOp("deadCode");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 10), deadCode(10));
    }

    @CodeReflection
    static int ifelseLoopNested(int n) {
        int counter = 10;
        while (n > 0) {
            if (n % 2 == 0) {
                int sum = n;
                for (int i = 0; i < 5; i++) {
                    if (sum > n / 2) {
                        sum -= i;
                    } else {
                        sum += i;
                        break;
                    }
                }
                n += sum;
            } else {
                int difference = (n % 3 == 0) ? n / 2 : n * 2;
                n -= difference;
            }
            counter--;
        }
        return n;
    }

    @Test
    public void testIfelseLoopNested() {
        CoreOp.FuncOp f = getFuncOp("ifelseLoopNested");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 10), ifelseLoopNested(10));
    }

    @CodeReflection
    static int violaJones(int x, int maxX, int length, int integral) {
        int scale = 0;
        scale++;
        while (x > scale && scale < length) {
        }
        for (int i = 0; i < integral; i++) {
            scale--;
        }
        return scale;
    }

    @Test
    public void testViolaJones() {
        CoreOp.FuncOp f = getFuncOp("violaJones");

        CoreOp.FuncOp lf = generate(f);

        Assert.assertEquals((int) Interpreter.invoke(MethodHandles.lookup(), lf, 0, 1, 0, 0), violaJones(0, 1, 0, 0));
    }

    @CodeReflection
    static boolean binarySearch(int[] arr, int target) {
        int l = 0;
        int r = arr.length - 1;
        while (l < r) {
            int m = (r - l) / 2;
            m += l;
            if (arr[m] < target) {
                l = m + 1;
            } else if (arr[m] > target) {
                r = m - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    @Test
    public void testBinarySearch() {
        CoreOp.FuncOp f = getFuncOp("binarySearch");

        CoreOp.FuncOp lf = generate(f);

        int[] arr = new int[]{1, 2, 4, 7, 11, 19, 21, 29, 30, 36};

        Assert.assertEquals((boolean) Interpreter.invoke(MethodHandles.lookup(), lf, arr, 4), binarySearch(arr, 4));
    }

    @CodeReflection
    static void quicksort(int[] arr, int lo, int hi) {
        if (lo >= hi || lo < 0) {
            return;
        }

        int pivot = arr[hi];
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (arr[j] <= pivot) {
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
                i++;
            }
        }
        int temp = arr[i];
        arr[i] = arr[hi];
        arr[hi] = temp;

        quicksort(arr, lo, i - 1);
        quicksort(arr, i + 1, hi);
    }

    @Test
    public void testQuicksort() {
        CoreOp.FuncOp f = getFuncOp("quicksort");

        CoreOp.FuncOp lf = generate(f);

        int[] arr1 = new int[]{5, 2, 7, 45, 34, 14, 0, 27, 43, 11, 38, 56, 81};
        int[] arr2 = new int[]{2, 11, 45, 34, 0, 27, 38, 56, 7, 43, 14, 5, 81};

        Interpreter.invoke(MethodHandles.lookup(), lf, arr1, 0, arr1.length - 1);
        quicksort(arr2, 0, arr2.length - 1);
        Assert.assertEquals(arr1, arr2);
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
