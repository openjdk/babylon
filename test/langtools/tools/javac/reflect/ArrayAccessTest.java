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

import java.lang.runtime.CodeReflection;

/*
 * @test
 * @summary Smoke test for code reflection with array access.
 * @build ArrayAccessTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ArrayAccessTest
 */

public class ArrayAccessTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : ArrayAccessTest, %1 : int[])int -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = constant @"0";
                %5 : int = array.load %3 %4;
                return %5;
            };
            """)
    int test1(int[] ia) {
        return ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : ArrayAccessTest, %1 : int[], %2 : int)int -> {
                %3 : Var<int[]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[] = var.load %3;
                %6 : int = var.load %4;
                %7 : int = constant @"1";
                %8 : int = add %6 %7;
                %9 : int = array.load %5 %8;
                return %9;
            };
            """)
    int test2(int[] ia, int i) {
        return ia[i + 1];
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : ArrayAccessTest, %1 : int[])void -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = constant @"0";
                %5 : int = constant @"1";
                array.store %3 %4 %5;
                return;
            };
            """)
    void test3(int[] ia) {
        ia[0] = 1;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : ArrayAccessTest, %1 : int[], %2 : int)void -> {
                %3 : Var<int[]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[] = var.load %3;
                %6 : int = var.load %4;
                %7 : int = constant @"1";
                %8 : int = add %6 %7;
                %9 : int = constant @"1";
                array.store %5 %8 %9;
                return;
            };
            """)
    void test4(int[] ia, int i) {
        ia[i + 1] = 1;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : ArrayAccessTest, %1 : int[][], %2 : int)int -> {
                %3 : Var<int[][]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[][] = var.load %3;
                %6 : int = var.load %4;
                %7 : int = constant @"1";
                %8 : int = add %6 %7;
                %9 : int[] = array.load %5 %8;
                %10 : int = var.load %4;
                %11 : int = constant @"2";
                %12 : int = add %10 %11;
                %13 : int = array.load %9 %12;
                return %13;
            };
            """)
    int test5(int[][] ia, int i) {
        return ia[i + 1][i + 2];
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : ArrayAccessTest, %1 : int[][], %2 : int)void -> {
                %3 : Var<int[][]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[][] = var.load %3;
                %6 : int = var.load %4;
                %7 : int = constant @"1";
                %8 : int = add %6 %7;
                %9 : int[] = array.load %5 %8;
                %10 : int = var.load %4;
                %11 : int = constant @"2";
                %12 : int = add %10 %11;
                %13 : int = constant @"1";
                array.store %9 %12 %13;
                return;
            };
            """)
    void test6(int[][] ia, int i) {
        ia[i + 1][i + 2] = 1;
    }

    int[] ia;

    @CodeReflection
    @IR("""
            func @"test7" (%0 : ArrayAccessTest)int -> {
                %1 : int[] = field.load %0 @"ArrayAccessTest::ia()int[]";
                %2 : int = constant @"0";
                %3 : int = array.load %1 %2;
                return %3;
            };
            """)
    int test7() {
        return ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : ArrayAccessTest)int -> {
                %1 : int[] = field.load %0 @"ArrayAccessTest::ia()int[]";
                %2 : int = constant @"0";
                %3 : int = array.load %1 %2;
                return %3;
            };
            """)
    int test8() {
        return this.ia[0];
    }

    static class A {
        int i;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : ArrayAccessTest, %1 : ArrayAccessTest$A[])int -> {
                %2 : Var<ArrayAccessTest$A[]> = var %1 @"aa";
                %3 : ArrayAccessTest$A[] = var.load %2;
                %4 : int = constant @"0";
                %5 : ArrayAccessTest$A = array.load %3 %4;
                %6 : int = field.load %5 @"ArrayAccessTest$A::i()int";
                return %6;
            };
            """)
    int test9(A[] aa) {
        return aa[0].i;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : ArrayAccessTest, %1 : ArrayAccessTest$A[])void -> {
                %2 : Var<ArrayAccessTest$A[]> = var %1 @"aa";
                %3 : ArrayAccessTest$A[] = var.load %2;
                %4 : int = constant @"0";
                %5 : ArrayAccessTest$A = array.load %3 %4;
                %6 : int = constant @"1";
                field.store %5 %6 @"ArrayAccessTest$A::i()int";
                return;
            };
            """)
    void test10(A[] aa) {
        aa[0].i = 1;
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : ArrayAccessTest, %1 : int[])void -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = constant @"0";
                %5 : int = array.load %3 %4;
                %6 : int = constant @"1";
                %7 : int = add %5 %6;
                array.store %3 %4 %7;
                return;
            };
            """)
    void test11(int[] ia) {
        ia[0] += 1;
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : ArrayAccessTest, %1 : int[], %2 : int)void -> {
                %3 : Var<int[]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[] = var.load %3;
                %6 : int = constant @"1";
                %7 : int[] = var.load %3;
                %8 : int = var.load %4;
                %9 : int = constant @"2";
                %10 : int = add %8 %9;
                %11 : int = array.load %7 %10;
                %12 : int = constant @"1";
                %13 : int = add %11 %12;
                array.store %7 %10 %13;
                array.store %5 %6 %13;
                return;
            };
            """)
    void test12(int[] ia, int i) {
        ia[1] = ia[i + 2] += 1;
    }

    @CodeReflection
    @IR("""
            func @"test13" (%0 : ArrayAccessTest, %1 : int[], %2 : int)void -> {
                %3 : Var<int[]> = var %1 @"ia";
                %4 : Var<int> = var %2 @"i";
                %5 : int[] = var.load %3;
                %6 : int = constant @"1";
                %7 : int = array.load %5 %6;
                %8 : int[] = var.load %3;
                %9 : int = var.load %4;
                %10 : int = constant @"2";
                %11 : int = add %9 %10;
                %12 : int = array.load %8 %11;
                %13 : int = constant @"1";
                %14 : int = add %12 %13;
                array.store %8 %11 %14;
                %15 : int = add %7 %14;
                array.store %5 %6 %15;
                return;
            };
            """)
    void test13(int[] ia, int i) {
        ia[1] += ia[i + 2] += 1;
    }


    @CodeReflection
    @IR("""
            func @"test14" (%0 : ArrayAccessTest, %1 : int[])void -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = constant @"0";
                %5 : int = array.load %3 %4;
                %6 : int = constant @"1";
                %7 : int = add %5 %6;
                array.store %3 %4 %7;
                %8 : Var<int> = var %5 @"x";
                %9 : int[] = var.load %2;
                %10 : int = constant @"0";
                %11 : int = array.load %9 %10;
                %12 : int = constant @"1";
                %13 : int = sub %11 %12;
                array.store %9 %10 %13;
                %14 : Var<int> = var %11 @"y";
                return;
            };
            """)
    void test14(int[] ia) {
        int x = ia[0]++;
        int y = ia[0]--;
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : ArrayAccessTest, %1 : int[])void -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = constant @"0";
                %5 : int = array.load %3 %4;
                %6 : int = constant @"1";
                %7 : int = add %5 %6;
                array.store %3 %4 %7;
                %8 : Var<int> = var %7 @"x";
                %9 : int[] = var.load %2;
                %10 : int = constant @"0";
                %11 : int = array.load %9 %10;
                %12 : int = constant @"1";
                %13 : int = sub %11 %12;
                array.store %9 %10 %13;
                %14 : Var<int> = var %13 @"y";
                return;
            };
            """)
    void test15(int[] ia) {
        int x = ++ia[0];
        int y = --ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test16" (%0 : ArrayAccessTest, %1 : int[])int -> {
                %2 : Var<int[]> = var %1 @"ia";
                %3 : int[] = var.load %2;
                %4 : int = array.length %3;
                %5 : int[] = var.load %2;
                %6 : int = invoke %5 @"java.lang.Object::hashCode()int";
                %7 : int = add %4 %6;
                return %7;
            };
            """)
    int test16(int[] ia) {
        return ia.length + ia.hashCode();
    }

}
