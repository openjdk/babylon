/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.CodeReflection;

/*
 * @test
 * @summary Smoke test for code reflection with array access.
 * @modules jdk.incubator.code
 * @build ArrayAccessTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ArrayAccessTest
 */

public class ArrayAccessTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"int" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"int" = array.load %3 %4;
                return %5;
            };
            """)
    int test1(int[] ia) {
        return ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]", %2 : java.type:"int")java.type:"int" -> {
                %3 : Var<java.type:"int[]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[]" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = add %6 %7;
                %9 : java.type:"int" = array.load %5 %8;
                return %9;
            };
            """)
    int test2(int[] ia, int i) {
        return ia[i + 1];
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"void" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"int" = constant @1;
                array.store %3 %4 %5;
                return;
            };
            """)
    void test3(int[] ia) {
        ia[0] = 1;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int[]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[]" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = add %6 %7;
                %9 : java.type:"int" = constant @1;
                array.store %5 %8 %9;
                return;
            };
            """)
    void test4(int[] ia, int i) {
        ia[i + 1] = 1;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[][]", %2 : java.type:"int")java.type:"int" -> {
                %3 : Var<java.type:"int[][]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[][]" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = add %6 %7;
                %9 : java.type:"int[]" = array.load %5 %8;
                %10 : java.type:"int" = var.load %4;
                %11 : java.type:"int" = constant @2;
                %12 : java.type:"int" = add %10 %11;
                %13 : java.type:"int" = array.load %9 %12;
                return %13;
            };
            """)
    int test5(int[][] ia, int i) {
        return ia[i + 1][i + 2];
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[][]", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int[][]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[][]" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = add %6 %7;
                %9 : java.type:"int[]" = array.load %5 %8;
                %10 : java.type:"int" = var.load %4;
                %11 : java.type:"int" = constant @2;
                %12 : java.type:"int" = add %10 %11;
                %13 : java.type:"int" = constant @1;
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
            func @"test7" (%0 : java.type:"ArrayAccessTest")java.type:"int" -> {
                %1 : java.type:"int[]" = field.load %0 @java.ref:"ArrayAccessTest::ia:int[]";
                %2 : java.type:"int" = constant @0;
                %3 : java.type:"int" = array.load %1 %2;
                return %3;
            };
            """)
    int test7() {
        return ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"ArrayAccessTest")java.type:"int" -> {
                %1 : java.type:"int[]" = field.load %0 @java.ref:"ArrayAccessTest::ia:int[]";
                %2 : java.type:"int" = constant @0;
                %3 : java.type:"int" = array.load %1 %2;
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
            func @"test9" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"ArrayAccessTest$A[]")java.type:"int" -> {
                %2 : Var<java.type:"ArrayAccessTest$A[]"> = var %1 @"aa";
                %3 : java.type:"ArrayAccessTest$A[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"ArrayAccessTest$A" = array.load %3 %4;
                %6 : java.type:"int" = field.load %5 @java.ref:"ArrayAccessTest$A::i:int";
                return %6;
            };
            """)
    int test9(A[] aa) {
        return aa[0].i;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"ArrayAccessTest$A[]")java.type:"void" -> {
                %2 : Var<java.type:"ArrayAccessTest$A[]"> = var %1 @"aa";
                %3 : java.type:"ArrayAccessTest$A[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"ArrayAccessTest$A" = array.load %3 %4;
                %6 : java.type:"int" = constant @1;
                field.store %5 %6 @java.ref:"ArrayAccessTest$A::i:int";
                return;
            };
            """)
    void test10(A[] aa) {
        aa[0].i = 1;
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"void" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"int" = array.load %3 %4;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"int" = add %5 %6;
                array.store %3 %4 %7;
                return;
            };
            """)
    void test11(int[] ia) {
        ia[0] += 1;
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int[]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[]" = var.load %3;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"int[]" = var.load %3;
                %8 : java.type:"int" = var.load %4;
                %9 : java.type:"int" = constant @2;
                %10 : java.type:"int" = add %8 %9;
                %11 : java.type:"int" = array.load %7 %10;
                %12 : java.type:"int" = constant @1;
                %13 : java.type:"int" = add %11 %12;
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
            func @"test13" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int[]"> = var %1 @"ia";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int[]" = var.load %3;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"int" = array.load %5 %6;
                %8 : java.type:"int[]" = var.load %3;
                %9 : java.type:"int" = var.load %4;
                %10 : java.type:"int" = constant @2;
                %11 : java.type:"int" = add %9 %10;
                %12 : java.type:"int" = array.load %8 %11;
                %13 : java.type:"int" = constant @1;
                %14 : java.type:"int" = add %12 %13;
                array.store %8 %11 %14;
                %15 : java.type:"int" = add %7 %14;
                array.store %5 %6 %15;
                return;
            };
            """)
    void test13(int[] ia, int i) {
        ia[1] += ia[i + 2] += 1;
    }


    @CodeReflection
    @IR("""
            func @"test14" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"void" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"int" = array.load %3 %4;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"int" = add %5 %6;
                array.store %3 %4 %7;
                %8 : Var<java.type:"int"> = var %5 @"x";
                %9 : java.type:"int[]" = var.load %2;
                %10 : java.type:"int" = constant @0;
                %11 : java.type:"int" = array.load %9 %10;
                %12 : java.type:"int" = constant @1;
                %13 : java.type:"int" = sub %11 %12;
                array.store %9 %10 %13;
                %14 : Var<java.type:"int"> = var %11 @"y";
                return;
            };
            """)
    void test14(int[] ia) {
        int x = ia[0]++;
        int y = ia[0]--;
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"void" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"int" = array.load %3 %4;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"int" = add %5 %6;
                array.store %3 %4 %7;
                %8 : Var<java.type:"int"> = var %7 @"x";
                %9 : java.type:"int[]" = var.load %2;
                %10 : java.type:"int" = constant @0;
                %11 : java.type:"int" = array.load %9 %10;
                %12 : java.type:"int" = constant @1;
                %13 : java.type:"int" = sub %11 %12;
                array.store %9 %10 %13;
                %14 : Var<java.type:"int"> = var %13 @"y";
                return;
            };
            """)
    void test15(int[] ia) {
        int x = ++ia[0];
        int y = --ia[0];
    }

    @CodeReflection
    @IR("""
            func @"test16" (%0 : java.type:"ArrayAccessTest", %1 : java.type:"int[]")java.type:"int" -> {
                %2 : Var<java.type:"int[]"> = var %1 @"ia";
                %3 : java.type:"int[]" = var.load %2;
                %4 : java.type:"int" = array.length %3;
                %5 : java.type:"int[]" = var.load %2;
                %6 : java.type:"int" = invoke %5 @java.ref:"java.lang.Object::hashCode():int";
                %7 : java.type:"int" = add %4 %6;
                return %7;
            };
            """)
    int test16(int[] ia) {
        return ia.length + ia.hashCode();
    }

    @CodeReflection
    @IR("""
            func @"test17" (%0 : java.type:"java.lang.Object[]")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.Object[]"> = var %0 @"a";
                %2 : java.type:"java.lang.Object[]" = var.load %1;
                %3 : java.type:"char" = constant @'c';
                %4 : java.type:"int" = conv %3;
                %5 : java.type:"java.lang.Object" = array.load %2 %4;
                return %5;
            };
            """)
    static Object test17(Object[] a) {
        return a['c'];
    }

}
