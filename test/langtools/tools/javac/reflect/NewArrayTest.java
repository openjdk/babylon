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
 * @summary Smoke test for code reflection with constant values.
 * @build NewArrayTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester NewArrayTest
 */

import java.lang.runtime.CodeReflection;
import java.util.function.Function;

public class NewArrayTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"10";
                %2 : int[] = new %1 @"(int)int[]";
                %3 : Var<int[]> = var %2 @"a";
                return;
            };
            """)
    void test1() {
        int[] a = new int[10];
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : NewArrayTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"l";
                %3 : int = var.load %2;
                %4 : int = constant @"10";
                %5 : int = add %3 %4;
                %6 : int[] = new %5 @"(int)int[]";
                %7 : Var<int[]> = var %6 @"a";
                return;
            };
            """)
    void test2(int l) {
        int[] a = new int[l + 10];
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"10";
                %2 : java.lang.String[] = new %1 @"(int)java.lang.String[]";
                %3 : Var<java.lang.String[]> = var %2 @"a";
                return;
            };
            """)
    void test3() {
        String[] a = new String[10];
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"10";
                %2 : java.lang.String[][] = new %1 @"(int)java.lang.String[][]";
                %3 : Var<java.lang.String[][]> = var %2 @"a";
                return;
            };
            """)
    void test4() {
        String[][] a = new String[10][];
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"10";
                %2 : int = constant @"10";
                %3 : java.lang.String[][] = new %1 %2 @"(int, int)java.lang.String[][]";
                %4 : Var<java.lang.String[][]> = var %3 @"a";
                return;
            };
            """)
    void test5() {
        String[][] a = new String[10][10];
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"3";
                %2 : java.lang.String[][] = new %1 @"(int)java.lang.String[][]";
                %3 : int = constant @"2";
                %4 : java.lang.String[] = new %3 @"(int)java.lang.String[]";
                %5 : java.lang.String = constant @"one";
                %6 : int = constant @"0";
                array.store %4 %6 %5;
                %7 : java.lang.String = constant @"two";
                %8 : int = constant @"1";
                array.store %4 %8 %7;
                %9 : int = constant @"0";
                array.store %2 %9 %4;
                %10 : int = constant @"1";
                %11 : java.lang.String[] = new %10 @"(int)java.lang.String[]";
                %12 : java.lang.String = constant @"three";
                %13 : int = constant @"0";
                array.store %11 %13 %12;
                %14 : int = constant @"1";
                array.store %2 %14 %11;
                %15 : java.lang.String[] = constant @null;
                %16 : int = constant @"2";
                array.store %2 %16 %15;
                %17 : Var<java.lang.String[][]> = var %2 @"a";
                return;
            };
            """)
    void test6() {
        String[][] a = { { "one", "two" }, { "three" }, null };
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : NewArrayTest)void -> {
                %1 : int = constant @"3";
                %2 : java.lang.String[][] = new %1 @"(int)java.lang.String[][]";
                %3 : int = constant @"2";
                %4 : java.lang.String[] = new %3 @"(int)java.lang.String[]";
                %5 : java.lang.String = constant @"one";
                %6 : int = constant @"0";
                array.store %4 %6 %5;
                %7 : java.lang.String = constant @"two";
                %8 : int = constant @"1";
                array.store %4 %8 %7;
                %9 : int = constant @"0";
                array.store %2 %9 %4;
                %10 : int = constant @"1";
                %11 : java.lang.String[] = new %10 @"(int)java.lang.String[]";
                %12 : java.lang.String = constant @"three";
                %13 : int = constant @"0";
                array.store %11 %13 %12;
                %14 : int = constant @"1";
                array.store %2 %14 %11;
                %15 : java.lang.String[] = constant @null;
                %16 : int = constant @"2";
                array.store %2 %16 %15;
                %17 : Var<java.lang.String[][]> = var %2 @"a";
                return;
            };
            """)
    void test7() {
        String[][] a = new String[][] { { "one", "two" }, { "three" }, null };
    }
}
