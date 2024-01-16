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
 * @summary Smoke test for code reflection with local variables.
 * @build LocalVarTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester LocalVarTest
 */

public class LocalVarTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"1";
                %2 : Var<int> = var %1 @"x";
                %3 : int = constant @"2";
                %4 : Var<int> = var %3 @"y";
                %5 : int = var.load %2;
                %6 : int = var.load %4;
                %7 : int = add %5 %6;
                return %7;
            };
            """)
    int test1() {
        int x = 1;
        int y = 2;
        return x + y;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : LocalVarTest, %1 : int, %2 : int)int -> {
                %3 : Var<int> = var %1 @"x";
                %4 : Var<int> = var %2 @"y";
                %5 : int = var.load %3;
                %6 : int = var.load %4;
                %7 : int = add %5 %6;
                return %7;
            };
            """)
    int test2(int x, int y) {
        return x + y;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"x";
                %3 : int = constant @"0";
                %4 : Var<int> = var %3 @"y";
                %5 : int = constant @"1";
                var.store %2 %5;
                %6 : int = constant @"2";
                var.store %4 %6;
                %7 : int = var.load %2;
                %8 : int = var.load %4;
                %9 : int = add %7 %8;
                return %9;
            };
            """)
    int test3() {
        int x;
        int y;
        x = 1;
        y = 2;
        return x + y;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"1";
                %2 : Var<int> = var %1 @"x";
                %3 : int = var.load %2;
                %4 : int = constant @"1";
                %5 : int = add %3 %4;
                %6 : Var<int> = var %5 @"y";
                %7 : int = var.load %6;
                return %7;
            };
            """)
    int test4() {
        int x = 1;
        int y = x + 1;
        return y;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"1";
                %2 : Var<int> = var %1 @"x";
                %3 : int = var.load %2;
                %4 : Var<int> = var %3 @"y";
                %5 : int = var.load %4;
                return %5;
            };
            """)
    int test5() {
        int x = 1;
        int y = x;
        return y;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"1";
                %2 : Var<int> = var %1 @"x";
                %3 : int = constant @"1";
                %4 : Var<int> = var %3 @"y";
                %5 : int = constant @"1";
                %6 : Var<int> = var %5 @"z";
                %7 : int = var.load %2;
                var.store %4 %7;
                var.store %6 %7;
                %8 : int = var.load %6;
                return %8;
            };
            """)
    int test6() {
        int x = 1;
        int y = 1;
        int z = 1;
        z = y = x;
        return z;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : LocalVarTest)int -> {
                %1 : int = constant @"1";
                %2 : Var<int> = var %1 @"x";
                %3 : int = var.load %2;
                %4 : int = constant @"2";
                %5 : int = add %3 %4;
                var.store %2 %5;
                %6 : Var<int> = var %5 @"y";
                %7 : int = var.load %6;
                %8 : int = constant @"3";
                %9 : int = add %7 %8;
                var.store %6 %9;
                %10 : int = var.load %2;
                %11 : int = constant @"4";
                %12 : int = add %10 %11;
                var.store %2 %12;
                %13 : int = add %9 %12;
                return %13;
            };
            """)
    int test7() {
        int x = 1;
        int y = x += 2;
        return (y += 3) + (x += 4);
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : LocalVarTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = constant @"1";
                %5 : int = add %3 %4;
                var.store %2 %5;
                %6 : Var<int> = var %3 @"x";
                %7 : int = var.load %2;
                %8 : int = constant @"1";
                %9 : int = sub %7 %8;
                var.store %2 %9;
                %10 : Var<int> = var %7 @"y";
                return;
            };
            """)
    void test8(int i) {
        int x = i++;
        int y = i--;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : LocalVarTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = constant @"1";
                %5 : int = add %3 %4;
                var.store %2 %5;
                %6 : Var<int> = var %5 @"x";
                %7 : int = var.load %2;
                %8 : int = constant @"1";
                %9 : int = sub %7 %8;
                var.store %2 %9;
                %10 : Var<int> = var %9 @"y";
                return;
            };
            """)
    void test9(int i) {
        int x = ++i;
        int y = --i;
    }
}
