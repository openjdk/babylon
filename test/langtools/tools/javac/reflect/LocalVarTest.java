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

import jdk.incubator.code.CodeReflection;

/*
 * @test
 * @summary Smoke test for code reflection with local variables.
 * @modules jdk.incubator.code
 * @build LocalVarTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester LocalVarTest
 */

public class LocalVarTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : Var<java.type:"int"> = var %1 @"x";
                %3 : java.type:"int" = constant @2;
                %4 : Var<java.type:"int"> = var %3 @"y";
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = add %5 %6;
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
            func @"test2" (%0 : java.type:"LocalVarTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"int" -> {
                %3 : Var<java.type:"int"> = var %1 @"x";
                %4 : Var<java.type:"int"> = var %2 @"y";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"int" = add %5 %6;
                return %7;
            };
            """)
    int test2(int x, int y) {
        return x + y;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var @"x";
                %2 : Var<java.type:"int"> = var @"y";
                %3 : java.type:"int" = constant @1;
                var.store %1 %3;
                %4 : java.type:"int" = constant @2;
                var.store %2 %4;
                %5 : java.type:"int" = var.load %1;
                %6 : java.type:"int" = var.load %2;
                %7 : java.type:"int" = add %5 %6;
                return %7;
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
            func @"test4" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : Var<java.type:"int"> = var %1 @"x";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"int" = add %3 %4;
                %6 : Var<java.type:"int"> = var %5 @"y";
                %7 : java.type:"int" = var.load %6;
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
            func @"test5" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : Var<java.type:"int"> = var %1 @"x";
                %3 : java.type:"int" = var.load %2;
                %4 : Var<java.type:"int"> = var %3 @"y";
                %5 : java.type:"int" = var.load %4;
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
            func @"test6" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : Var<java.type:"int"> = var %1 @"x";
                %3 : java.type:"int" = constant @1;
                %4 : Var<java.type:"int"> = var %3 @"y";
                %5 : java.type:"int" = constant @1;
                %6 : Var<java.type:"int"> = var %5 @"z";
                %7 : java.type:"int" = var.load %2;
                var.store %4 %7;
                var.store %6 %7;
                %8 : java.type:"int" = var.load %6;
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
            func @"test7" (%0 : java.type:"LocalVarTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : Var<java.type:"int"> = var %1 @"x";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @2;
                %5 : java.type:"int" = add %3 %4;
                var.store %2 %5;
                %6 : Var<java.type:"int"> = var %5 @"y";
                %7 : java.type:"int" = var.load %6;
                %8 : java.type:"int" = constant @3;
                %9 : java.type:"int" = add %7 %8;
                var.store %6 %9;
                %10 : java.type:"int" = var.load %2;
                %11 : java.type:"int" = constant @4;
                %12 : java.type:"int" = add %10 %11;
                var.store %2 %12;
                %13 : java.type:"int" = add %9 %12;
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
            func @"test8" (%0 : java.type:"LocalVarTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"int" = add %3 %4;
                var.store %2 %5;
                %6 : Var<java.type:"int"> = var %3 @"x";
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"int" = constant @1;
                %9 : java.type:"int" = sub %7 %8;
                var.store %2 %9;
                %10 : Var<java.type:"int"> = var %7 @"y";
                return;
            };
            """)
    void test8(int i) {
        int x = i++;
        int y = i--;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"LocalVarTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"int" = add %3 %4;
                var.store %2 %5;
                %6 : Var<java.type:"int"> = var %5 @"x";
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"int" = constant @1;
                %9 : java.type:"int" = sub %7 %8;
                var.store %2 %9;
                %10 : Var<java.type:"int"> = var %9 @"y";
                return;
            };
            """)
    void test9(int i) {
        int x = ++i;
        int y = --i;
    }
}
