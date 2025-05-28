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
 * @summary Smoke test for code reflection with unary operations.
 * @modules jdk.incubator.code
 * @build UnaryopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester UnaryopTest
 */

import jdk.incubator.code.CodeReflection;

public class UnaryopTest {
    @CodeReflection
    @IR("""
            func @"test" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"v";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = neg %2;
                return %3;
            };
            """)
    static int test(int v) {
        return -v;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"v";
                %2 : java.type:"int" = var.load %1;
                return %2;
            };
            """)
    static int test2(int v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"int")java.type:"java.lang.Integer" -> {
                %1 : Var<java.type:"int"> = var %0 @"v";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"java.lang.Integer" = invoke %2 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                return %3;
            };
            """)
    // Tests that numeric promotion occurs
    static Integer test3(int v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.Integer" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"v";
                %2 : java.type:"java.lang.Integer" = var.load %1;
                %3 : java.type:"int" = invoke %2 @java.ref:"java.lang.Integer::intValue():int";
                %4 : java.type:"java.lang.Integer" = invoke %3 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                return %4;
            };
            """)
    // Tests that numeric promotion is retained
    static Integer test4(Integer v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"v";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = compl %2;
                return %3;
            };
            """)
    static int test5(int v) {
        return ~v;
    }

    @IR("""
            func @"test6" (%0 : java.type:"byte")java.type:"void" -> {
                %1 : Var<java.type:"byte"> = var %0 @"b";
                %2 : java.type:"byte" = var.load %1;
                %3 : java.type:"int" = constant @"1";
                %4 : java.type:"byte" = conv %3;
                %5 : java.type:"byte" = add %2 %4;
                var.store %1 %5;
                %6 : java.type:"byte" = var.load %1;
                %7 : java.type:"int" = constant @"1";
                %8 : java.type:"byte" = conv %7;
                %9 : java.type:"byte" = sub %6 %8;
                var.store %1 %9;
                %10 : java.type:"byte" = var.load %1;
                %11 : java.type:"int" = constant @"1";
                %12 : java.type:"byte" = conv %11;
                %13 : java.type:"byte" = add %10 %12;
                var.store %1 %13;
                %14 : java.type:"byte" = var.load %1;
                %15 : java.type:"int" = constant @"1";
                %16 : java.type:"byte" = conv %15;
                %17 : java.type:"byte" = sub %14 %16;
                var.store %1 %17;
                return;
            };
            """)
    @CodeReflection
    static void test6(byte b) {
        b++;
        b--;
        ++b;
        --b;
    }

    @IR("""
            func @"test7" (%0 : java.type:"short")java.type:"void" -> {
                %1 : Var<java.type:"short"> = var %0 @"s";
                %2 : java.type:"short" = var.load %1;
                %3 : java.type:"int" = constant @"1";
                %4 : java.type:"short" = conv %3;
                %5 : java.type:"short" = add %2 %4;
                var.store %1 %5;
                %6 : java.type:"short" = var.load %1;
                %7 : java.type:"int" = constant @"1";
                %8 : java.type:"short" = conv %7;
                %9 : java.type:"short" = sub %6 %8;
                var.store %1 %9;
                %10 : java.type:"short" = var.load %1;
                %11 : java.type:"int" = constant @"1";
                %12 : java.type:"short" = conv %11;
                %13 : java.type:"short" = add %10 %12;
                var.store %1 %13;
                %14 : java.type:"short" = var.load %1;
                %15 : java.type:"int" = constant @"1";
                %16 : java.type:"short" = conv %15;
                %17 : java.type:"short" = sub %14 %16;
                var.store %1 %17;
                return;
            };
            """)
    @CodeReflection
    static void test7(short s) {
        s++;
        s--;
        ++s;
        --s;
    }
}
