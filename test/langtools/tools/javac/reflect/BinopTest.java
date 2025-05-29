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
 * @summary Smoke test for code reflection with binary operations.
 * @modules jdk.incubator.code
 * @build BinopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BinopTest
 */

import jdk.incubator.code.CodeReflection;

public class BinopTest {

    @CodeReflection
    @IR("""
            func @"test" (%0 : java.type:"BinopTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @5;
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"int" = constant @4;
                %4 : java.type:"int" = mul %2 %3;
                %5 : java.type:"int" = add %1 %4;
                %6 : java.type:"int" = constant @3;
                %7 : java.type:"int" = sub %5 %6;
                return %7;
            };
            """)
    int test() {
        return 5 + 2 * 4 - 3;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"BinopTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"int" = constant @3;
                %4 : java.type:"int" = constant @4;
                %5 : java.type:"int" = add %3 %4;
                %6 : java.type:"int" = add %2 %5;
                %7 : java.type:"int" = add %1 %6;
                return %7;
            };
            """)
    int test2() {
        return 1 + (2 + (3 + 4));
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"BinopTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @1;
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"int" = add %1 %2;
                %4 : java.type:"int" = constant @3;
                %5 : java.type:"int" = add %3 %4;
                %6 : java.type:"int" = constant @4;
                %7 : java.type:"int" = add %5 %6;
                return %7;
            };
            """)
    int test3() {
        return ((1 + 2) + 3) + 4;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"BinopTest", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"int" = add %3 %4;
                var.store %2 %5;
                %6 : java.type:"int" = var.load %2;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"int" = mul %6 %7;
                var.store %2 %8;
                %9 : java.type:"int" = add %5 %8;
                %10 : java.type:"int" = var.load %2;
                %11 : java.type:"int" = constant @1;
                %12 : java.type:"int" = div %10 %11;
                var.store %2 %12;
                %13 : java.type:"int" = add %9 %12;
                %14 : java.type:"int" = var.load %2;
                %15 : java.type:"int" = constant @1;
                %16 : java.type:"int" = sub %14 %15;
                var.store %2 %16;
                %17 : java.type:"int" = add %13 %16;
                %18 : java.type:"int" = var.load %2;
                %19 : java.type:"int" = constant @1;
                %20 : java.type:"int" = mod %18 %19;
                var.store %2 %20;
                %21 : java.type:"int" = add %17 %20;
                return %21;
            };
            """)
    int test4(int i) {
        return (i += 1) + (i *= 1) + (i /= 1) + (i -= 1) + (i %= 1);
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"BinopTest", %1 : java.type:"int")java.type:"boolean" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = constant @0;
                %5 : java.type:"boolean" = eq %3 %4;
                %6 : java.type:"boolean" = not %5;
                return %6;
            };
            """)
    boolean test5(int i) {
        return !(i == 0);
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"BinopTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @5;
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"int" = mod %1 %2;
                return %3;
            };
            """)
    int test6() {
        return 5 % 2;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"BinopTest", %1 : java.type:"double")java.type:"void" -> {
                %2 : Var<java.type:"double"> = var %1 @"d";
                %3 : java.type:"double" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"double" = conv %4;
                %6 : java.type:"double" = add %3 %5;
                var.store %2 %6;
                %7 : java.type:"long" = constant @1;
                %8 : java.type:"double" = conv %7;
                %9 : java.type:"double" = var.load %2;
                %10 : java.type:"double" = add %8 %9;
                var.store %2 %10;
                %11 : java.type:"double" = var.load %2;
                %12 : java.type:"long" = constant @1;
                %13 : java.type:"double" = conv %12;
                %14 : java.type:"double" = sub %11 %13;
                var.store %2 %14;
                %15 : java.type:"int" = constant @1;
                %16 : java.type:"double" = conv %15;
                %17 : java.type:"double" = var.load %2;
                %18 : java.type:"double" = sub %16 %17;
                var.store %2 %18;
                %19 : java.type:"double" = var.load %2;
                %20 : java.type:"int" = constant @1;
                %21 : java.type:"double" = conv %20;
                %22 : java.type:"double" = mul %19 %21;
                var.store %2 %22;
                %23 : java.type:"long" = constant @1;
                %24 : java.type:"double" = conv %23;
                %25 : java.type:"double" = var.load %2;
                %26 : java.type:"double" = mul %24 %25;
                var.store %2 %26;
                %27 : java.type:"double" = var.load %2;
                %28 : java.type:"long" = constant @1;
                %29 : java.type:"double" = conv %28;
                %30 : java.type:"double" = div %27 %29;
                var.store %2 %30;
                %31 : java.type:"int" = constant @1;
                %32 : java.type:"double" = conv %31;
                %33 : java.type:"double" = var.load %2;
                %34 : java.type:"double" = div %32 %33;
                var.store %2 %34;
                %35 : java.type:"double" = var.load %2;
                %36 : java.type:"int" = constant @1;
                %37 : java.type:"double" = conv %36;
                %38 : java.type:"double" = mod %35 %37;
                var.store %2 %38;
                %39 : java.type:"long" = constant @1;
                %40 : java.type:"double" = conv %39;
                %41 : java.type:"double" = var.load %2;
                %42 : java.type:"double" = mod %40 %41;
                var.store %2 %42;
                %43 : java.type:"int" = constant @-1;
                %44 : java.type:"double" = conv %43;
                var.store %2 %44;
                return;
            };
            """)
    void test7(double d) {
        d = d + 1;
        d = 1L + d;

        d = d - 1L;
        d = 1 - d;

        d = d * 1;
        d = 1L * d;

        d = d / 1L;
        d = 1 / d;

        d = d % 1;
        d = 1L % d;

        d = -1;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"BinopTest", %1 : java.type:"double")java.type:"void" -> {
                %2 : Var<java.type:"double"> = var %1 @"d";
                %3 : java.type:"double" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"double" = conv %4;
                %6 : java.type:"double" = add %3 %5;
                var.store %2 %6;
                %7 : java.type:"double" = var.load %2;
                %8 : java.type:"long" = constant @1;
                %9 : java.type:"double" = conv %8;
                %10 : java.type:"double" = sub %7 %9;
                var.store %2 %10;
                %11 : java.type:"double" = var.load %2;
                %12 : java.type:"int" = constant @1;
                %13 : java.type:"double" = conv %12;
                %14 : java.type:"double" = mul %11 %13;
                var.store %2 %14;
                %15 : java.type:"double" = var.load %2;
                %16 : java.type:"long" = constant @1;
                %17 : java.type:"double" = conv %16;
                %18 : java.type:"double" = div %15 %17;
                var.store %2 %18;
                %19 : java.type:"double" = var.load %2;
                %20 : java.type:"int" = constant @1;
                %21 : java.type:"double" = conv %20;
                %22 : java.type:"double" = mod %19 %21;
                var.store %2 %22;
                return;
            };
            """)
    void test8(double d) {
        d += 1;

        d -= 1L;

        d *= 1;

        d /= 1L;

        d %= 1;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"BinopTest", %1 : java.type:"byte", %2 : java.type:"byte", %3 : java.type:"short")java.type:"void" -> {
                %4 : Var<java.type:"byte"> = var %1 @"a";
                %5 : Var<java.type:"byte"> = var %2 @"b";
                %6 : Var<java.type:"short"> = var %3 @"s";
                %7 : java.type:"byte" = var.load %4;
                %8 : java.type:"byte" = var.load %5;
                %9 : java.type:"byte" = add %7 %8;
                var.store %4 %9;
                %10 : java.type:"byte" = var.load %4;
                %11 : java.type:"short" = var.load %6;
                %12 : java.type:"byte" = conv %11;
                %13 : java.type:"byte" = div %10 %12;
                var.store %4 %13;
                %14 : java.type:"byte" = var.load %4;
                %15 : java.type:"double" = constant @3.5d;
                %16 : java.type:"byte" = conv %15;
                %17 : java.type:"byte" = mul %14 %16;
                var.store %4 %17;
                %18 : java.type:"byte" = var.load %4;
                %19 : java.type:"byte" = var.load %5;
                %20 : java.type:"byte" = lshl %18 %19;
                var.store %4 %20;
                %21 : java.type:"byte" = var.load %4;
                %22 : java.type:"int" = constant @1;
                %23 : java.type:"byte" = conv %22;
                %24 : java.type:"byte" = ashr %21 %23;
                var.store %4 %24;
                %25 : java.type:"byte" = var.load %4;
                %26 : java.type:"long" = constant @1;
                %27 : java.type:"byte" = conv %26;
                %28 : java.type:"byte" = ashr %25 %27;
                var.store %4 %28;
                return;
            };
            """)
    void test9(byte a, byte b, short s) {
        a += b;

        a /= s;

        a *= 3.5d;

        a <<= b;

        a >>= 1;

        a >>= 1L;
    }
}
