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
 * @build BinopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BinopTest
 */

import java.lang.runtime.CodeReflection;

public class BinopTest {

    @CodeReflection
    @IR("""
            func @"test" (%0 : BinopTest)int -> {
                %1 : int = constant @"5";
                %2 : int = constant @"2";
                %3 : int = constant @"4";
                %4 : int = mul %2 %3;
                %5 : int = add %1 %4;
                %6 : int = constant @"3";
                %7 : int = sub %5 %6;
                return %7;
            };
            """)
    int test() {
        return 5 + 2 * 4 - 3;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : BinopTest)int -> {
                %1 : int = constant @"1";
                %2 : int = constant @"2";
                %3 : int = constant @"3";
                %4 : int = constant @"4";
                %5 : int = add %3 %4;
                %6 : int = add %2 %5;
                %7 : int = add %1 %6;
                return %7;
            };
            """)
    int test2() {
        return 1 + (2 + (3 + 4));
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : BinopTest)int -> {
                %1 : int = constant @"1";
                %2 : int = constant @"2";
                %3 : int = add %1 %2;
                %4 : int = constant @"3";
                %5 : int = add %3 %4;
                %6 : int = constant @"4";
                %7 : int = add %5 %6;
                return %7;
            };
            """)
    int test3() {
        return ((1 + 2) + 3) + 4;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : BinopTest, %1 : int)int -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = constant @"1";
                %5 : int = add %3 %4;
                var.store %2 %5;
                %6 : int = var.load %2;
                %7 : int = constant @"1";
                %8 : int = mul %6 %7;
                var.store %2 %8;
                %9 : int = add %5 %8;
                %10 : int = var.load %2;
                %11 : int = constant @"1";
                %12 : int = div %10 %11;
                var.store %2 %12;
                %13 : int = add %9 %12;
                %14 : int = var.load %2;
                %15 : int = constant @"1";
                %16 : int = sub %14 %15;
                var.store %2 %16;
                %17 : int = add %13 %16;
                %18 : int = var.load %2;
                %19 : int = constant @"1";
                %20 : int = mod %18 %19;
                var.store %2 %20;
                %21 : int = add %17 %20;
                return %21;
            };
            """)
    int test4(int i) {
        return (i += 1) + (i *= 1) + (i /= 1) + (i -= 1) + (i %= 1);
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : BinopTest, %1 : int)boolean -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = constant @"0";
                %5 : boolean = eq %3 %4;
                %6 : boolean = not %5;
                return %6;
            };
            """)
    boolean test5(int i) {
        return !(i == 0);
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : BinopTest)int -> {
                %1 : int = constant @"5";
                %2 : int = constant @"2";
                %3 : int = mod %1 %2;
                return %3;
            };
            """)
    int test6() {
        return 5 % 2;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : BinopTest, %1 : double)void -> {
                %2 : Var<double> = var %1 @"d";
                %3 : double = var.load %2;
                %4 : int = constant @"1";
                %5 : double = conv %4;
                %6 : double = add %3 %5;
                var.store %2 %6;
                %7 : long = constant @"1";
                %8 : double = conv %7;
                %9 : double = var.load %2;
                %10 : double = add %8 %9;
                var.store %2 %10;
                %11 : double = var.load %2;
                %12 : long = constant @"1";
                %13 : double = conv %12;
                %14 : double = sub %11 %13;
                var.store %2 %14;
                %15 : int = constant @"1";
                %16 : double = conv %15;
                %17 : double = var.load %2;
                %18 : double = sub %16 %17;
                var.store %2 %18;
                %19 : double = var.load %2;
                %20 : int = constant @"1";
                %21 : double = conv %20;
                %22 : double = mul %19 %21;
                var.store %2 %22;
                %23 : long = constant @"1";
                %24 : double = conv %23;
                %25 : double = var.load %2;
                %26 : double = mul %24 %25;
                var.store %2 %26;
                %27 : double = var.load %2;
                %28 : long = constant @"1";
                %29 : double = conv %28;
                %30 : double = div %27 %29;
                var.store %2 %30;
                %31 : int = constant @"1";
                %32 : double = conv %31;
                %33 : double = var.load %2;
                %34 : double = div %32 %33;
                var.store %2 %34;
                %35 : double = var.load %2;
                %36 : int = constant @"1";
                %37 : double = conv %36;
                %38 : double = mod %35 %37;
                var.store %2 %38;
                %39 : long = constant @"1";
                %40 : double = conv %39;
                %41 : double = var.load %2;
                %42 : double = mod %40 %41;
                var.store %2 %42;
                %43 : int = constant @"-1";
                %44 : double = conv %43;
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
            func @"test8" (%0 : BinopTest, %1 : double)void -> {
                %2 : Var<double> = var %1 @"d";
                %3 : double = var.load %2;
                %4 : int = constant @"1";
                %5 : double = conv %4;
                %6 : double = add %3 %5;
                var.store %2 %6;
                %7 : double = var.load %2;
                %8 : long = constant @"1";
                %9 : double = conv %8;
                %10 : double = sub %7 %9;
                var.store %2 %10;
                %11 : double = var.load %2;
                %12 : int = constant @"1";
                %13 : double = conv %12;
                %14 : double = mul %11 %13;
                var.store %2 %14;
                %15 : double = var.load %2;
                %16 : long = constant @"1";
                %17 : double = conv %16;
                %18 : double = div %15 %17;
                var.store %2 %18;
                %19 : double = var.load %2;
                %20 : int = constant @"1";
                %21 : double = conv %20;
                %22 : double = mod %19 %21;
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
            TODO
            """)
    int test9(short s) {
        s <<= 1;
        s <<= 2L;
        return s << s;
    }

}
