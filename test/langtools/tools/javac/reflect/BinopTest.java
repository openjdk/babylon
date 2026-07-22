/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;

public class BinopTest {

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
    @IR("""
            func @"test9" (%0 : java.type:"BinopTest", %1 : java.type:"byte", %2 : java.type:"byte", %3 : java.type:"short")java.type:"void" -> {
                %4 : Var<java.type:"byte"> = var %1 @"a";
                %5 : Var<java.type:"byte"> = var %2 @"b";
                %6 : Var<java.type:"short"> = var %3 @"s";
                %7 : java.type:"byte" = var.load %4;
                %8 : java.type:"int" = conv %7;
                %9 : java.type:"byte" = var.load %5;
                %10 : java.type:"int" = conv %9;
                %11 : java.type:"int" = add %8 %10;
                %12 : java.type:"byte" = conv %11;
                var.store %4 %12;
                %13 : java.type:"byte" = var.load %4;
                %14 : java.type:"int" = conv %13;
                %15 : java.type:"short" = var.load %6;
                %16 : java.type:"int" = conv %15;
                %17 : java.type:"int" = div %14 %16;
                %18 : java.type:"byte" = conv %17;
                var.store %4 %18;
                %19 : java.type:"byte" = var.load %4;
                %20 : java.type:"double" = conv %19;
                %21 : java.type:"double" = constant @3.5d;
                %22 : java.type:"double" = mul %20 %21;
                %23 : java.type:"byte" = conv %22;
                var.store %4 %23;
                %24 : java.type:"byte" = var.load %4;
                %25 : java.type:"int" = conv %24;
                %26 : java.type:"byte" = var.load %5;
                %27 : java.type:"int" = conv %26;
                %28 : java.type:"int" = lshl %25 %27;
                %29 : java.type:"byte" = conv %28;
                var.store %4 %29;
                %30 : java.type:"byte" = var.load %4;
                %31 : java.type:"int" = conv %30;
                %32 : java.type:"int" = constant @1;
                %33 : java.type:"int" = ashr %31 %32;
                %34 : java.type:"byte" = conv %33;
                var.store %4 %34;
                %35 : java.type:"byte" = var.load %4;
                %36 : java.type:"int" = conv %35;
                %37 : java.type:"long" = constant @1L;
                %38 : java.type:"int" = ashr %36 %37;
                %39 : java.type:"byte" = conv %38;
                var.store %4 %39;
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

    @Reflect
    @IR("""
            func @"test10" (%0 : java.type:"BinopTest", %1 : java.type:"boolean", %2 : java.type:"java.lang.Boolean")java.type:"void" -> {
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                %4 : Var<java.type:"java.lang.Boolean"> = var %2 @"B";
                %5 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %6 : java.type:"boolean" = var.load %3;
                        yield %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"java.lang.Boolean" = var.load %4;
                        %8 : java.type:"boolean" = invoke %7 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %8;
                    };
                %9 : Var<java.type:"boolean"> = var %5 @"or1";
                %10 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %11 : java.type:"java.lang.Boolean" = var.load %4;
                        %12 : java.type:"boolean" = invoke %11 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %12;
                    }
                    ()java.type:"boolean" -> {
                        %13 : java.type:"boolean" = var.load %3;
                        yield %13;
                    };
                %14 : Var<java.type:"boolean"> = var %10 @"or2";
                %15 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = var.load %3;
                        yield %16;
                    }
                    ()java.type:"boolean" -> {
                        %17 : java.type:"boolean" = var.load %3;
                        yield %17;
                    };
                %18 : Var<java.type:"boolean"> = var %15 @"or3";
                %19 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %20 : java.type:"java.lang.Boolean" = var.load %4;
                        %21 : java.type:"boolean" = invoke %20 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %21;
                    }
                    ()java.type:"boolean" -> {
                        %22 : java.type:"java.lang.Boolean" = var.load %4;
                        %23 : java.type:"boolean" = invoke %22 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %23;
                    };
                %24 : Var<java.type:"boolean"> = var %19 @"or4";
                %25 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %26 : java.type:"boolean" = var.load %3;
                        yield %26;
                    }
                    ()java.type:"boolean" -> {
                        %27 : java.type:"java.lang.Boolean" = var.load %4;
                        %28 : java.type:"boolean" = invoke %27 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %28;
                    };
                %29 : Var<java.type:"boolean"> = var %25 @"and1";
                %30 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %31 : java.type:"java.lang.Boolean" = var.load %4;
                        %32 : java.type:"boolean" = invoke %31 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %32;
                    }
                    ()java.type:"boolean" -> {
                        %33 : java.type:"boolean" = var.load %3;
                        yield %33;
                    };
                %34 : Var<java.type:"boolean"> = var %30 @"and2";
                %35 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %36 : java.type:"boolean" = var.load %3;
                        yield %36;
                    }
                    ()java.type:"boolean" -> {
                        %37 : java.type:"boolean" = var.load %3;
                        yield %37;
                    };
                %38 : Var<java.type:"boolean"> = var %35 @"and3";
                %39 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %40 : java.type:"java.lang.Boolean" = var.load %4;
                        %41 : java.type:"boolean" = invoke %40 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %41;
                    }
                    ()java.type:"boolean" -> {
                        %42 : java.type:"java.lang.Boolean" = var.load %4;
                        %43 : java.type:"boolean" = invoke %42 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %43;
                    };
                %44 : Var<java.type:"boolean"> = var %39 @"and4";
                %45 : java.type:"boolean" = var.load %3;
                %46 : java.type:"boolean" = not %45;
                %47 : Var<java.type:"boolean"> = var %46 @"not1";
                %48 : java.type:"java.lang.Boolean" = var.load %4;
                %49 : java.type:"boolean" = invoke %48 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                %50 : java.type:"boolean" = not %49;
                %51 : Var<java.type:"boolean"> = var %50 @"not2";
                return;
            };
            """)
    void test10(boolean b, Boolean B) {
        var or1 = b || B;
        var or2 = B || b;
        var or3 = b || b;
        var or4 = B || B;

        var and1 = b && B;
        var and2 = B && b;
        var and3 = b && b;
        var and4 = B && B;

        var not1 = !b;
        var not2 = !B;
    }
}
