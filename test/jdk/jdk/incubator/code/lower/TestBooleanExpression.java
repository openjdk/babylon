/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @modules jdk.incubator.code
 * @summary test lowering of boolean expressions
 * @build TestBooleanExpression
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestBooleanExpression
 */

import jdk.incubator.code.Reflect;

public class TestBooleanExpression {

    @Reflect
    @LoweredModel("""
            func @"testIf" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"a";
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                %4 : java.type:"boolean" = var.load %2;
                cbranch %4 ^block_1 ^block_3;

              ^block_1:
                %5 : java.type:"boolean" = var.load %3;
                cbranch %5 ^block_2 ^block_3;

              ^block_2:
                %6 : java.type:"java.lang.String" = constant @"BODY";
                invoke %6 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_3;

              ^block_3:
                return;
            };
            """)
    static void testIf(boolean a, boolean b) {
        if (a && b) {
            IO.println("BODY");
        }
    }

    @Reflect
    @LoweredModel("""
            func @"testWhile" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"a";
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                branch ^block_1;

              ^block_1:
                %4 : java.type:"boolean" = var.load %2;
                cbranch %4 ^block_2 ^block_4;

              ^block_2:
                %5 : java.type:"boolean" = var.load %3;
                cbranch %5 ^block_3 ^block_4;

              ^block_3:
                %6 : java.type:"java.lang.String" = constant @"BODY";
                invoke %6 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_1;

              ^block_4:
                return;
            };
            """)
    static void testWhile(boolean a, boolean b) {
        while (a && b) {
            IO.println("BODY");
        }
    }

    @Reflect
    @LoweredModel("""
            func @"testDoWhile" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"a";
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                branch ^block_1;

              ^block_1:
                %4 : java.type:"java.lang.String" = constant @"BODY";
                invoke %4 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_2;

              ^block_2:
                %5 : java.type:"boolean" = var.load %2;
                %6 : java.type:"boolean" = var.load %3;
                %7 : java.type:"boolean" = and %5 %6;
                cbranch %7 ^block_1 ^block_3;

              ^block_3:
                return;
            };
            """)
    static void testDoWhile(boolean a, boolean b) {
        do {
            IO.println("BODY");
        } while (a & b);
    }

    @Reflect
    @LoweredModel("""
            func @"testFor" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"a";
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                branch ^block_1;

              ^block_1:
                %4 : java.type:"boolean" = var.load %2;
                %5 : java.type:"boolean" = var.load %3;
                %6 : java.type:"boolean" = and %4 %5;
                cbranch %6 ^block_2 ^block_4;

              ^block_2:
                %7 : java.type:"java.lang.String" = constant @"BODY";
                invoke %7 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_3;

              ^block_3:
                branch ^block_1;

              ^block_4:
                return;
            };
            """)
    static void testFor(boolean a, boolean b) {
        for (; a & b;) {
            IO.println("BODY");
        }
    }

    record Box(Object o) {}

    @Reflect
    @LoweredModel("""
            func @"testSwitchLabel" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                %5 : java.type:"java.lang.Object" = constant @null;
                %6 : java.type:"boolean" = invoke %2 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                cbranch %6 ^block_1 ^block_2;

              ^block_1:
                %7 : java.type:"java.lang.NullPointerException" = new @java.ref:"java.lang.NullPointerException::()";
                throw %7;

              ^block_2:
                %8 : java.type:"boolean" = instanceof %2 @java.type:"TestBooleanExpression$Box";
                cbranch %8 ^block_3 ^block_6;

              ^block_3:
                %9 : java.type:"TestBooleanExpression$Box" = cast %2 @java.type:"TestBooleanExpression$Box";
                %10 : java.type:"java.lang.Object" = invoke %9 @java.ref:"TestBooleanExpression$Box::o():java.lang.Object";
                %11 : java.type:"boolean" = instanceof %10 @java.type:"java.lang.String";
                cbranch %11 ^block_4 ^block_6;

              ^block_4:
                %12 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                var.store %4 %12;
                branch ^block_5;

              ^block_5:
                %13 : java.type:"java.lang.String" = constant @"CASE";
                invoke %13 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_7;

              ^block_6:
                %14 : java.type:"java.lang.String" = constant @"DEFAULT";
                invoke %14 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_7;

              ^block_7:
                return;
            };
            """)
    static void testSwitchLabel(Object o) {
        switch (o) {
            case Box(String s) -> {
                IO.println("CASE");
            }
            default -> {
                IO.println("DEFAULT");
            }
        }
    }

    @Reflect
    @LoweredModel("""
            func @"testSwitchLabelAndGuard" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                %5 : java.type:"java.lang.Object" = constant @null;
                %6 : java.type:"boolean" = invoke %2 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                cbranch %6 ^block_1 ^block_2;

              ^block_1:
                %7 : java.type:"java.lang.NullPointerException" = new @java.ref:"java.lang.NullPointerException::()";
                throw %7;

              ^block_2:
                %8 : java.type:"boolean" = instanceof %2 @java.type:"TestBooleanExpression$Box";
                cbranch %8 ^block_3 ^block_8;

              ^block_3:
                %9 : java.type:"TestBooleanExpression$Box" = cast %2 @java.type:"TestBooleanExpression$Box";
                %10 : java.type:"java.lang.Object" = invoke %9 @java.ref:"TestBooleanExpression$Box::o():java.lang.Object";
                %11 : java.type:"boolean" = instanceof %10 @java.type:"java.lang.String";
                cbranch %11 ^block_4 ^block_8;

              ^block_4:
                %12 : java.type:"java.lang.String" = cast %10 @java.type:"java.lang.String";
                var.store %4 %12;
                branch ^block_5;

              ^block_5:
                %13 : java.type:"java.lang.String" = var.load %4;
                %14 : java.type:"int" = invoke %13 @java.ref:"java.lang.String::length():int";
                %15 : java.type:"int" = constant @10;
                %16 : java.type:"boolean" = gt %14 %15;
                cbranch %16 ^block_6 ^block_8;

              ^block_6:
                %17 : java.type:"java.lang.String" = var.load %4;
                %18 : java.type:"int" = invoke %17 @java.ref:"java.lang.String::length():int";
                %19 : java.type:"int" = constant @20;
                %20 : java.type:"boolean" = lt %18 %19;
                cbranch %20 ^block_7 ^block_8;

              ^block_7:
                %21 : java.type:"java.lang.String" = constant @"CASE";
                invoke %21 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_9;

              ^block_8:
                %22 : java.type:"java.lang.String" = constant @"DEFAULT";
                invoke %22 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_9;

              ^block_9:
                return;
            };
            """)
    static void testSwitchLabelAndGuard(Object o) {
        switch (o) {
            case Box(String s) when s.length() > 10 && s.length() < 20 -> {
                IO.println("CASE");
            }
            default -> {
                IO.println("DEFAULT");
            }
        }
    }

    @Reflect
    @LoweredModel("""
            func @"testConditional" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %0 @"a";
                %3 : Var<java.type:"boolean"> = var %1 @"b";
                %4 : java.type:"boolean" = var.load %2;
                %5 : java.type:"boolean" = var.load %3;
                %6 : java.type:"boolean" = and %4 %5;
                cbranch %6 ^block_1 ^block_2;

              ^block_1:
                %7 : java.type:"java.lang.String" = constant @"LEFT";
                branch ^block_3(%7);

              ^block_2:
                %8 : java.type:"java.lang.String" = constant @"RIGHT";
                branch ^block_3(%8);

              ^block_3(%9 : java.type:"java.lang.String"):
                %10 : Var<java.type:"java.lang.String"> = var %9 @"s";
                return;
            };
            """)
    static void testConditional(boolean a, boolean b) {
        String s = (a & b) ? "LEFT" : "RIGHT";
    }



    // Boolean expressions
    // &&
    // ||
    // patterns
    // ternary
    // switch expression
    // Composed with !


    @Reflect
    @LoweredModel("""
            func @"testConditionalAnd" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"boolean" -> {
                %3 : Var<java.type:"boolean"> = var %0 @"a";
                %4 : Var<java.type:"boolean"> = var %1 @"b";
                %5 : Var<java.type:"boolean"> = var %2 @"c";
                %6 : java.type:"boolean" = constant @false;
                %7 : java.type:"boolean" = var.load %3;
                cbranch %7 ^block_1 ^block_3(%6);

              ^block_1:
                %8 : java.type:"boolean" = var.load %4;
                cbranch %8 ^block_2 ^block_3(%6);

              ^block_2:
                %9 : java.type:"boolean" = var.load %5;
                branch ^block_3(%9);

              ^block_3(%10 : java.type:"boolean"):
                return %10;
            };
            """)
    static boolean testConditionalAnd(boolean a, boolean b, boolean c) {
        return a && b && c;
    }

    @Reflect
    @LoweredModel("""
            func @"testConditionalOr" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"boolean" -> {
                %3 : Var<java.type:"boolean"> = var %0 @"a";
                %4 : Var<java.type:"boolean"> = var %1 @"b";
                %5 : Var<java.type:"boolean"> = var %2 @"c";
                %6 : java.type:"boolean" = constant @true;
                %7 : java.type:"boolean" = var.load %3;
                cbranch %7 ^block_3(%6) ^block_1;

              ^block_1:
                %8 : java.type:"boolean" = var.load %4;
                cbranch %8 ^block_3(%6) ^block_2;

              ^block_2:
                %9 : java.type:"boolean" = var.load %5;
                branch ^block_3(%9);

              ^block_3(%10 : java.type:"boolean"):
                return %10;
            };
            """)
    static boolean testConditionalOr(boolean a, boolean b, boolean c) {
        return a || b || c;
    }

    @Reflect
    @LoweredModel("""
            func @"testConditionalAndOr" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"boolean")java.type:"boolean" -> {
                %4 : Var<java.type:"boolean"> = var %0 @"a";
                %5 : Var<java.type:"boolean"> = var %1 @"b";
                %6 : Var<java.type:"boolean"> = var %2 @"c";
                %7 : Var<java.type:"boolean"> = var %3 @"d";
                %8 : java.type:"boolean" = constant @false;
                %9 : java.type:"boolean" = var.load %4;
                cbranch %9 ^block_2 ^block_1;

              ^block_1:
                %10 : java.type:"boolean" = var.load %5;
                cbranch %10 ^block_2 ^block_4(%8);

              ^block_2:
                %11 : java.type:"boolean" = constant @true;
                %12 : java.type:"boolean" = constant @false;
                %13 : java.type:"boolean" = var.load %6;
                cbranch %13 ^block_4(%11) ^block_3;

              ^block_3:
                %14 : java.type:"boolean" = var.load %7;
                cbranch %14 ^block_4(%11) ^block_4(%12);

              ^block_4(%15 : java.type:"boolean"):
                return %15;
            };
            """)
    static boolean testConditionalAndOr(boolean a, boolean b, boolean c, boolean d) {
        return (a || b) && (c || d);
    }

    @Reflect
    @LoweredModel("""
            func @"testPattern" (%0 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                %5 : java.type:"boolean" = constant @true;
                %6 : java.type:"boolean" = constant @false;
                %7 : java.type:"boolean" = instanceof %2 @java.type:"TestBooleanExpression$Box";
                cbranch %7 ^block_1 ^block_4(%6);

              ^block_1:
                %8 : java.type:"TestBooleanExpression$Box" = cast %2 @java.type:"TestBooleanExpression$Box";
                %9 : java.type:"java.lang.Object" = invoke %8 @java.ref:"TestBooleanExpression$Box::o():java.lang.Object";
                %10 : java.type:"boolean" = instanceof %9 @java.type:"TestBooleanExpression$Box";
                cbranch %10 ^block_2 ^block_4(%6);

              ^block_2:
                %11 : java.type:"TestBooleanExpression$Box" = cast %9 @java.type:"TestBooleanExpression$Box";
                %12 : java.type:"java.lang.Object" = invoke %11 @java.ref:"TestBooleanExpression$Box::o():java.lang.Object";
                %13 : java.type:"boolean" = instanceof %12 @java.type:"java.lang.String";
                cbranch %13 ^block_3 ^block_4(%6);

              ^block_3:
                %14 : java.type:"java.lang.String" = cast %12 @java.type:"java.lang.String";
                var.store %4 %14;
                branch ^block_4(%5);

              ^block_4(%15 : java.type:"boolean"):
                return %15;
            };
            """)
    static boolean testPattern(Object o) {
        return o instanceof Box(Box(String s));
    }

    @Reflect
    @LoweredModel("""
            func @"testBooleanConditional" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"boolean")java.type:"boolean" -> {
                %4 : Var<java.type:"boolean"> = var %0 @"a";
                %5 : Var<java.type:"boolean"> = var %1 @"b";
                %6 : Var<java.type:"boolean"> = var %2 @"c";
                %7 : Var<java.type:"boolean"> = var %3 @"d";
                %8 : java.type:"boolean" = var.load %4;
                cbranch %8 ^block_1 ^block_3;

              ^block_1:
                %9 : java.type:"boolean" = constant @true;
                %10 : java.type:"boolean" = constant @false;
                %11 : java.type:"boolean" = var.load %4;
                cbranch %11 ^block_2 ^block_5(%10);

              ^block_2:
                %12 : java.type:"boolean" = var.load %5;
                cbranch %12 ^block_5(%9) ^block_5(%10);

              ^block_3:
                %13 : java.type:"boolean" = constant @true;
                %14 : java.type:"boolean" = constant @false;
                %15 : java.type:"boolean" = var.load %6;
                cbranch %15 ^block_4 ^block_5(%14);

              ^block_4:
                %16 : java.type:"boolean" = var.load %7;
                cbranch %16 ^block_5(%13) ^block_5(%14);

              ^block_5(%17 : java.type:"boolean"):
                return %17;
            };
            """)
    static boolean testBooleanConditional(boolean a, boolean b, boolean c, boolean d) {
        return a ? a && b : c && d;
    }

    @Reflect
    @LoweredModel("""
            func @"testBooleanSwitch" (%0 : java.type:"int", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"boolean", %4 : java.type:"boolean")java.type:"boolean" -> {
                %5 : Var<java.type:"int"> = var %0 @"i";
                %6 : Var<java.type:"boolean"> = var %1 @"a";
                %7 : Var<java.type:"boolean"> = var %2 @"b";
                %8 : Var<java.type:"boolean"> = var %3 @"c";
                %9 : Var<java.type:"boolean"> = var %4 @"d";
                %10 : java.type:"int" = var.load %5;
                %11 : java.type:"int" = constant @0;
                %12 : java.type:"boolean" = eq %10 %11;
                cbranch %12 ^block_1 ^block_3;

              ^block_1:
                %13 : java.type:"boolean" = constant @true;
                %14 : java.type:"boolean" = constant @false;
                %15 : java.type:"boolean" = var.load %6;
                cbranch %15 ^block_2 ^block_5(%14);

              ^block_2:
                %16 : java.type:"boolean" = var.load %7;
                cbranch %16 ^block_5(%13) ^block_5(%14);

              ^block_3:
                %17 : java.type:"boolean" = constant @true;
                %18 : java.type:"boolean" = constant @false;
                %19 : java.type:"boolean" = var.load %8;
                cbranch %19 ^block_4 ^block_5(%18);

              ^block_4:
                %20 : java.type:"boolean" = var.load %9;
                cbranch %20 ^block_5(%17) ^block_5(%18);

              ^block_5(%21 : java.type:"boolean"):
                return %21;
            };
            """)
    static boolean testBooleanSwitch(int i, boolean a, boolean b, boolean c, boolean d) {
        return switch (i) {
            case 0 -> a && b;
            default -> c && d;
        };
    }


    @Reflect
    @LoweredModel("""
            func @"testNot" (%0 : java.type:"int", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"boolean")java.type:"boolean" -> {
                %4 : Var<java.type:"int"> = var %0 @"i";
                %5 : Var<java.type:"boolean"> = var %1 @"a";
                %6 : Var<java.type:"boolean"> = var %2 @"b";
                %7 : Var<java.type:"boolean"> = var %3 @"c";
                %8 : java.type:"boolean" = constant @false;
                %9 : java.type:"boolean" = var.load %5;
                cbranch %9 ^block_1 ^block_3(%8);

              ^block_1:
                %10 : java.type:"boolean" = constant @false;
                %11 : java.type:"boolean" = constant @true;
                %12 : java.type:"boolean" = var.load %6;
                cbranch %12 ^block_2 ^block_3(%11);

              ^block_2:
                %13 : java.type:"boolean" = var.load %7;
                cbranch %13 ^block_3(%10) ^block_3(%11);

              ^block_3(%14 : java.type:"boolean"):
                return %14;
            };
            """)
    static boolean testNot(int i, boolean a, boolean b, boolean c) {
        return (a && !(b && c));
    }

    @Reflect
    @LoweredModel("""
            func @"testNotNot" (%0 : java.type:"int", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"boolean")java.type:"boolean" -> {
                %4 : Var<java.type:"int"> = var %0 @"i";
                %5 : Var<java.type:"boolean"> = var %1 @"a";
                %6 : Var<java.type:"boolean"> = var %2 @"b";
                %7 : Var<java.type:"boolean"> = var %3 @"c";
                %8 : java.type:"boolean" = constant @false;
                %9 : java.type:"boolean" = var.load %5;
                cbranch %9 ^block_1 ^block_3(%8);

              ^block_1:
                %10 : java.type:"boolean" = constant @true;
                %11 : java.type:"boolean" = constant @false;
                %12 : java.type:"boolean" = var.load %6;
                cbranch %12 ^block_2 ^block_3(%11);

              ^block_2:
                %13 : java.type:"boolean" = var.load %7;
                cbranch %13 ^block_3(%10) ^block_3(%11);

              ^block_3(%14 : java.type:"boolean"):
                return %14;
            };
            """)
    static boolean testNotNot(int i, boolean a, boolean b, boolean c) {
        return (a && !!(b && c));
    }
}
