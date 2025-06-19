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
 * @summary Smoke test for code reflection with patterns.
 * @modules jdk.incubator.code
 * @enablePreview
 * @build PatternsTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PatternsTest
 */

public class PatternsTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : java.type:"java.lang.String" = constant @null;
                %5 : Var<java.type:"java.lang.String"> = var %4 @"s";
                %6 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"void" -> {
                        var.store %5 %8;
                        yield;
                    };
                %9 : Var<java.type:"boolean"> = var %6 @"x";
                return;
            };
            """)
    void test1(Object o) {
        boolean x = o instanceof String s;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                java.if
                    ()java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Object" = var.load %2;
                        %6 : java.type:"boolean" = pattern.match %5
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %7;
                            }
                            (%8 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %4 %8;
                                yield;
                            };
                        yield %6;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = var.load %4;
                        return %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"java.lang.String" = constant @"";
                        return %10;
                    };
                unreachable;
            };
            """)
    String test2(Object o) {
        if (o instanceof String s) {
            return s;
        } else {
            return "";
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                java.if
                    ()java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Object" = var.load %2;
                        %6 : java.type:"boolean" = pattern.match %5
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %7;
                            }
                            (%8 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %4 %8;
                                yield;
                            };
                        %9 : java.type:"boolean" = not %6;
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"java.lang.String" = constant @"";
                        return %10;
                    }
                    ()java.type:"void" -> {
                        yield;
                    };
                %11 : java.type:"java.lang.String" = var.load %4;
                return %11;
            };
            """)
    String test3(Object o) {
        if (!(o instanceof String s)) {
            return "";
        }
        return s;
    }

    interface Point {
    }

    record ConcretePoint(int x, int y) implements Point {
    }

    enum Color {RED, GREEN, BLUE}

    record ColoredPoint(ConcretePoint p, Color c) implements Point {
    }

    record Rectangle(Point upperLeft, Point lowerRight) {
    }


    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"PatternsTest", %1 : java.type:"PatternsTest$Rectangle")java.type:"void" -> {
                %2 : Var<java.type:"PatternsTest$Rectangle"> = var %1 @"r";
                %3 : java.type:"PatternsTest$ConcretePoint" = constant @null;
                %4 : Var<java.type:"PatternsTest$ConcretePoint"> = var %3 @"p";
                %5 : java.type:"PatternsTest$Color" = constant @null;
                %6 : Var<java.type:"PatternsTest$Color"> = var %5 @"c";
                %7 : java.type:"PatternsTest$ColoredPoint" = constant @null;
                %8 : Var<java.type:"PatternsTest$ColoredPoint"> = var %7 @"lr";
                java.if
                    ()java.type:"boolean" -> {
                        %9 : java.type:"PatternsTest$Rectangle" = var.load %2;
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ConcretePoint>" = pattern.type @"p";
                                %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$Color>" = pattern.type @"c";
                                %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$ColoredPoint>" = pattern.record %11 %12 @java.ref:"(PatternsTest$ConcretePoint p, PatternsTest$Color c)PatternsTest$ColoredPoint";
                                %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ColoredPoint>" = pattern.type @"lr";
                                %15 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" = pattern.record %13 %14 @java.ref:"(PatternsTest$Point upperLeft, PatternsTest$Point lowerRight)PatternsTest$Rectangle";
                                yield %15;
                            }
                            (%16 : java.type:"PatternsTest$ConcretePoint", %17 : java.type:"PatternsTest$Color", %18 : java.type:"PatternsTest$ColoredPoint")java.type:"void" -> {
                                var.store %4 %16;
                                var.store %6 %17;
                                var.store %8 %18;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %19 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %20 : java.type:"PatternsTest$ConcretePoint" = var.load %4;
                        invoke %19 %20 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        %21 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %22 : java.type:"PatternsTest$Color" = var.load %6;
                        invoke %21 %22 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        %23 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %24 : java.type:"PatternsTest$ColoredPoint" = var.load %8;
                        invoke %23 %24 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        yield;
                    }
                    ()java.type:"void" -> {
                        %25 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %26 : java.type:"java.lang.String" = constant @"NO MATCH";
                        invoke %25 %26 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                        yield;
                    };
                return;
            };
            """)
    void test4(Rectangle r) {
        if (r instanceof Rectangle(
                ColoredPoint(ConcretePoint p, Color c),
                ColoredPoint lr)){
            System.out.println(p);
            System.out.println(c);
            System.out.println(lr);
        }
        else {
            System.out.println("NO MATCH");
        }
    }


    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                java.while
                    ()java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Object" = var.load %2;
                        %6 : java.type:"boolean" = pattern.match %5
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %7;
                            }
                            (%8 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %4 %8;
                                yield;
                            };
                        yield %6;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %10 : java.type:"java.lang.String" = var.load %4;
                        invoke %9 %10 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test5(Object o) {
        while (o instanceof String s) {
            System.out.println(s);
        }
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : Var<java.type:"java.lang.String"> = var %3 @"s";
                java.do.while
                    ()java.type:"void" -> {
                        java.continue;
                    }
                    ()java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Object" = var.load %2;
                        %6 : java.type:"boolean" = pattern.match %5
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %7;
                            }
                            (%8 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %4 %8;
                                yield;
                            };
                        %9 : java.type:"boolean" = not %6;
                        yield %9;
                    };
                %10 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                %11 : java.type:"java.lang.String" = var.load %4;
                invoke %10 %11 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                return;
            };
            """)
    void test6(Object o) {
        do {
        } while (!(o instanceof String s));
        System.out.println(s);
    }


    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Number" = constant @null;
                %4 : Var<java.type:"java.lang.Number"> = var %3 @"n";
                java.for
                    ()Var<java.type:"int"> -> {
                        %5 : java.type:"int" = constant @0;
                        %6 : Var<java.type:"int"> = var %5 @"i";
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"boolean" -> {
                        %8 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %9 : java.type:"int" = var.load %7;
                                %10 : java.type:"int" = constant @10;
                                %11 : java.type:"boolean" = lt %9 %10;
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %12 : java.type:"java.lang.Object" = var.load %2;
                                %13 : java.type:"boolean" = pattern.match %12
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                        %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                        yield %14;
                                    }
                                    (%15 : java.type:"java.lang.Number")java.type:"void" -> {
                                        var.store %4 %15;
                                        yield;
                                    };
                                yield %13;
                            };
                        yield %8;
                    }
                    (%16 : Var<java.type:"int">)java.type:"void" -> {
                        %17 : java.type:"int" = var.load %16;
                        %18 : java.type:"java.lang.Number" = var.load %4;
                        %19 : java.type:"int" = invoke %18 @java.ref:"java.lang.Number::intValue():int";
                        %20 : java.type:"int" = add %17 %19;
                        var.store %16 %20;
                        yield;
                    }
                    (%21 : Var<java.type:"int">)java.type:"void" -> {
                        %22 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %23 : java.type:"java.lang.Number" = var.load %4;
                        invoke %22 %23 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test7(Object o) {
        for (int i = 0;
             i < 10 && o instanceof Number n; i += n.intValue()) {
            System.out.println(n);
        }
    }

    @IR("""
            func @"test8" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : java.type:"java.lang.String" = constant @null;
                %5 : Var<java.type:"java.lang.String"> = var %4;
                %6 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type;
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"void" -> {
                        var.store %5 %8;
                        yield;
                    };
                return %6;
            };
            """)
    @CodeReflection
    boolean test8(Object o) {
        return o instanceof String _;
    }

    @IR("""
            func @"test9" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : java.type:"PatternsTest$ConcretePoint" = constant @null;
                %5 : Var<java.type:"PatternsTest$ConcretePoint"> = var %4 @"cp";
                %6 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" -> {
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$MatchAll" = pattern.match.all;
                        %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ConcretePoint>" = pattern.type @"cp";
                        %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" = pattern.record %7 %8 @java.ref:"(PatternsTest$Point upperLeft, PatternsTest$Point lowerRight)PatternsTest$Rectangle";
                        yield %9;
                    }
                    (%10 : java.type:"PatternsTest$ConcretePoint")java.type:"void" -> {
                        var.store %5 %10;
                        yield;
                    };
                return %6;
            };
            """)
    @CodeReflection
    boolean test9(Object o) {
        return o instanceof Rectangle(_, ConcretePoint cp);
    }

    @IR("""
            func @"test10" (%0 : java.type:"int")java.type:"boolean" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"byte" = conv %3;
                %5 : Var<java.type:"byte"> = var %4 @"b";
                %6 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" -> {
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" = pattern.type @"b";
                        yield %7;
                    }
                    (%8 : java.type:"byte")java.type:"void" -> {
                        var.store %5 %8;
                        yield;
                    };
                return %6;
            };
            """)
    @CodeReflection
    static boolean test10(int i) {
        return i instanceof byte b;
    }

    @IR("""
            func @"test11" (%0 : java.type:"int")java.type:"boolean" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"short" = conv %3;
                %5 : Var<java.type:"short"> = var %4 @"s";
                %6 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<short>" -> {
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<short>" = pattern.type @"s";
                        yield %7;
                    }
                    (%8 : java.type:"short")java.type:"void" -> {
                        var.store %5 %8;
                        yield;
                    };
                return %6;
            };
            """)
    @CodeReflection
    static boolean test11(int i) {
        return i instanceof short s;
    }
}
