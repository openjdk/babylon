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

import jdk.incubator.code.Reflect;


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

    @Reflect
    @IR("""
            func @"test1" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : Var<java.type:"java.lang.String"> = var @"s";
                %5 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                        %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                        yield %6;
                    }
                    (%7 : java.type:"java.lang.String")java.type:"void" -> {
                        var.store %4 %7;
                        yield;
                    };
                %8 : Var<java.type:"boolean"> = var %5 @"x";
                return;
            };
            """)
    void test1(Object o) {
        boolean x = o instanceof String s;
    }

    @Reflect
    @IR("""
            func @"test2" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : Var<java.type:"java.lang.String"> = var @"s";
                java.if
                    ()java.type:"boolean" -> {
                        %4 : java.type:"java.lang.Object" = var.load %2;
                        %5 : java.type:"boolean" = pattern.match %4
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %6;
                            }
                            (%7 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %3 %7;
                                yield;
                            };
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        return %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = constant @"";
                        return %9;
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

    @Reflect
    @IR("""
            func @"test3" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : Var<java.type:"java.lang.String"> = var @"s";
                java.if
                    ()java.type:"boolean" -> {
                        %4 : java.type:"java.lang.Object" = var.load %2;
                        %5 : java.type:"boolean" = pattern.match %4
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %6;
                            }
                            (%7 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %3 %7;
                                yield;
                            };
                        %8 : java.type:"boolean" = not %5;
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = constant @"";
                        return %9;
                    };
                %10 : java.type:"java.lang.String" = var.load %3;
                return %10;
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


    @Reflect
    @IR("""
            func @"test4" (%0 : java.type:"PatternsTest", %1 : java.type:"PatternsTest$Rectangle")java.type:"void" -> {
                %2 : Var<java.type:"PatternsTest$Rectangle"> = var %1 @"r";
                %3 : Var<java.type:"PatternsTest$ConcretePoint"> = var @"p";
                %4 : Var<java.type:"PatternsTest$Color"> = var @"c";
                %5 : Var<java.type:"PatternsTest$ColoredPoint"> = var @"lr";
                java.if
                    ()java.type:"boolean" -> {
                        %6 : java.type:"PatternsTest$Rectangle" = var.load %2;
                        %7 : java.type:"boolean" = pattern.match %6
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" -> {
                                %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ConcretePoint>" = pattern.type @"p";
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$Color>" = pattern.type @"c";
                                %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$ColoredPoint>" = pattern.record %8 %9 @java.ref:"(PatternsTest$ConcretePoint p, PatternsTest$Color c)PatternsTest$ColoredPoint";
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ColoredPoint>" = pattern.type @"lr";
                                %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" = pattern.record %10 %11 @java.ref:"(PatternsTest$Point upperLeft, PatternsTest$Point lowerRight)PatternsTest$Rectangle";
                                yield %12;
                            }
                            (%13 : java.type:"PatternsTest$ConcretePoint", %14 : java.type:"PatternsTest$Color", %15 : java.type:"PatternsTest$ColoredPoint")java.type:"void" -> {
                                var.store %3 %13;
                                var.store %4 %14;
                                var.store %5 %15;
                                yield;
                            };
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %16 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %17 : java.type:"PatternsTest$ConcretePoint" = var.load %3;
                        invoke %16 %17 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        %18 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %19 : java.type:"PatternsTest$Color" = var.load %4;
                        invoke %18 %19 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        %20 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %21 : java.type:"PatternsTest$ColoredPoint" = var.load %5;
                        invoke %20 %21 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
                        yield;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %23 : java.type:"java.lang.String" = constant @"NO MATCH";
                        invoke %22 %23 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
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


    @Reflect
    @IR("""
            func @"test5" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : Var<java.type:"java.lang.String"> = var @"s";
                java.while
                    ()java.type:"boolean" -> {
                        %4 : java.type:"java.lang.Object" = var.load %2;
                        %5 : java.type:"boolean" = pattern.match %4
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %6;
                            }
                            (%7 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %3 %7;
                                yield;
                            };
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %9 : java.type:"java.lang.String" = var.load %3;
                        invoke %8 %9 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
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

    @Reflect
    @IR("""
            func @"test6" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : Var<java.type:"java.lang.String"> = var @"s";
                java.do.while
                    ()java.type:"void" -> {
                        java.continue;
                    }
                    ()java.type:"boolean" -> {
                        %4 : java.type:"java.lang.Object" = var.load %2;
                        %5 : java.type:"boolean" = pattern.match %4
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %6;
                            }
                            (%7 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %3 %7;
                                yield;
                            };
                        %8 : java.type:"boolean" = not %5;
                        yield %8;
                    };
                %9 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                %10 : java.type:"java.lang.String" = var.load %3;
                invoke %9 %10 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                return;
            };
            """)
    void test6(Object o) {
        do {
        } while (!(o instanceof String s));
        System.out.println(s);
    }


    @Reflect
    @IR("""
            func @"test7" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : Var<java.type:"java.lang.Number"> = var @"n";
                java.for
                    ()Var<java.type:"int"> -> {
                        %4 : java.type:"int" = constant @0;
                        %5 : Var<java.type:"int"> = var %4 @"i";
                        yield %5;
                    }
                    (%6 : Var<java.type:"int">)java.type:"boolean" -> {
                        %7 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %8 : java.type:"int" = var.load %6;
                                %9 : java.type:"int" = constant @10;
                                %10 : java.type:"boolean" = lt %8 %9;
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %11 : java.type:"java.lang.Object" = var.load %2;
                                %12 : java.type:"boolean" = pattern.match %11
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                        %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                        yield %13;
                                    }
                                    (%14 : java.type:"java.lang.Number")java.type:"void" -> {
                                        var.store %3 %14;
                                        yield;
                                    };
                                yield %12;
                            };
                        yield %7;
                    }
                    (%15 : Var<java.type:"int">)java.type:"void" -> {
                        %16 : java.type:"int" = var.load %15;
                        %17 : java.type:"java.lang.Number" = var.load %3;
                        %18 : java.type:"int" = invoke %17 @java.ref:"java.lang.Number::intValue():int";
                        %19 : java.type:"int" = add %16 %18;
                        var.store %15 %19;
                        yield;
                    }
                    (%20 : Var<java.type:"int">)java.type:"void" -> {
                        %21 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %22 : java.type:"java.lang.Number" = var.load %3;
                        invoke %21 %22 @java.ref:"java.io.PrintStream::println(java.lang.Object):void";
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
                %4 : Var<java.type:"java.lang.String"> = var;
                %5 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                        %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type;
                        yield %6;
                    }
                    (%7 : java.type:"java.lang.String")java.type:"void" -> {
                        var.store %4 %7;
                        yield;
                    };
                return %5;
            };
            """)
    @Reflect
    boolean test8(Object o) {
        return o instanceof String _;
    }

    @IR("""
            func @"test9" (%0 : java.type:"PatternsTest", %1 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : Var<java.type:"PatternsTest$ConcretePoint"> = var @"cp";
                %5 : java.type:"boolean" = pattern.match %3
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" -> {
                        %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$MatchAll" = pattern.match.all;
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<PatternsTest$ConcretePoint>" = pattern.type @"cp";
                        %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternsTest$Rectangle>" = pattern.record %6 %7 @java.ref:"(PatternsTest$Point upperLeft, PatternsTest$Point lowerRight)PatternsTest$Rectangle";
                        yield %8;
                    }
                    (%9 : java.type:"PatternsTest$ConcretePoint")java.type:"void" -> {
                        var.store %4 %9;
                        yield;
                    };
                return %5;
            };
            """)
    @Reflect
    boolean test9(Object o) {
        return o instanceof Rectangle(_, ConcretePoint cp);
    }

    @IR("""
            func @"test10" (%0 : java.type:"int")java.type:"boolean" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : Var<java.type:"byte"> = var @"b";
                %4 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" -> {
                        %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" = pattern.type @"b";
                        yield %5;
                    }
                    (%6 : java.type:"byte")java.type:"void" -> {
                        var.store %3 %6;
                        yield;
                    };
                return %4;
            };
            """)
    @Reflect
    static boolean test10(int i) {
        return i instanceof byte b;
    }

    @IR("""
            func @"test11" (%0 : java.type:"int")java.type:"boolean" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : Var<java.type:"short"> = var @"s";
                %4 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<short>" -> {
                        %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<short>" = pattern.type @"s";
                        yield %5;
                    }
                    (%6 : java.type:"short")java.type:"void" -> {
                        var.store %3 %6;
                        yield;
                    };
                return %4;
            };
            """)
    @Reflect
    static boolean test11(int i) {
        return i instanceof short s;
    }

    @IR("""
            func @"test12" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : Var<java.type:"byte"> = var @"b";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %1;
                        %4 : java.type:"boolean" = pattern.match %3
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" -> {
                                %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<byte>" = pattern.type @"b";
                                yield %5;
                            }
                            (%6 : java.type:"byte")java.type:"void" -> {
                                var.store %2 %6;
                                yield;
                            };
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        yield;
                    };
                return;
            };
            """)
    @Reflect
    static void test12(int i) {
        if (i instanceof byte b) {
        }
    }

    @IR("""
            func @"test13" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : Var<java.type:"java.lang.String"> = var @"s";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"java.lang.Object" = var.load %1;
                        %4 : java.type:"boolean" = pattern.match %3
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %5;
                            }
                            (%6 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %2 %6;
                                yield;
                            };
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"java.lang.String" = constant @"";
                        var.store %2 %7;
                        yield;
                    };
                return;
            };
            """)
    @Reflect
    static void test13(Object o) {
        if (o instanceof String s) {
            s = "";
        }
    }

    @IR("""
            func @"test14" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : Var<java.type:"int"> = var @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"java.lang.Object" = var.load %1;
                        %4 : java.type:"boolean" = pattern.match %3
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" -> {
                                %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" = pattern.type @"i";
                                yield %5;
                            }
                            (%6 : java.type:"int")java.type:"void" -> {
                                var.store %2 %6;
                                yield;
                            };
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = constant @1;
                        var.store %2 %7;
                        yield;
                    };
                return;
            };
            """)
    @Reflect
    static void test14(Object o) {
        if (o instanceof int i) {
            i = 1;
        }
    }

    @IR("""
            func @"test15" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : Var<java.type:"int"> = var @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"java.lang.Object" = var.load %1;
                        %4 : java.type:"boolean" = pattern.match %3
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" -> {
                                %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" = pattern.type @"i";
                                yield %5;
                            }
                            (%6 : java.type:"int")java.type:"void" -> {
                                var.store %2 %6;
                                yield;
                            };
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"int" = add %7 %8;
                        var.store %2 %9;
                        yield;
                    };
                return;
            };
            """)
    @Reflect
    static void test15(Object o) {
        if (o instanceof int i) {
            i++;
        }
    }

    @IR("""
            func @"test16" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : Var<java.type:"int"> = var @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"java.lang.Object" = var.load %1;
                        %4 : java.type:"boolean" = pattern.match %3
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" -> {
                                %5 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int>" = pattern.type @"i";
                                yield %5;
                            }
                            (%6 : java.type:"int")java.type:"void" -> {
                                var.store %2 %6;
                                yield;
                            };
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"int" = add %7 %8;
                        var.store %2 %9;
                        yield;
                    };
                return;
            };
            """)
    @Reflect
    static void test16(Object o) {
        if (o instanceof int i) {
            i += 1;
        }
    }

}
