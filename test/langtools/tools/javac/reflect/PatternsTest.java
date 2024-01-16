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
 * @summary Smoke test for code reflection with patterns.
 * @enablePreview
 * @build PatternsTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PatternsTest
 */

public class PatternsTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : PatternsTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.Object = var.load %2;
                %4 : java.lang.String = constant @null;
                %5 : Var<java.lang.String> = var %4 @"s";
                %6 : boolean = pattern.match %3
                    ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                        %7 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                        yield %7;
                    }
                    ^match(%8 : java.lang.String)void -> {
                        var.store %5 %8;
                        yield;
                    };
                %9 : Var<boolean> = var %6 @"x";
                return;
            };
            """)
    void test1(Object o) {
        boolean x = o instanceof String s;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : PatternsTest, %1 : java.lang.Object)java.lang.String -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.String = constant @null;
                %4 : Var<java.lang.String> = var %3 @"s";
                java.if
                    ()boolean -> {
                        %5 : java.lang.Object = var.load %2;
                        %6 : boolean = pattern.match %5
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                %7 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %7;
                            }
                            ^match(%8 : java.lang.String)void -> {
                                var.store %4 %8;
                                yield;
                            };
                        yield %6;
                    }
                    ^then()void -> {
                        %9 : java.lang.String = var.load %4;
                        return %9;
                    }
                    ^else()void -> {
                        %10 : java.lang.String = constant @"";
                        return %10;
                    };
                return;
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
            func @"test3" (%0 : PatternsTest, %1 : java.lang.Object)java.lang.String -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.String = constant @null;
                %4 : Var<java.lang.String> = var %3 @"s";
                java.if
                    ()boolean -> {
                        %5 : java.lang.Object = var.load %2;
                        %6 : boolean = pattern.match %5
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                %7 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %7;
                            }
                            ^match(%8 : java.lang.String)void -> {
                                var.store %4 %8;
                                yield;
                            };
                        %9 : boolean = not %6;
                        yield %9;
                    }
                    ^then()void -> {
                        %10 : java.lang.String = constant @"";
                        return %10;
                    }
                    ^else()void -> {
                        yield;
                    };
                %11 : java.lang.String = var.load %4;
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
            func @"test4" (%0 : PatternsTest, %1 : PatternsTest$Rectangle)void -> {
                %2 : Var<PatternsTest$Rectangle> = var %1 @"r";
                %3 : PatternsTest$ConcretePoint = constant @null;
                %4 : Var<PatternsTest$ConcretePoint> = var %3 @"p";
                %5 : PatternsTest$Color = constant @null;
                %6 : Var<PatternsTest$Color> = var %5 @"c";
                %7 : PatternsTest$ColoredPoint = constant @null;
                %8 : Var<PatternsTest$ColoredPoint> = var %7 @"lr";
                java.if
                    ()boolean -> {
                        %9 : PatternsTest$Rectangle = var.load %2;
                        %10 : boolean = pattern.match %9
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Record<PatternsTest$Rectangle> -> {
                                %11 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<PatternsTest$ConcretePoint> = pattern.binding @"p";
                                %12 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<PatternsTest$Color> = pattern.binding @"c";
                                %13 : java.lang.reflect.code.ExtendedOps$Pattern$Record<PatternsTest$ColoredPoint> = pattern.record %11 %12 @"(PatternsTest$ConcretePoint p, PatternsTest$Color c)PatternsTest$ColoredPoint";
                                %14 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<PatternsTest$ColoredPoint> = pattern.binding @"lr";
                                %15 : java.lang.reflect.code.ExtendedOps$Pattern$Record<PatternsTest$Rectangle> = pattern.record %13 %14 @"(PatternsTest$Point upperLeft, PatternsTest$Point lowerRight)PatternsTest$Rectangle";
                                yield %15;
                            }
                            ^match(%16 : PatternsTest$ConcretePoint, %17 : PatternsTest$Color, %18 : PatternsTest$ColoredPoint)void -> {
                                var.store %4 %16;
                                var.store %6 %17;
                                var.store %8 %18;
                                yield;
                            };
                        yield %10;
                    }
                    ^then()void -> {
                        %19 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %20 : PatternsTest$ConcretePoint = var.load %4;
                        invoke %19 %20 @"java.io.PrintStream::println(java.lang.Object)void";
                        %21 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %22 : PatternsTest$Color = var.load %6;
                        invoke %21 %22 @"java.io.PrintStream::println(java.lang.Object)void";
                        %23 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %24 : PatternsTest$ColoredPoint = var.load %8;
                        invoke %23 %24 @"java.io.PrintStream::println(java.lang.Object)void";
                        yield;
                    }
                    ^else()void -> {
                        %25 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %26 : java.lang.String = constant @"NO MATCH";
                        invoke %25 %26 @"java.io.PrintStream::println(java.lang.String)void";
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
            func @"test5" (%0 : PatternsTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.String = constant @null;
                %4 : Var<java.lang.String> = var %3 @"s";
                java.while
                    ^cond()boolean -> {
                        %5 : java.lang.Object = var.load %2;
                        %6 : boolean = pattern.match %5
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                %7 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %7;
                            }
                            ^match(%8 : java.lang.String)void -> {
                                var.store %4 %8;
                                yield;
                            };
                        yield %6;
                    }
                    ^body()void -> {
                        %9 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %10 : java.lang.String = var.load %4;
                        invoke %9 %10 @"java.io.PrintStream::println(java.lang.String)void";
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
            func @"test6" (%0 : PatternsTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.String = constant @null;
                %4 : Var<java.lang.String> = var %3 @"s";
                java.do.while
                    ^body()void -> {
                        java.continue;
                    }
                    ^cond()boolean -> {
                        %5 : java.lang.Object = var.load %2;
                        %6 : boolean = pattern.match %5
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                %7 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %7;
                            }
                            ^match(%8 : java.lang.String)void -> {
                                var.store %4 %8;
                                yield;
                            };
                        %9 : boolean = not %6;
                        yield %9;
                    };
                %10 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                %11 : java.lang.String = var.load %4;
                invoke %10 %11 @"java.io.PrintStream::println(java.lang.String)void";
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
            func @"test7" (%0 : PatternsTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.Number = constant @null;
                %4 : Var<java.lang.Number> = var %3 @"n";
                java.for
                    ^init()Var<int> -> {
                        %5 : int = constant @"0";
                        %6 : Var<int> = var %5 @"i";
                        yield %6;
                    }
                    ^cond(%7 : Var<int>)boolean -> {
                        %8 : boolean = java.cand
                            ()boolean -> {
                                %9 : int = var.load %7;
                                %10 : int = constant @"10";
                                %11 : boolean = lt %9 %10;
                                yield %11;
                            }
                            ()boolean -> {
                                %12 : java.lang.Object = var.load %2;
                                %13 : boolean = pattern.match %12
                                    ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.Number> -> {
                                        %14 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                        yield %14;
                                    }
                                    ^match(%15 : java.lang.Number)void -> {
                                        var.store %4 %15;
                                        yield;
                                    };
                                yield %13;
                            };
                        yield %8;
                    }
                    ^update(%16 : Var<int>)void -> {
                        %17 : int = var.load %16;
                        %18 : java.lang.Number = var.load %4;
                        %19 : int = invoke %18 @"java.lang.Number::intValue()int";
                        %20 : int = add %17 %19;
                        var.store %16 %20;
                        yield;
                    }
                    ^body(%21 : Var<int>)void -> {
                        %22 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %23 : java.lang.Number = var.load %4;
                        invoke %22 %23 @"java.io.PrintStream::println(java.lang.Object)void";
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
}
