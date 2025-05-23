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
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with for loops.
 * @modules jdk.incubator.code
 * @build ForLoopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ForLoopTest
 */

public class ForLoopTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"ForLoopTest", %1 : java.type:"java.util.List<java.util.List<java.lang.String>>")java.type:"void" -> {
                %2 : Var<java.type:"java.util.List<java.util.List<java.lang.String>>"> = var %1 @"ll";
                java.enhancedFor
                    ()java.type:"java.util.List<java.util.List<java.lang.String>>" -> {
                        %3 : java.type:"java.util.List<java.util.List<java.lang.String>>" = var.load %2;
                        yield %3;
                    }
                    (%4 : java.type:"java.util.List<java.lang.String>")Var<java.type:"java.util.List<java.lang.String>"> -> {
                        %5 : Var<java.type:"java.util.List<java.lang.String>"> = var %4 @"l";
                        yield %5;
                    }
                    (%6 : Var<java.type:"java.util.List<java.lang.String>">)java.type:"void" -> {
                        java.enhancedFor
                            ()java.type:"java.util.List<java.lang.String>" -> {
                                %7 : java.type:"java.util.List<java.lang.String>" = var.load %6;
                                yield %7;
                            }
                            (%8 : java.type:"java.lang.String")Var<java.type:"java.lang.String"> -> {
                                %9 : Var<java.type:"java.lang.String"> = var %8 @"s";
                                yield %9;
                            }
                            (%10 : Var<java.type:"java.lang.String">)java.type:"void" -> {
                                %11 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                                %12 : java.type:"java.lang.String" = var.load %10;
                                invoke %11 %12 @"java.io.PrintStream::println(java.lang.String):void";
                                java.continue;
                            };
                        java.continue;
                    };
                return;
            };
            """)
    void test1(List<List<String>> ll) {
        for (List<String> l : ll) {
            for (String s : l) {
                System.out.println(s);
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"ForLoopTest", %1 : java.type:"java.util.List<java.lang.String>")java.type:"void" -> {
                %2 : Var<java.type:"java.util.List<java.lang.String>"> = var %1 @"l";
                java.enhancedFor
                    ()java.type:"java.util.List<java.lang.String>" -> {
                        %3 : java.type:"java.util.List<java.lang.String>" = var.load %2;
                        %4 : java.type:"java.util.stream.Stream<java.lang.String>" = invoke %3 @"java.util.List::stream():java.util.stream.Stream";
                        %5 : java.type:"java.util.function.Predicate<java.lang.String>" = lambda (%6 : java.type:"java.lang.String")java.type:"boolean" -> {
                            %7 : Var<java.type:"java.lang.String"> = var %6 @"s";
                            %8 : java.type:"java.lang.String" = var.load %7;
                            %9 : java.type:"int" = invoke %8 @"java.lang.String::length():int";
                            %10 : java.type:"int" = constant @"10";
                            %11 : java.type:"boolean" = lt %9 %10;
                            return %11;
                        };
                        %12 : java.type:"java.util.stream.Stream<java.lang.String>" = invoke %4 %5 @"java.util.stream.Stream::filter(java.util.function.Predicate):java.util.stream.Stream";
                        %13 : java.type:"java.util.List<java.lang.String>" = invoke %12 @"java.util.stream.Stream::toList():java.util.List";
                        yield %13;
                    }
                    (%14 : java.type:"java.lang.String")Var<java.type:"java.lang.String"> -> {
                        %15 : Var<java.type:"java.lang.String"> = var %14 @"s";
                        yield %15;
                    }
                    (%16 : Var<java.type:"java.lang.String">)java.type:"void" -> {
                        %17 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %18 : java.type:"java.lang.String" = var.load %16;
                        invoke %17 %18 @"java.io.PrintStream::println(java.lang.String):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test2(List<String> l) {
        for (String s : l.stream().filter(s -> s.length() < 10).toList()) {
            System.out.println(s);
        }
    }

    @CodeReflection
    @IR("""
            func @"test2_1" (%0 : java.type:"ForLoopTest", %1 : java.type:"java.util.List<java.lang.String>")java.type:"void" -> {
                %2 : Var<java.type:"java.util.List<java.lang.String>"> = var %1 @"l";
                java.enhancedFor
                    ()java.type:"java.util.List<java.lang.String>" -> {
                        %3 : java.type:"java.util.List<java.lang.String>" = var.load %2;
                        yield %3;
                    }
                    (%4 : java.type:"java.lang.String")Var<java.type:"java.lang.String"> -> {
                        %5 : Var<java.type:"java.lang.String"> = var %4 @"s";
                        yield %5;
                    }
                    (%6 : Var<java.type:"java.lang.String">)java.type:"void" -> {
                        java.continue;
                    };
                return;
            };
            """)
    void test2_1(List<String> l) {
        for (String s : l);
    }

    @CodeReflection
    @IR("""
            func @"test2_2" (%0 : java.type:"ForLoopTest", %1 : java.type:"java.util.List<java.lang.String>")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.util.List<java.lang.String>"> = var %1 @"l";
                java.enhancedFor
                    ()java.type:"java.util.List<java.lang.String>" -> {
                        %3 : java.type:"java.util.List<java.lang.String>" = var.load %2;
                        yield %3;
                    }
                    (%4 : java.type:"java.lang.String")Var<java.type:"java.lang.String"> -> {
                        %5 : Var<java.type:"java.lang.String"> = var %4 @"s";
                        yield %5;
                    }
                    (%6 : Var<java.type:"java.lang.String">)java.type:"void" -> {
                        %7 : java.type:"java.lang.String" = var.load %6;
                        return %7;
                    };
                %8 : java.type:"java.lang.String" = constant @"";
                return %8;
            };
            """)
    String test2_2(List<String> l) {
        for (String s : l) {
            return s;
        }
        return "";
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %3;
                        %5 : java.type:"int" = constant @"10";
                        %6 : java.type:"boolean" = lt %4 %5;
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"void" -> {
                        %8 : java.type:"int" = var.load %7;
                        %9 : java.type:"int" = constant @"1";
                        %10 : java.type:"int" = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<java.type:"int">)java.type:"void" -> {
                        %12 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %13 : java.type:"int" = var.load %11;
                        invoke %12 %13 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test3() {
        for (int i = 0; i < 10; i++) {
            System.out.println(i);
        }
    }

    @CodeReflection
    @IR("""
            func @"test3_1" (%0 : java.type:"ForLoopTest")java.type:"int" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %3;
                        %5 : java.type:"int" = constant @"10";
                        %6 : java.type:"boolean" = lt %4 %5;
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"void" -> {
                        %8 : java.type:"int" = var.load %7;
                        %9 : java.type:"int" = constant @"1";
                        %10 : java.type:"int" = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<java.type:"int">)java.type:"void" -> {
                        %12 : java.type:"int" = var.load %11;
                        return %12;
                    };
                %13 : java.type:"int" = constant @"-1";
                return %13;
            };
            """)
    int test3_1() {
        for (int i = 0; i < 10; i++) {
            return i;
        }
        return -1;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %3;
                        %5 : java.type:"int" = constant @"10";
                        %6 : java.type:"boolean" = lt %4 %5;
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"void" -> {
                        %8 : java.type:"int" = var.load %7;
                        %9 : java.type:"int" = constant @"1";
                        %10 : java.type:"int" = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<java.type:"int">)java.type:"void" -> {
                        %12 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %13 : java.type:"int" = var.load %11;
                        invoke %12 %13 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test4() {
        for (int i = 0; i < 10; i = i + 1)
            System.out.println(i);
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %3;
                        %5 : java.type:"int" = constant @"10";
                        %6 : java.type:"boolean" = lt %4 %5;
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"void" -> {
                        %8 : java.type:"int" = var.load %7;
                        %9 : java.type:"int" = constant @"1";
                        %10 : java.type:"int" = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<java.type:"int">)java.type:"void" -> {
                        java.continue;
                    };
                return;
            };
            """)
    void test5() {
        for (int i = 0; i < 10; i = i + 1);
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @"0";
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.for
                    ()java.type:"void" -> {
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"10";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = var.load %2;
                        %7 : java.type:"int" = constant @"1";
                        %8 : java.type:"int" = add %6 %7;
                        var.store %2 %8;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %10 : java.type:"int" = var.load %2;
                        invoke %9 %10 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test6() {
        int i = 0;
        for (; i < 10; i = i + 1) {
            System.out.println(i);
        }
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @"0";
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.for
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"int" = add %3 %4;
                        var.store %2 %5;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %6 : java.type:"int" = var.load %2;
                        %7 : java.type:"int" = constant @"10";
                        %8 : java.type:"boolean" = lt %6 %7;
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"int" = var.load %2;
                        %10 : java.type:"int" = constant @"1";
                        %11 : java.type:"int" = add %9 %10;
                        var.store %2 %11;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %13 : java.type:"int" = var.load %2;
                        invoke %12 %13 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test7() {
        int i = 0;
        for (i = i + 1; i < 10; i = i + 1) {
            System.out.println(i);
        }
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"boolean" = constant @"true";
                        yield %4;
                    }
                    (%5 : Var<java.type:"int">)java.type:"void" -> {
                        %6 : java.type:"int" = var.load %5;
                        %7 : java.type:"int" = constant @"1";
                        %8 : java.type:"int" = add %6 %7;
                        var.store %5 %8;
                        yield;
                    }
                    (%9 : Var<java.type:"int">)java.type:"void" -> {
                        %10 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %11 : java.type:"int" = var.load %9;
                        invoke %10 %11 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                unreachable;
            };
            """)
    void test8() {
        for (int i = 0; ; i = i + 1) {
            System.out.println(i);
        }
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"boolean" = constant @"true";
                        yield %4;
                    }
                    (%5 : Var<java.type:"int">)java.type:"void" -> {
                        yield;
                    }
                    (%6 : Var<java.type:"int">)java.type:"void" -> {
                        %7 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %8 : java.type:"int" = var.load %6;
                        invoke %7 %8 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                unreachable;
            };
            """)
    void test9() {
        for (int i = 0; ; ) {
            System.out.println(i);
        }
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()java.type:"void" -> {
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %1 : java.type:"boolean" = constant @"true";
                        yield %1;
                    }
                    ()java.type:"void" -> {
                        yield;
                    }
                    ()java.type:"void" -> {
                        java.continue;
                    };
                unreachable;
            };
            """)
    void test10() {
        for (; ; ) {
        }
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : java.type:"ForLoopTest")java.type:"void" -> {
                java.for
                    ()Tuple<Var<java.type:"int">, Var<java.type:"int">> -> {
                        %1 : java.type:"int" = constant @"0";
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        %3 : java.type:"int" = constant @"0";
                        %4 : Var<java.type:"int"> = var %3 @"j";
                        %5 : Tuple<Var<java.type:"int">, Var<java.type:"int">> = tuple %2 %4;
                        yield %5;
                    }
                    (%6 : Var<java.type:"int">, %7 : Var<java.type:"int">)java.type:"boolean" -> {
                        %8 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %9 : java.type:"int" = var.load %6;
                                %10 : java.type:"int" = constant @"10";
                                %11 : java.type:"boolean" = lt %9 %10;
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %12 : java.type:"int" = var.load %7;
                                %13 : java.type:"int" = constant @"20";
                                %14 : java.type:"boolean" = lt %12 %13;
                                yield %14;
                            };
                        yield %8;
                    }
                    (%15 : Var<java.type:"int">, %16 : Var<java.type:"int">)java.type:"void" -> {
                        %17 : java.type:"int" = var.load %15;
                        %18 : java.type:"int" = constant @"1";
                        %19 : java.type:"int" = add %17 %18;
                        var.store %15 %19;
                        %20 : java.type:"int" = var.load %16;
                        %21 : java.type:"int" = constant @"2";
                        %22 : java.type:"int" = add %20 %21;
                        var.store %16 %22;
                        yield;
                    }
                    (%23 : Var<java.type:"int">, %24 : Var<java.type:"int">)java.type:"void" -> {
                        %25 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %26 : java.type:"int" = var.load %23;
                        invoke %25 %26 @"java.io.PrintStream::println(int):void";
                        %27 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %28 : java.type:"int" = var.load %24;
                        invoke %27 %28 @"java.io.PrintStream::println(int):void";
                        java.continue;
                    };
                return;
            };
            """)
    void test11() {
        for (int i = 0, j = 0; i < 10 && j < 20; i = i + 1, j = j + 2) {
            System.out.println(i);
            System.out.println(j);
        }
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : java.type:"ForLoopTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"r";
                java.for
                    ()Var<java.type:"int"> -> {
                        %3 : java.type:"int" = constant @"0";
                        %4 : Var<java.type:"int"> = var %3 @"i";
                        yield %4;
                    }
                    (%5 : Var<java.type:"int">)java.type:"boolean" -> {
                        %6 : java.type:"int" = var.load %5;
                        %7 : java.type:"int" = constant @"10";
                        %8 : java.type:"boolean" = lt %6 %7;
                        yield %8;
                    }
                    (%9 : Var<java.type:"int">)java.type:"void" -> {
                        %10 : java.type:"int" = var.load %9;
                        %11 : java.type:"int" = constant @"1";
                        %12 : java.type:"int" = add %10 %11;
                        var.store %9 %12;
                        yield;
                    }
                    (%13 : Var<java.type:"int">)java.type:"void" -> {
                        java.if
                            ()java.type:"boolean" -> {
                                %14 : java.type:"int" = var.load %2;
                                %15 : java.type:"int" = constant @"0";
                                %16 : java.type:"boolean" = eq %14 %15;
                                yield %16;
                            }
                            ()java.type:"void" -> {
                                java.break;
                            }
                            ()java.type:"boolean" -> {
                                %17 : java.type:"int" = var.load %2;
                                %18 : java.type:"int" = constant @"1";
                                %19 : java.type:"boolean" = eq %17 %18;
                                yield %19;
                            }
                            ()java.type:"void" -> {
                                java.continue;
                            }
                            ()java.type:"void" -> {
                                yield;
                            };
                        java.continue;
                    };
                return;
            };
            """)
    void test12(int r) {
        for (int i = 0; i < 10; i++) {
            if (r == 0) {
                break;
            } else if (r == 1) {
                continue;
            }
        }
    }

}
