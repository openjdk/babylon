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
 * @summary Smoke test for code reflection with while loops.
 * @modules jdk.incubator.code
 * @build WhileLoopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester WhileLoopTest
 */

public class WhileLoopTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"WhileLoopTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @"0";
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.while
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"10";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %7 : java.type:"int" = var.load %2;
                        invoke %6 %7 @"java.io.PrintStream::println(int):void";
                        %8 : java.type:"int" = var.load %2;
                        %9 : java.type:"int" = constant @"1";
                        %10 : java.type:"int" = add %8 %9;
                        var.store %2 %10;
                        java.continue;
                    };
                return;
            };
            """)
    void test1() {
        int i = 0;
        while (i < 10) {
            System.out.println(i);
            i = i + 1;
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"WhileLoopTest")java.type:"int" -> {
                %1 : java.type:"int" = constant @"0";
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.while
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"10";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = var.load %2;
                        return %6;
                    };
                %7 : java.type:"int" = constant @"-1";
                return %7;
            };
            """)
    int test2() {
        int i = 0;
        while (i < 10) {
            return i;
        }
        return -1;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"WhileLoopTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @"0";
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.do.while
                    ()java.type:"void" -> {
                        %3 : java.type:"java.io.PrintStream" = field.load @"java.lang.System::out:java.io.PrintStream";
                        %4 : java.type:"int" = var.load %2;
                        invoke %3 %4 @"java.io.PrintStream::println(int):void";
                        %5 : java.type:"int" = var.load %2;
                        %6 : java.type:"int" = constant @"1";
                        %7 : java.type:"int" = add %5 %6;
                        var.store %2 %7;
                        java.continue;
                    }
                    ()java.type:"boolean" -> {
                        %8 : java.type:"int" = var.load %2;
                        %9 : java.type:"int" = constant @"10";
                        %10 : java.type:"boolean" = lt %8 %9;
                        yield %10;
                    };
                return;
            };
            """)
    void test3() {
        int i = 0;
        do {
            System.out.println(i);
            i = i + 1;
        } while (i < 10);
    }


    @IR("""
            func @"test4" ()java.type:"void" -> {
                %0 : java.type:"boolean" = constant @"true";
                %1 : java.type:"java.lang.Boolean" = invoke %0 @"java.lang.Boolean::valueOf(boolean):java.lang.Boolean";
                %2 : Var<java.type:"java.lang.Boolean"> = var %1 @"b";
                %3 : java.type:"int" = constant @"0";
                %4 : Var<java.type:"int"> = var %3 @"i";
                java.while
                    ()java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Boolean" = var.load %2;
                        %6 : java.type:"boolean" = invoke %5 @"java.lang.Boolean::booleanValue():boolean";
                        yield %6;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = var.load %4;
                        %8 : java.type:"int" = constant @"1";
                        %9 : java.type:"int" = add %7 %8;
                        var.store %4 %9;
                        %10 : java.type:"int" = var.load %4;
                        %11 : java.type:"int" = constant @"10";
                        %12 : java.type:"boolean" = lt %10 %11;
                        %13 : java.type:"java.lang.Boolean" = invoke %12 @"java.lang.Boolean::valueOf(boolean):java.lang.Boolean";
                        var.store %2 %13;
                        java.continue;
                    };
                return;
            };
            """)
    @CodeReflection
    static void test4() {
        Boolean b = true;
        int i = 0;
        while (b) {
            i++;
            b = i < 10;
        }
    }

    @IR("""
            func @"test5" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : Var<java.type:"java.lang.Boolean"> = var @"b";
                java.do.while
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = var.load %1;
                        %4 : java.type:"int" = constant @"10";
                        %5 : java.type:"boolean" = lt %3 %4;
                        %6 : java.type:"java.lang.Boolean" = invoke %5 @"java.lang.Boolean::valueOf(boolean):java.lang.Boolean";
                        var.store %2 %6;
                        java.continue;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"java.lang.Boolean" = var.load %2;
                        %8 : java.type:"boolean" = invoke %7 @"java.lang.Boolean::booleanValue():boolean";
                        yield %8;
                    };
                return;
            };
            """)
    @CodeReflection
    static void test5(int i) {
        Boolean b;
        do {
            b = i < 10;
        } while (b);
    }
}
