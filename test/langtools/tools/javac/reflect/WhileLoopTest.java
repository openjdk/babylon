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
            func @"test1" (%0 : WhileLoopTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.while
                    ^cond()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"10";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^body()void -> {
                        %6 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %7 : int = var.load %2;
                        invoke %6 %7 @"java.io.PrintStream::println(int)void";
                        %8 : int = var.load %2;
                        %9 : int = constant @"1";
                        %10 : int = add %8 %9;
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
            func @"test2" (%0 : WhileLoopTest)int -> {
              %1 : int = constant @"0";
              %2 : Var<int> = var %1 @"i";
              java.while
                  ^cond()boolean -> {
                      %3 : int = var.load %2;
                      %4 : int = constant @"10";
                      %5 : boolean = lt %3 %4;
                      yield %5;
                  }
                  ^body()void -> {
                      %6 : int = var.load %2;
                      return %6;
                  };
              %7 : int = constant @"-1";
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
            func @"test3" (%0 : WhileLoopTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.do.while
                    ^body()void -> {
                        %3 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %4 : int = var.load %2;
                        invoke %3 %4 @"java.io.PrintStream::println(int)void";
                        %5 : int = var.load %2;
                        %6 : int = constant @"1";
                        %7 : int = add %5 %6;
                        var.store %2 %7;
                        java.continue;
                    }
                    ^cond()boolean -> {
                        %8 : int = var.load %2;
                        %9 : int = constant @"10";
                        %10 : boolean = lt %8 %9;
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
            func @"test4" ()void -> {
                  %0 : boolean = constant @"true";
                  %1 : java.lang.Boolean = invoke %0 @"java.lang.Boolean::valueOf(boolean)java.lang.Boolean";
                  %2 : Var<java.lang.Boolean> = var %1 @"b";
                  %3 : int = constant @"0";
                  %4 : Var<int> = var %3 @"i";
                  java.while
                      ()boolean -> {
                          %5 : java.lang.Boolean = var.load %2;
                          %6 : boolean = invoke %5 @"java.lang.Boolean::booleanValue()boolean";
                          yield %6;
                      }
                      ()void -> {
                          %7 : int = var.load %4;
                          %8 : int = constant @"1";
                          %9 : int = add %7 %8;
                          var.store %4 %9;
                          %10 : int = var.load %4;
                          %11 : int = constant @"10";
                          %12 : boolean = lt %10 %11;
                          %13 : java.lang.Boolean = invoke %12 @"java.lang.Boolean::valueOf(boolean)java.lang.Boolean";
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
            func @"test5" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"i";
                %3 : Var<java.lang.Boolean> = var @"b";
                java.do.while
                    ()void -> {
                        %4 : int = var.load %1;
                        %5 : int = constant @"10";
                        %6 : boolean = lt %4 %5;
                        %7 : java.lang.Boolean = invoke %6 @"java.lang.Boolean::valueOf(boolean)java.lang.Boolean";
                        var.store %3 %7;
                        java.continue;
                    }
                    ()boolean -> {
                        %8 : java.lang.Boolean = var.load %3;
                        %9 : boolean = invoke %8 @"java.lang.Boolean::booleanValue()boolean";
                        yield %9;
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
