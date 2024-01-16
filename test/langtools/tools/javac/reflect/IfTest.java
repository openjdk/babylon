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
 * @summary Smoke test for code reflection with if statements.
 * @build IfTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester IfTest
 */

public class IfTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : IfTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                java.if
                    ()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^then()void -> {
                        %6 : int = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ^else()void -> {
                        yield;
                    };
                return;
            };
            """)
    void test1(int i) {
        if (i < 1) {
            i = 1;
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : IfTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                java.if
                    ()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^then()void -> {
                        %6 : int = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ^else()void -> {
                        %7 : int = constant @"2";
                        var.store %2 %7;
                        yield;
                    };
                return;
            };
            """)
    void test2(int i) {
        if (i < 1) {
            i = 1;
        } else {
            i = 2;
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : IfTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                java.if
                    ()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^then()void -> {
                        %6 : int = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ^else_if()boolean -> {
                        %7 : int = var.load %2;
                        %8 : int = constant @"2";
                        %9 : boolean = lt %7 %8;
                        yield %9;
                    }
                    ^then()void -> {
                        %10 : int = constant @"2";
                        var.store %2 %10;
                        yield;
                    }
                    ^else()void -> {
                        yield;
                    };
                return;
            };
            """)
    void test3(int i) {
        if (i < 1) {
            i = 1;
        } else if (i < 2) {
            i = 2;
        }
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : IfTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                java.if
                    ()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^then()void -> {
                        %6 : int = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ^else_if()boolean -> {
                        %7 : int = var.load %2;
                        %8 : int = constant @"2";
                        %9 : boolean = lt %7 %8;
                        yield %9;
                    }
                    ^then()void -> {
                        %10 : int = constant @"2";
                        var.store %2 %10;
                        yield;
                    }
                    ^else()void -> {
                        %11 : int = constant @"3";
                        var.store %2 %11;
                        yield;
                    };
                return;
            };
            """)
    void test4(int i) {
        if (i < 1) {
            i = 1;
        } else if (i < 2) {
            i = 2;
        } else {
            i = 3;
        }
    }

    @IR("""
            func @"test5" (%0 : IfTest, %1 : int)int -> {
              %2 : Var<int> = var %1 @"i";
              java.if
                  ()boolean -> {
                      %3 : int = var.load %2;
                      %4 : int = constant @"1";
                      %5 : boolean = lt %3 %4;
                      yield %5;
                  }
                  ^then()void -> {
                      %6 : int = constant @"1";
                      return %6;
                  }
                  ^else_if()boolean -> {
                      %7 : int = var.load %2;
                      %8 : int = constant @"2";
                      %9 : boolean = lt %7 %8;
                      yield %9;
                  }
                  ^then()void -> {
                      %10 : int = constant @"2";
                      return %10;
                  }
                  ^else()void -> {
                      %11 : int = constant @"3";
                      return %11;
                  };
              return;
            };
            """)
    @CodeReflection
    int test5(int i) {
        if (i < 1) {
            return 1;
        } else if (i < 2) {
            return 2;
        } else {
            return 3;
        }
    }
}
