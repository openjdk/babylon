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
 * @summary Smoke test for code reflection with if statements.
 * @modules jdk.incubator.code
 * @build IfTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester IfTest
 */

public class IfTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"void" -> {
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
            func @"test2" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = constant @"2";
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
            func @"test3" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"2";
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"int" = constant @"2";
                        var.store %2 %10;
                        yield;
                    }
                    ()java.type:"void" -> {
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
            func @"test4" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"2";
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"int" = constant @"2";
                        var.store %2 %10;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"int" = constant @"3";
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
            func @"test5" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        return %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"2";
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"int" = constant @"2";
                        return %10;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"int" = constant @"3";
                        return %11;
                    };
                unreachable;
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

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"void" -> {
                        yield;
                    };
                return;
            };
            """)
    void test6(int i) {
        if (i < 1)
            i = 1;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = constant @"2";
                        var.store %2 %7;
                        yield;
                    };
                return;
            };
            """)
    void test7(int i) {
        if (i < 1)
            i = 1;
        else
            i = 2;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"IfTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"int" = var.load %2;
                        %4 : java.type:"int" = constant @"1";
                        %5 : java.type:"boolean" = lt %3 %4;
                        yield %5;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"1";
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"2";
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"int" = constant @"2";
                        var.store %2 %10;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"int" = constant @"3";
                        var.store %2 %11;
                        yield;
                    };
                return;
            };
            """)
    void test8(int i) {
        if (i < 1)
            i = 1;
        else if (i < 2)
            i = 2;
        else
            i = 3;
    }

    @IR("""
            func @"test9" (%0 : java.type:"java.lang.Boolean")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Boolean"> = var %0 @"b";
                %2 : Var<java.type:"int"> = var @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %3 : java.type:"java.lang.Boolean" = var.load %1;
                        %4 : java.type:"boolean" = invoke %3 @"java.lang.Boolean::booleanValue():boolean";
                        yield %4;
                    }
                    ()java.type:"void" -> {
                        %5 : java.type:"int" = constant @"1";
                        var.store %2 %5;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %6 : java.type:"int" = constant @"2";
                        var.store %2 %6;
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    static void test9(Boolean b) {
        int i;
        if (b) {
            i = 1;
        } else {
            i = 2;
        }
    }
}
