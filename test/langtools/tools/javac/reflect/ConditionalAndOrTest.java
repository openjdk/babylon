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
 * @summary Smoke test for code reflection with conditional and/or expressions.
 * @modules jdk.incubator.code
 * @build ConditionalAndOrTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ConditionalAndOrTest
 */

public class ConditionalAndOrTest {

    @Reflect
    @IR("""
            func @"test1" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %2;
                        %5 : java.type:"int" = constant @1;
                        %6 : java.type:"boolean" = gt %4 %5;
                        yield %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @10;
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    };
                %10 : Var<java.type:"boolean"> = var %3 @"b";
                return;
            };
            """)
    void test1(int i) {
        boolean b = i > 1 && i < 10;
    }

    @Reflect
    @IR("""
            func @"test2" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %2;
                        %5 : java.type:"int" = constant @1;
                        %6 : java.type:"boolean" = gt %4 %5;
                        yield %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @10;
                        %9 : java.type:"boolean" = lt %7 %8;
                        yield %9;
                    };
                %10 : Var<java.type:"boolean"> = var %3 @"b";
                return;
            };
            """)
    void test2(int i) {
        boolean b = i > 1 || i < 10;
    }

    @Reflect
    @IR("""
            func @"test3" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %5 : java.type:"int" = var.load %2;
                                %6 : java.type:"int" = constant @1;
                                %7 : java.type:"boolean" = gt %5 %6;
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"int" = var.load %2;
                                %9 : java.type:"int" = constant @10;
                                %10 : java.type:"boolean" = lt %8 %9;
                                yield %10;
                            };
                        yield %4;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"int" = var.load %2;
                        %12 : java.type:"int" = constant @100;
                        %13 : java.type:"boolean" = eq %11 %12;
                        yield %13;
                    };
                %14 : Var<java.type:"boolean"> = var %3 @"b";
                return;
            };
            """)
    void test3(int i) {
        boolean b = i > 1 && i < 10 || i == 100;
    }

    @Reflect
    @IR("""
            func @"test4" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"java.lang.Boolean", %2 : java.type:"int", %3 : java.type:"int")java.type:"int" -> {
                %4 : Var<java.type:"java.lang.Boolean"> = var %1 @"c";
                %5 : Var<java.type:"int"> = var %2 @"i1";
                %6 : Var<java.type:"int"> = var %3 @"i2";
                %7 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %8 : java.type:"java.lang.Boolean" = var.load %4;
                        %9 : java.type:"boolean" = invoke %8 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"int" = var.load %5;
                        yield %10;
                    }
                    ()java.type:"int" -> {
                        %11 : java.type:"int" = var.load %6;
                        yield %11;
                    };
                return %7;
            };
            """)
    int test4(Boolean c, int i1, int i2) {
        return c ? i1 : i2;
    }
}
