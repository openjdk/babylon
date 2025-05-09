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
 * @summary Smoke test for code reflection with conditional and/or expressions.
 * @modules jdk.incubator.code
 * @build ConditionalAndOrTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ConditionalAndOrTest
 */

public class ConditionalAndOrTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cand
                    ()java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %2;
                        %5 : java.type:"int" = constant @"1";
                        %6 : java.type:"boolean" = gt %4 %5;
                        yield %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"10";
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

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %2;
                        %5 : java.type:"int" = constant @"1";
                        %6 : java.type:"boolean" = gt %4 %5;
                        yield %6;
                    }
                    ()java.type:"boolean" -> {
                        %7 : java.type:"int" = var.load %2;
                        %8 : java.type:"int" = constant @"10";
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

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"ConditionalAndOrTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"boolean" = java.cor
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %5 : java.type:"int" = var.load %2;
                                %6 : java.type:"int" = constant @"1";
                                %7 : java.type:"boolean" = gt %5 %6;
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"int" = var.load %2;
                                %9 : java.type:"int" = constant @"10";
                                %10 : java.type:"boolean" = lt %8 %9;
                                yield %10;
                            };
                        yield %4;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"int" = var.load %2;
                        %12 : java.type:"int" = constant @"100";
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
}
