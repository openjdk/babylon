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
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with conditional expressions.
 * @modules jdk.incubator.code
 * @build ConditionalExpressionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ConditionalExpressionTest
 */

public class ConditionalExpressionTest {

    @IR("""
            func @"test1" (%0 : java.type:"ConditionalExpressionTest", %1 : java.type:"boolean", %2 : java.type:"int", %3 : java.type:"int")java.type:"void" -> {
                %4 : Var<java.type:"boolean"> = var %1 @"b";
                %5 : Var<java.type:"int"> = var %2 @"x";
                %6 : Var<java.type:"int"> = var %3 @"y";
                %7 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = var.load %4;
                        yield %8;
                    }
                    ()java.type:"int" -> {
                        %9 : java.type:"int" = var.load %5;
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"int" = var.load %6;
                        yield %10;
                    };
                %11 : Var<java.type:"int"> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test1(boolean b, int x, int y) {
        int z = b ? x : y;
    }

    @IR("""
            func @"test2" (%0 : java.type:"ConditionalExpressionTest", %1 : java.type:"boolean", %2 : java.type:"int", %3 : java.type:"double")java.type:"void" -> {
                %4 : Var<java.type:"boolean"> = var %1 @"b";
                %5 : Var<java.type:"int"> = var %2 @"x";
                %6 : Var<java.type:"double"> = var %3 @"y";
                %7 : java.type:"double" = java.cexpression
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = var.load %4;
                        %9 : java.type:"boolean" = not %8;
                        yield %9;
                    }
                    ()java.type:"double" -> {
                        %10 : java.type:"int" = var.load %5;
                        %11 : java.type:"double" = conv %10;
                        yield %11;
                    }
                    ()java.type:"double" -> {
                        %12 : java.type:"double" = var.load %6;
                        yield %12;
                    };
                %13 : Var<java.type:"double"> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test2(boolean b, int x, double y) {
        double z = !b ? x : y;
    }

    @IR("""
            func @"test3" (%0 : java.type:"ConditionalExpressionTest", %1 : java.type:"boolean", %2 : java.type:"int", %3 : java.type:"double")java.type:"void" -> {
                %4 : Var<java.type:"boolean"> = var %1 @"b";
                %5 : Var<java.type:"int"> = var %2 @"x";
                %6 : Var<java.type:"double"> = var %3 @"y";
                %7 : java.type:"java.util.function.Supplier<java.lang.Double>" = java.cexpression
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = var.load %4;
                        yield %8;
                    }
                    ()java.type:"java.util.function.Supplier<java.lang.Double>" -> {
                        %9 : java.type:"java.util.function.Supplier<java.lang.Double>" = lambda ()java.type:"java.lang.Double" -> {
                            %10 : java.type:"int" = var.load %5;
                            %11 : java.type:"double" = conv %10;
                            %12 : java.type:"java.lang.Double" = invoke %11 @java.ref:"java.lang.Double::valueOf(double):java.lang.Double";
                            return %12;
                        };
                        yield %9;
                    }
                    ()java.type:"java.util.function.Supplier<java.lang.Double>" -> {
                        %13 : java.type:"java.util.function.Supplier<java.lang.Double>" = lambda ()java.type:"java.lang.Double" -> {
                            %14 : java.type:"double" = var.load %6;
                            %15 : java.type:"java.lang.Double" = invoke %14 @java.ref:"java.lang.Double::valueOf(double):java.lang.Double";
                            return %15;
                        };
                        yield %13;
                    };
                %16 : Var<java.type:"java.util.function.Supplier<java.lang.Double>"> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test3(boolean b, int x, double y) {
        Supplier<Double> z = b ? () -> (double) x : () -> y;
    }

    @IR("""
            func @"test4" (%0 : java.type:"ConditionalExpressionTest", %1 : java.type:"boolean", %2 : java.type:"boolean", %3 : java.type:"int", %4 : java.type:"double", %5 : java.type:"double")java.type:"void" -> {
                %6 : Var<java.type:"boolean"> = var %1 @"b1";
                %7 : Var<java.type:"boolean"> = var %2 @"b2";
                %8 : Var<java.type:"int"> = var %3 @"x";
                %9 : Var<java.type:"double"> = var %4 @"y";
                %10 : Var<java.type:"double"> = var %5 @"z";
                %11 : java.type:"double" = java.cexpression
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = var.load %6;
                        yield %12;
                    }
                    ()java.type:"double" -> {
                        %13 : java.type:"double" = java.cexpression
                            ()java.type:"boolean" -> {
                                %14 : java.type:"boolean" = var.load %7;
                                yield %14;
                            }
                            ()java.type:"double" -> {
                                %15 : java.type:"int" = var.load %8;
                                %16 : java.type:"double" = conv %15;
                                yield %16;
                            }
                            ()java.type:"double" -> {
                                %17 : java.type:"double" = var.load %9;
                                yield %17;
                            };
                        yield %13;
                    }
                    ()java.type:"double" -> {
                        %18 : java.type:"double" = var.load %10;
                        yield %18;
                    };
                %19 : Var<java.type:"double"> = var %11 @"r";
                return;
            };
            """)
    @CodeReflection
    void test4(boolean b1, boolean b2, int x, double y, double z) {
        double r = b1 ? (b2 ? x : y) : z;
    }

}