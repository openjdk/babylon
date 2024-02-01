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
import java.util.List;
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with conditional expressions.
 * @build ConditionalExpressionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ConditionalExpressionTest
 */

public class ConditionalExpressionTest {

    @IR("""
            func @"test1" (%0 : ConditionalExpressionTest, %1 : boolean, %2 : int, %3 : int)void -> {
                %4 : Var<boolean> = var %1 @"b";
                %5 : Var<int> = var %2 @"x";
                %6 : Var<int> = var %3 @"y";
                %7 : int = java.cexpression
                    ^cond()boolean -> {
                        %8 : boolean = var.load %4;
                        yield %8;
                    }
                    ^truepart()int -> {
                        %9 : int = var.load %5;
                        yield %9;
                    }
                    ^falsepart()int -> {
                        %10 : int = var.load %6;
                        yield %10;
                    };
                %11 : Var<int> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test1(boolean b, int x, int y) {
        int z = b ? x : y;
    }

    @IR("""
            func @"test2" (%0 : ConditionalExpressionTest, %1 : boolean, %2 : int, %3 : double)void -> {
                %4 : Var<boolean> = var %1 @"b";
                %5 : Var<int> = var %2 @"x";
                %6 : Var<double> = var %3 @"y";
                %7 : double = java.cexpression
                    ^cond()boolean -> {
                        %8 : boolean = var.load %4;
                        %9 : boolean = not %8;
                        yield %9;
                    }
                    ^truepart()double -> {
                        %10 : int = var.load %5;
                        %11 : double = conv %10;
                        yield %11;
                    }
                    ^falsepart()double -> {
                        %12 : double = var.load %6;
                        yield %12;
                    };
                %13 : Var<double> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test2(boolean b, int x, double y) {
        double z = !b ? x : y;
    }

    @IR("""
            func @"test3" (%0 : ConditionalExpressionTest, %1 : boolean, %2 : int, %3 : double)void -> {
                %4 : Var<boolean> = var %1 @"b";
                %5 : Var<int> = var %2 @"x";
                %6 : Var<double> = var %3 @"y";
                %7 : java.util.function.Supplier<java.lang.Double> = java.cexpression
                    ^cond()boolean -> {
                        %8 : boolean = var.load %4;
                        yield %8;
                    }
                    ^truepart()java.util.function.Supplier<java.lang.Double> -> {
                        %9 : java.util.function.Supplier<java.lang.Double> = lambda ()java.lang.Double -> {
                            %10 : int = var.load %5;
                            %11 : double = conv %10;
                            %12 : java.lang.Double = invoke %11 @"java.lang.Double::valueOf(double)java.lang.Double";
                            return %12;
                        };
                        yield %9;
                    }
                    ^falsepart()java.util.function.Supplier<java.lang.Double> -> {
                        %13 : java.util.function.Supplier<java.lang.Double> = lambda ()java.lang.Double -> {
                            %14 : double = var.load %6;
                            %15 : java.lang.Double = invoke %14 @"java.lang.Double::valueOf(double)java.lang.Double";
                            return %15;
                        };
                        yield %13;
                    };
                %16 : Var<java.util.function.Supplier<java.lang.Double>> = var %7 @"z";
                return;
            };
            """)
    @CodeReflection
    void test3(boolean b, int x, double y) {
        Supplier<Double> z = b ? () -> (double) x : () -> y;
    }

    @IR("""
            func @"test4" (%0 : ConditionalExpressionTest, %1 : boolean, %2 : boolean, %3 : int, %4 : double, %5 : double)void -> {
                %6 : Var<boolean> = var %1 @"b1";
                %7 : Var<boolean> = var %2 @"b2";
                %8 : Var<int> = var %3 @"x";
                %9 : Var<double> = var %4 @"y";
                %10 : Var<double> = var %5 @"z";
                %11 : double = java.cexpression
                    ^cond()boolean -> {
                        %12 : boolean = var.load %6;
                        yield %12;
                    }
                    ^truepart()double -> {
                        %13 : double = java.cexpression
                            ^cond()boolean -> {
                                %14 : boolean = var.load %7;
                                yield %14;
                            }
                            ^truepart()double -> {
                                %15 : int = var.load %8;
                                %16 : double = conv %15;
                                yield %16;
                            }
                            ^falsepart()double -> {
                                %17 : double = var.load %9;
                                yield %17;
                            };
                        yield %13;
                    }
                    ^falsepart()double -> {
                        %18 : double = var.load %10;
                        yield %18;
                    };
                %19 : Var<double> = var %11 @"r";
                return;
            };
            """)
    @CodeReflection
    void test4(boolean b1, boolean b2, int x, double y, double z) {
        double r = b1 ? (b2 ? x : y) : z;
    }

}