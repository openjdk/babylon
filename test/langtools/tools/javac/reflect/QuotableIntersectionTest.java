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

/*
 * @test
 * @summary Smoke test for code reflection with quotable lambdas.
 * @modules jdk.incubator.code
 * @build QuotableIntersectionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotableIntersectionTest
 */

import jdk.incubator.code.Quotable;
import jdk.incubator.code.CodeReflection;
import java.util.function.IntBinaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

public class QuotableIntersectionTest {
    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda ()java.type:"void" -> {
                    return;
                };
                return;
            };
            """)
    static final Quotable QUOTED_NO_PARAM_VOID = (Quotable & Runnable) () -> {
    };

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda ()java.type:"int" -> {
                    %1 : java.type:"int" = constant @"1";
                    return %1;
                };
                return;
            };
            """)
    static final Quotable QUOTED_NO_PARAM_CONST = (Quotable & IntSupplier) () -> 1;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x";
                    %3 : java.type:"int" = var.load %2;
                    return %3;
                };
                return;
            };
            """)
    static final Quotable QUOTED_ID = (Quotable & IntUnaryOperator) x -> x;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda (%1 : java.type:"int", %2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %1 @"x";
                    %4 : Var<java.type:"int"> = var %2 @"y";
                    %5 : java.type:"int" = var.load %3;
                    %6 : java.type:"int" = var.load %4;
                    %7 : java.type:"int" = add %5 %6;
                    return %7;
                };
                return;
            };
            """)
    static final Quotable QUOTED_PLUS = (Quotable & IntBinaryOperator) (x, y) -> x + y;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda ()java.type:"void" -> {
                    %1 : java.type:"java.lang.AssertionError" = new @"java.lang.AssertionError::()";
                    throw %1;
                };
                return;
            };
            """)
    static final Quotable QUOTED_THROW_NO_PARAM = (Quotable & Runnable) () -> {
        throw new AssertionError();
    };

    @IR("""
            func @"f" (%0 : Var<java.type:"int">)java.type:"void" -> {
                %1 : java.type:"jdk.incubator.code.Quotable" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"y";
                    %4 : java.type:"int" = var.load %0;
                    %5 : java.type:"int" = var.load %3;
                    %6 : java.type:"int" = add %4 %5;
                    return %6;
                };
                return;
            };
            """)
    static final Quotable QUOTED_CAPTURE_PARAM = new Object() {
        Quotable captureContext(int x) {
            return (Quotable & IntUnaryOperator) y -> x + y;
        }
    }.captureContext(42);

    static class Context {
        int x, y;

        Quotable capture() {
            return (Quotable & IntUnaryOperator) z -> x + y + z;
        }
    }

    @IR("""
            func @"f" (%0 : java.type:"QuotableIntersectionTest$Context")java.type:"void" -> {
                %1 : java.type:"jdk.incubator.code.Quotable" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"z";
                    %4 : java.type:"int" = field.load %0 @"QuotableIntersectionTest$Context::x:int";
                    %5 : java.type:"int" = field.load %0 @"QuotableIntersectionTest$Context::y:int";
                    %6 : java.type:"int" = add %4 %5;
                    %7 : java.type:"int" = var.load %3;
                    %8 : java.type:"int" = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final Quotable QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"x";
                %2 : java.type:"java.util.function.IntUnaryOperator" = lambda (%3 : java.type:"int")java.type:"int" -> {
                    %4 : Var<java.type:"int"> = var %3 @"y";
                    %5 : java.type:"int" = var.load %1;
                    %6 : java.type:"int" = var.load %4;
                    %7 : java.type:"int" = add %5 %6;
                    return %7;
                };
                %8 : Var<java.type:"java.util.function.IntUnaryOperator"> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        IntUnaryOperator op = (IntUnaryOperator & Quotable) y -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : java.type:"QuotableIntersectionTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.IntUnaryOperator" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"z";
                    %4 : java.type:"int" = field.load %0 @"QuotableIntersectionTest::x:int";
                    %5 : java.type:"int" = field.load %0 @"QuotableIntersectionTest::y:int";
                    %6 : java.type:"int" = add %4 %5;
                    %7 : java.type:"int" = var.load %3;
                    %8 : java.type:"int" = add %6 %7;
                    return %8;
                };
                %9 : Var<java.type:"java.util.function.IntUnaryOperator"> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        IntUnaryOperator op = (IntUnaryOperator & Quotable) z -> x + y + z;
    }

    static void m() {
    }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda ()java.type:"void" -> {
                    invoke @"QuotableIntersectionTest::m():void";
                    return;
                };
                return;
            };
            """)
    static final Quotable QUOTED_NO_PARAM_VOID_REF = (Quotable & Runnable) QuotableIntersectionTest::m;

    static int g(int i) {
        return i;
    }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x$0";
                    %3 : java.type:"int" = var.load %2;
                    %4 : java.type:"int" = invoke %3 @"QuotableIntersectionTest::g(int):int";
                    return %4;
                };
                return;
            };
            """)
    static final Quotable QUOTED_INT_PARAM_INT_RET_REF = (Quotable & IntUnaryOperator) QuotableIntersectionTest::g;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"jdk.incubator.code.Quotable" = lambda (%1 : java.type:"int")java.type:"int[]" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x$0";
                    %3 : java.type:"int" = var.load %2;
                    %4 : java.type:"int[]" = new %3 @"int[]::(int)";
                    return %4;
                };
                return;
            };
            """)
    static final Quotable QUOTED_INT_PARAM_ARR_RET_REF = (Quotable & IntFunction<int[]>) int[]::new;

    static class ContextRef {
        int g(int i) {
            return i;
        }

        Quotable capture() {
            return (Quotable & IntUnaryOperator) this::g;
        }
    }

    @IR("""
            func @"f" (%0 : java.type:"QuotableIntersectionTest$ContextRef")java.type:"void" -> {
                %1 : java.type:"jdk.incubator.code.Quotable" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"x$0";
                    %4 : java.type:"int" = var.load %3;
                    %5 : java.type:"int" = invoke %0 %4 @"QuotableIntersectionTest$ContextRef::g(int):int";
                    return %5;
                };
                return;
            };
            """)
    static final Quotable QUOTED_CAPTURE_THIS_REF = new ContextRef().capture();
}
