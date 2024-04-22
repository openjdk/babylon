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
 * @build QuotableIntersectionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotableIntersectionTest
 */

import java.lang.reflect.code.Quotable;
import java.lang.runtime.CodeReflection;
import java.util.function.IntBinaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

public class QuotableIntersectionTest {
    @IR("""
            func @"f" ()void -> {
                  %0 : java.lang.reflect.code.Quotable = lambda ()void -> {
                      return;
                  };
                  return;
            };
            """)
    static final Quotable QUOTED_NO_PARAM_VOID = (Quotable & Runnable) () -> {
    };

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.Quotable = lambda ()int -> {
                    %2 : int = constant @"1";
                    return %2;
                };
                return;
            };
            """)
    static final Quotable QUOTED_NO_PARAM_CONST = (Quotable & IntSupplier) () -> 1;

    @IR("""
            func @"f" ()void -> {
                  %0 : java.lang.reflect.code.Quotable = lambda (%1 : int)int -> {
                      %2 : Var<int> = var %1 @"x";
                      %3 : int = var.load %2;
                      return %3;
                  };
                  return;
            };
            """)
    static final Quotable QUOTED_ID = (Quotable & IntUnaryOperator) x -> x;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.Quotable = lambda (%1 : int, %2 : int)int -> {
                    %3 : Var<int> = var %1 @"x";
                    %4 : Var<int> = var %2 @"y";
                    %5 : int = var.load %3;
                    %6 : int = var.load %4;
                    %7 : int = add %5 %6;
                    return %7;
                };
                return;
            };
            """)
    static final Quotable QUOTED_PLUS = (Quotable & IntBinaryOperator) (x, y) -> x + y;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.Quotable = lambda ()void -> {
                    %1 : java.lang.AssertionError = new @"func<java.lang.AssertionError>";
                    throw %1;
                };
                return;
            };
            """)
    static final Quotable QUOTED_THROW_NO_PARAM = (Quotable & Runnable) () -> {
        throw new AssertionError();
    };

    @IR("""
            func @"f" (%1 : Var<int>)void -> {
                %2 : java.lang.reflect.code.Quotable = lambda (%4 : int)int -> {
                    %5 : Var<int> = var %4 @"y";
                    %6 : int = var.load %1;
                    %7 : int = var.load %5;
                    %8 : int = add %6 %7;
                    return %8;
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
            func @"f" (%0 : QuotableIntersectionTest$Context)void -> {
                %1 : java.lang.reflect.code.Quotable = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"z";
                    %5 : int = field.load %0 @"QuotableIntersectionTest$Context::x()int";
                    %6 : int = field.load %0 @"QuotableIntersectionTest$Context::y()int";
                    %7 : int = add %5 %6;
                    %8 : int = var.load %4;
                    %9 : int = add %7 %8;
                    return %9;
                };
                return;
            };
            """)
    static final Quotable QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"x";
                %2 : java.util.function.IntUnaryOperator = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"y";
                    %5 : int = var.load %1;
                    %6 : int = var.load %4;
                    %7 : int = add %5 %6;
                    return %7;
                };
                %8 : Var<java.util.function.IntUnaryOperator> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        IntUnaryOperator op = (IntUnaryOperator & Quotable) y -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : QuotableIntersectionTest)void -> {
                %1 : java.util.function.IntUnaryOperator = lambda (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"z";
                    %4 : int = field.load %0 @"QuotableIntersectionTest::x()int";
                    %5 : int = field.load %0 @"QuotableIntersectionTest::y()int";
                    %6 : int = add %4 %5;
                    %7 : int = var.load %3;
                    %8 : int = add %6 %7;
                    return %8;
                };
                %9 : Var<java.util.function.IntUnaryOperator> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        IntUnaryOperator op = (IntUnaryOperator & Quotable) z -> x + y + z;
    }

    static void m() {
    }

    @IR("""
            func @"f" ()void -> {
                  %0 : java.lang.reflect.code.Quotable = lambda ()void -> {
                      invoke @"QuotableIntersectionTest::m()void";
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
            func @"f" ()void -> {
                  %0 : java.lang.reflect.code.Quotable = lambda (%1 : int)int -> {
                      %2 : Var<int> = var %1 @"x$0";
                      %3 : int = var.load %2;
                      %4 : int = invoke %3 @"QuotableIntersectionTest::g(int)int";
                      return %4;
                  };
                  return;
            };
            """)
    static final Quotable QUOTED_INT_PARAM_INT_RET_REF = (Quotable & IntUnaryOperator) QuotableIntersectionTest::g;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.Quotable = lambda (%1 : int)int[] -> {
                    %2 : Var<int> = var %1 @"x$0";
                    %3 : int = var.load %2;
                    %4 : int[] = new %3 @"func<int[], int>";
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
            func @"f" (%0 : QuotableIntersectionTest$ContextRef)void -> {
                %1 : java.lang.reflect.code.Quotable = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"x$0";
                    %5 : int = var.load %4;
                    %6 : int = invoke %0 %5 @"QuotableIntersectionTest$ContextRef::g(int)int";
                    return %6;
                };
                return;
            };
            """)
    static final Quotable QUOTED_CAPTURE_THIS_REF = new ContextRef().capture();
}
