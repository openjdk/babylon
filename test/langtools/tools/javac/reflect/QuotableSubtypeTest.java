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
 * @build QuotableSubtypeTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotableSubtypeTest
 */

import java.lang.reflect.code.Quotable;
import java.lang.runtime.CodeReflection;
import java.util.function.IntBinaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

public class QuotableSubtypeTest {

    interface QuotableRunnable extends Runnable, Quotable { }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableRunnable = lambda ()void -> {
                    return;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_NO_PARAM_VOID = () -> { };

    interface QuotableIntSupplier extends IntSupplier, Quotable { }

    @IR("""
           func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableIntSupplier = lambda ()int -> {
                    %2 : int = constant @"1";
                    return %2;
                };
                return;
           };
            """)
    static final QuotableIntSupplier QUOTED_NO_PARAM_CONST = () -> 1;

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"x";
                    %4 : int = var.load %3;
                    return %4;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_ID = x -> x;

    interface QuotableIntBinaryOperator extends IntBinaryOperator, Quotable { }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableIntBinaryOperator = lambda (%2 : int, %3 : int)int -> {
                    %4 : Var<int> = var %2 @"x";
                    %5 : Var<int> = var %3 @"y";
                    %6 : int = var.load %4;
                    %7 : int = var.load %5;
                    %8 : int = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final QuotableIntBinaryOperator QUOTED_PLUS = (x, y) -> x + y;
    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableRunnable = lambda ()void -> {
                    %2 : java.lang.AssertionError = new @"func<java.lang.AssertionError>";
                    throw %2;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_THROW_NO_PARAM = () -> { throw new AssertionError(); };

    @IR("""
            func @"f" (%1 : Var<int>)void -> {
                %0 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%4 : int)int -> {
                    %5 : Var<int> = var %4 @"y";
                    %6 : int = var.load %1;
                    %7 : int = var.load %5;
                    %8 : int = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_CAPTURE_PARAM = new Object() {
        QuotableIntUnaryOperator captureContext(int x) {
            return y -> x + y;
        }
    }.captureContext(42);

    static class Context {
        int x, y;

        QuotableIntUnaryOperator capture() {
            return z -> x + y + z;
        }
    }

    @IR("""
            func @"f" (%0 : QuotableSubtypeTest$Context)void -> {
                %1 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"z";
                    %5 : int = field.load %0 @"QuotableSubtypeTest$Context::x()int";
                    %6 : int = field.load %0 @"QuotableSubtypeTest$Context::y()int";
                    %7 : int = add %5 %6;
                    %8 : int = var.load %4;
                    %9 : int = add %7 %8;
                    return %9;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"x";
                %2 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"y";
                    %5 : int = var.load %1;
                    %6 : int = var.load %4;
                    %7 : int = add %5 %6;
                    return %7;
                };
                %8 : Var<QuotableSubtypeTest$QuotableIntUnaryOperator> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        QuotableIntUnaryOperator op = y -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : QuotableSubtypeTest)void -> {
                %1 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"z";
                    %4 : int = field.load %0 @"QuotableSubtypeTest::x()int";
                    %5 : int = field.load %0 @"QuotableSubtypeTest::y()int";
                    %6 : int = add %4 %5;
                    %7 : int = var.load %3;
                    %8 : int = add %6 %7;
                    return %8;
                };
                %9 : Var<QuotableSubtypeTest$QuotableIntUnaryOperator> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        QuotableIntUnaryOperator op = z -> x + y + z;
    }

    static void m() { }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableRunnable = lambda ()void -> {
                    invoke @"QuotableSubtypeTest::m()void";
                    return;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_NO_PARAM_VOID_REF = QuotableSubtypeTest::m;

    static int g(int i) { return i; }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"x$0";
                    %4 : int = var.load %3;
                    %5 : int = invoke %4 @"QuotableSubtypeTest::g(int)int";
                    return %5;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_INT_PARAM_INT_RET_REF = QuotableSubtypeTest::g;

    interface QuotableIntFunction<A> extends Quotable, IntFunction<A> { }

    @IR("""
            func @"f" ()void -> {
                %0 : QuotableSubtypeTest$QuotableIntFunction<int[]> = lambda (%2 : int)int[] -> {
                    %3 : Var<int> = var %2 @"x$0";
                    %4 : int = var.load %3;
                    %5 : int[] = new %4 @"func<int[], int>";
                    return %5;
                };
                return;
            };
            """)
    static final QuotableIntFunction<int[]> QUOTED_INT_PARAM_ARR_RET_REF = int[]::new;

    static class ContextRef {
        int g(int i) { return i; }

        QuotableIntUnaryOperator capture() {
            return this::g;
        }
    }

    @IR("""
            func @"f" (%0 : QuotableSubtypeTest$ContextRef)void -> {
                %1 : QuotableSubtypeTest$QuotableIntUnaryOperator = lambda (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"x$0";
                    %5 : int = var.load %4;
                    %6 : int = invoke %0 %5 @"QuotableSubtypeTest$ContextRef::g(int)int";
                    return %6;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_CAPTURE_THIS_REF = new ContextRef().capture();
}
