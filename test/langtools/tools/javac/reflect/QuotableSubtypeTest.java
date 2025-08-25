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
 * @build QuotableSubtypeTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotableSubtypeTest
 */

import jdk.incubator.code.Quotable;
import jdk.incubator.code.CodeReflection;
import java.util.function.IntBinaryOperator;
import java.util.function.IntFunction;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

public class QuotableSubtypeTest {

    interface QuotableRunnable extends Runnable, Quotable { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableRunnable" = lambda ()java.type:"void" -> {
                    return;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_NO_PARAM_VOID = () -> { };

    interface QuotableIntSupplier extends IntSupplier, Quotable { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableIntSupplier" = lambda ()java.type:"int" -> {
                    %1 : java.type:"int" = constant @1;
                    return %1;
                };
                return;
            };
            """)
    static final QuotableIntSupplier QUOTED_NO_PARAM_CONST = () -> 1;

    interface QuotableIntUnaryOperator extends IntUnaryOperator, Quotable { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x";
                    %3 : java.type:"int" = var.load %2;
                    return %3;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_ID = x -> x;

    interface QuotableIntBinaryOperator extends IntBinaryOperator, Quotable { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableIntBinaryOperator" = lambda (%1 : java.type:"int", %2 : java.type:"int")java.type:"int" -> {
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
    static final QuotableIntBinaryOperator QUOTED_PLUS = (x, y) -> x + y;
    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableRunnable" = lambda ()java.type:"void" -> {
                    %1 : java.type:"java.lang.AssertionError" = new @java.ref:"java.lang.AssertionError::()";
                    throw %1;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_THROW_NO_PARAM = () -> { throw new AssertionError(); };

    @IR("""
            func @"f" (%0 : Var<java.type:"int">)java.type:"void" -> {
                %1 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"y";
                    %4 : java.type:"int" = var.load %0;
                    %5 : java.type:"int" = var.load %3;
                    %6 : java.type:"int" = add %4 %5;
                    return %6;
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
            func @"f" (%0 : java.type:"QuotableSubtypeTest$Context")java.type:"void" -> {
                %1 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"z";
                    %4 : java.type:"int" = field.load %0 @java.ref:"QuotableSubtypeTest$Context::x:int";
                    %5 : java.type:"int" = field.load %0 @java.ref:"QuotableSubtypeTest$Context::y:int";
                    %6 : java.type:"int" = add %4 %5;
                    %7 : java.type:"int" = var.load %3;
                    %8 : java.type:"int" = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"x";
                %2 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%3 : java.type:"int")java.type:"int" -> {
                    %4 : Var<java.type:"int"> = var %3 @"y";
                    %5 : java.type:"int" = var.load %1;
                    %6 : java.type:"int" = var.load %4;
                    %7 : java.type:"int" = add %5 %6;
                    return %7;
                };
                %8 : Var<java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator"> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        QuotableIntUnaryOperator op = y -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : java.type:"QuotableSubtypeTest")java.type:"void" -> {
                %1 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"z";
                    %4 : java.type:"int" = field.load %0 @java.ref:"QuotableSubtypeTest::x:int";
                    %5 : java.type:"int" = field.load %0 @java.ref:"QuotableSubtypeTest::y:int";
                    %6 : java.type:"int" = add %4 %5;
                    %7 : java.type:"int" = var.load %3;
                    %8 : java.type:"int" = add %6 %7;
                    return %8;
                };
                %9 : Var<java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator"> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        QuotableIntUnaryOperator op = z -> x + y + z;
    }

    static void m() { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableRunnable" = lambda ()java.type:"void" -> {
                    invoke @java.ref:"QuotableSubtypeTest::m():void";
                    return;
                };
                return;
            };
            """)
    static final QuotableRunnable QUOTED_NO_PARAM_VOID_REF = QuotableSubtypeTest::m;

    static int g(int i) { return i; }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x$0";
                    %3 : java.type:"int" = var.load %2;
                    %4 : java.type:"int" = invoke %3 @java.ref:"QuotableSubtypeTest::g(int):int";
                    return %4;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_INT_PARAM_INT_RET_REF = QuotableSubtypeTest::g;

    interface QuotableIntFunction<A> extends Quotable, IntFunction<A> { }

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"QuotableSubtypeTest$QuotableIntFunction<int[]>" = lambda (%1 : java.type:"int")java.type:"int[]" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x$0";
                    %3 : java.type:"int" = var.load %2;
                    %4 : java.type:"int[]" = new %3 @java.ref:"int[]::(int)";
                    return %4;
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
            func @"f" (%0 : java.type:"QuotableSubtypeTest$ContextRef")java.type:"void" -> {
                %1 : java.type:"QuotableSubtypeTest$QuotableIntUnaryOperator" = lambda (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"x$0";
                    %4 : java.type:"int" = var.load %3;
                    %5 : java.type:"int" = invoke %0 %4 @java.ref:"QuotableSubtypeTest$ContextRef::g(int):int";
                    return %5;
                };
                return;
            };
            """)
    static final QuotableIntUnaryOperator QUOTED_CAPTURE_THIS_REF = new ContextRef().capture();

    static final int Z = 42;
    @IR("""
            func @"f" (%0 : Var<java.type:"int">)java.type:"void" -> {
                %1 : java.type:"QuotableSubtypeTest$QuotableRunnable" = lambda ()java.type:"void" -> {
                    %2 : java.type:"int" = var.load %0;
                    %3 : Var<java.type:"int"> = var %2 @"x";
                    return;
                };
                return;
            };
            """)
    static QuotableRunnable QUOTED_CAPTURE_FINAL_STATIC_FIELD = () -> {
        int x = Z;
    };
}
