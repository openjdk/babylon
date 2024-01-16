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
 * @summary Smoke test for code reflection with quoted lambdas.
 * @build QuotedTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotedTest
 */

import java.lang.reflect.code.Quoted;
import java.lang.runtime.CodeReflection;

public class QuotedTest {
    @IR("""
            func @"f" ()void -> {
                 %0 : java.lang.reflect.code.CoreOps$Closure<void> = closure ()void -> {
                    return;
                };
                return;
            };
            """)
    static final Quoted QUOTED_NO_PARAM_VOID = () -> {
    };

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.CoreOps$Closure<int> = closure ()int -> {
                    %2 : int = constant @"1";
                    return %2;
                };
                return;
            };
            """)
    static final Quoted QUOTED_NO_PARAM_CONST = () -> 1;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.CoreOps$Closure<int, int> = closure (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"x";
                    %4 : int = var.load %3;
                    return %4;
                };
                return;
            };
            """)
    static final Quoted QUOTED_ID = (int x) -> x;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.CoreOps$Closure<int, int, int> = closure (%2 : int, %3 : int)int -> {
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
    static final Quoted QUOTED_PLUS = (int x, int y) -> x + y;

    @IR("""
            func @"f" ()void -> {
                %0 : java.lang.reflect.code.CoreOps$Closure<java.lang.Object> = closure ()java.lang.Object -> {
                    %2 : java.lang.AssertionError = new @"()java.lang.AssertionError";
                    throw %2;
                };
                return;
            };
            """)
    static final Quoted QUOTED_THROW_NO_PARAM = () -> {
        throw new AssertionError();
    };

    // can we write out the root op then extract the closure ?

    @IR("""
            func @"f" (%1: Var<int>)void -> {
                %0 : java.lang.reflect.code.CoreOps$Closure<int, int> = closure (%4 : int)int -> {
                    %5 : Var<int> = var %4 @"y";
                    %6 : int = var.load %1;
                    %7 : int = var.load %5;
                    %8 : int = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final Quoted QUOTED_CAPTURE_PARAM = new Object() {
        Quoted captureContext(int x) {
            return (int y) -> x + y;
        }
    }.captureContext(42);

    static class Context {
        int x, y;

        Quoted capture() {
            return (int z) -> x + y + z;
        }
    }

    @IR("""
            func @"f" (%0 : QuotedTest$Context)void -> {
                %1 : java.lang.reflect.code.CoreOps$Closure<int, int> = closure (%3 : int)int -> {
                    %4 : Var<int> = var %3 @"z";
                    %5 : int = field.load %0 @"QuotedTest$Context::x()int";
                    %6 : int = field.load %0 @"QuotedTest$Context::y()int";
                    %7 : int = add %5 %6;
                    %8 : int = var.load %4;
                    %9 : int = add %7 %8;
                    return %9;
                };
                return;
            };
            """)
    static final Quoted QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"x";
                %2 : java.lang.reflect.code.Quoted = quoted ()void -> {
                    %3 : java.lang.reflect.code.CoreOps$Closure<int, int> = closure (%4 : int)int -> {
                        %5 : Var<int> = var %4 @"y";
                        %6 : int = var.load %1;
                        %7 : int = var.load %5;
                        %8 : int = add %6 %7;
                        return %8;
                    };
                    yield %3;
                };
                %9 : Var<java.lang.reflect.code.Quoted> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        Quoted op = (int y) -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : QuotedTest)void -> {
                %1 : java.lang.reflect.code.Quoted = quoted ()void -> {
                    %2 : java.lang.reflect.code.CoreOps$Closure<int, int> = closure (%3 : int)int -> {
                        %4 : Var<int> = var %3 @"z";
                        %5 : int = field.load %0 @"QuotedTest::x()int";
                        %6 : int = field.load %0 @"QuotedTest::y()int";
                        %7 : int = add %5 %6;
                        %8 : int = var.load %4;
                        %9 : int = add %7 %8;
                        return %9;
                    };
                    yield %2;
                };
                %10 : Var<java.lang.reflect.code.Quoted> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        Quoted op = (int z) -> x + y + z;
    }
}
