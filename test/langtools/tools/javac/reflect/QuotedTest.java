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
 * @modules jdk.incubator.code
 * @build QuotedTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester QuotedTest
 */

import jdk.incubator.code.Quoted;
import jdk.incubator.code.CodeReflection;

public class QuotedTest {
    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"void"> = closure ()java.type:"void" -> {
                    return;
                };
                return;
            };
            """)
    static final Quoted QUOTED_NO_PARAM_VOID = () -> {
    };

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"int"> = closure ()java.type:"int" -> {
                    %1 : java.type:"int" = constant @1;
                    return %1;
                };
                return;
            };
            """)
    static final Quoted QUOTED_NO_PARAM_CONST = () -> 1;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"int", java.type:"int"> = closure (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"x";
                    %3 : java.type:"int" = var.load %2;
                    return %3;
                };
                return;
            };
            """)
    static final Quoted QUOTED_ID = (int x) -> x;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"int", java.type:"int", java.type:"int"> = closure (%1 : java.type:"int", %2 : java.type:"int")java.type:"int" -> {
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
    static final Quoted QUOTED_PLUS = (int x, int y) -> x + y;

    @IR("""
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"java.lang.Object"> = closure ()java.type:"java.lang.Object" -> {
                    %1 : java.type:"java.lang.AssertionError" = new @java.ref:"java.lang.AssertionError::()";
                    throw %1;
                };
                return;
            };
            """)
    static final Quoted QUOTED_THROW_NO_PARAM = () -> {
        throw new AssertionError();
    };

    // can we write out the root op then extract the closure ?

    @IR("""
            func @"f" (%0 : Var<java.type:"int">)java.type:"void" -> {
                %1 : func<java.type:"int", java.type:"int"> = closure (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"y";
                    %4 : java.type:"int" = var.load %0;
                    %5 : java.type:"int" = var.load %3;
                    %6 : java.type:"int" = add %4 %5;
                    return %6;
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
            func @"f" (%0 : java.type:"QuotedTest$Context")java.type:"void" -> {
                %1 : func<java.type:"int", java.type:"int"> = closure (%2 : java.type:"int")java.type:"int" -> {
                    %3 : Var<java.type:"int"> = var %2 @"z";
                    %4 : java.type:"int" = field.load %0 @java.ref:"QuotedTest$Context::x:int";
                    %5 : java.type:"int" = field.load %0 @java.ref:"QuotedTest$Context::y:int";
                    %6 : java.type:"int" = add %4 %5;
                    %7 : java.type:"int" = var.load %3;
                    %8 : java.type:"int" = add %6 %7;
                    return %8;
                };
                return;
            };
            """)
    static final Quoted QUOTED_CAPTURE_FIELD = new Context().capture();

    @CodeReflection
    @IR("""
            func @"captureParam" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"x";
                %2 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
                    %3 : func<java.type:"int", java.type:"int"> = closure (%4 : java.type:"int")java.type:"int" -> {
                        %5 : Var<java.type:"int"> = var %4 @"y";
                        %6 : java.type:"int" = var.load %1;
                        %7 : java.type:"int" = var.load %5;
                        %8 : java.type:"int" = add %6 %7;
                        return %8;
                    };
                    yield %3;
                };
                %9 : Var<java.type:"jdk.incubator.code.Quoted"> = var %2 @"op";
                return;
            };
            """)
    static void captureParam(int x) {
        Quoted op = (int y) -> x + y;
    }

    int x, y;

    @CodeReflection
    @IR("""
            func @"captureField" (%0 : java.type:"QuotedTest")java.type:"void" -> {
                %1 : java.type:"jdk.incubator.code.Quoted" = quoted ()java.type:"void" -> {
                    %2 : func<java.type:"int", java.type:"int"> = closure (%3 : java.type:"int")java.type:"int" -> {
                        %4 : Var<java.type:"int"> = var %3 @"z";
                        %5 : java.type:"int" = field.load %0 @java.ref:"QuotedTest::x:int";
                        %6 : java.type:"int" = field.load %0 @java.ref:"QuotedTest::y:int";
                        %7 : java.type:"int" = add %5 %6;
                        %8 : java.type:"int" = var.load %4;
                        %9 : java.type:"int" = add %7 %8;
                        return %9;
                    };
                    yield %2;
                };
                %10 : Var<java.type:"jdk.incubator.code.Quoted"> = var %1 @"op";
                return;
            };
            """)
    void captureField() {
        Quoted op = (int z) -> x + y + z;
    }
}
