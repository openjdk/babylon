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
import java.util.function.Supplier;


/*
 * @test
 * @summary Smoke test for code reflection with null constants.
 * @enablePreview
 * @build NullTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester NullTest
 */

public class NullTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : Var<java.lang.String> = var %1 @"s";
                return;
            };
            """)
    void test1() {
        String s = null;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : NullTest)void -> {
                %1 : java.lang.Object = constant @null;
                %2 : java.lang.String = cast %1 @"java.lang.String";
                %3 : Var<java.lang.String> = var %2 @"s";
                return;
            };
            """)
    void test2() {
        String s = (String)null;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : NullTest, %1 : boolean)java.lang.String -> {
                %2 : Var<boolean> = var %1 @"cond";
                %3 : java.lang.String = java.cexpression
                    ^cond()boolean -> {
                        %4 : boolean = var.load %2;
                        yield %4;
                    }
                    ^truepart()java.lang.String -> {
                        %5 : java.lang.String = constant @null;
                        yield %5;
                    }
                    ^falsepart()java.lang.String -> {
                        %6 : java.lang.String = constant @"";
                        yield %6;
                    };
                return %3;
            };
            """)
    String test3(boolean cond) {
        return cond ? null : "";
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : NullTest, %1 : boolean)java.lang.String -> {
                %2 : Var<boolean> = var %1 @"cond";
                %3 : java.lang.String = java.cexpression
                    ^cond()boolean -> {
                        %4 : boolean = var.load %2;
                        yield %4;
                    }
                    ^truepart()java.lang.String -> {
                        %5 : java.lang.String = constant @"";
                        yield %5;
                    }
                    ^falsepart()java.lang.String -> {
                        %6 : java.lang.String = constant @null;
                        yield %6;
                    };
                return %3;
            };
            """)
    String test4(boolean cond) {
        return cond ? "" : null;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : NullTest, %1 : boolean)java.lang.String -> {
                %2 : Var<boolean> = var %1 @"cond";
                %3 : java.lang.String = java.cexpression
                    ^cond()boolean -> {
                        %4 : boolean = var.load %2;
                        yield %4;
                    }
                    ^truepart()java.lang.String -> {
                        %5 : java.lang.String = constant @null;
                        yield %5;
                    }
                    ^falsepart()java.lang.String -> {
                        %6 : java.lang.String = constant @null;
                        yield %6;
                    };
                return %3;
            };
            """)
    String test5(boolean cond) {
        return cond ? null : null;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : NullTest, %1 : boolean)java.lang.String -> {
                %2 : Var<boolean> = var %1 @"cond";
                %3 : java.lang.Object = java.cexpression
                    ^cond()boolean -> {
                        %4 : boolean = var.load %2;
                        yield %4;
                    }
                    ^truepart()java.lang.Object -> {
                        %5 : java.lang.Object = constant @null;
                        yield %5;
                    }
                    ^falsepart()java.lang.Object -> {
                        %6 : java.lang.Object = constant @null;
                        yield %6;
                    };
                %7 : java.lang.String = cast %3 @"java.lang.String";
                return %7;
            };
            """)
    String test6(boolean cond) {
        return (String)(cond ? null : null);
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @"";
                        yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @null;
                        yield %9;
                    };
                return %4;
            };
            """)
    String test7(int cond) {
        return switch(cond) {
            case 1  -> "";
            default -> null;
        };
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @null;
                        yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @"";
                        yield %9;
                    };
                return %4;
            };
            """)
    String test8(int cond) {
        return switch(cond) {
            case 1  -> null;
            default -> "";
        };
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @null;
                        yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @null;
                        yield %9;
                    };
                return %4;
            };
            """)
    String test9(int cond) {
        return switch(cond) {
            case 1  -> null;
            default -> null;
        };
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.Object = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.Object -> {
                        %8 : java.lang.Object = constant @null;
                        yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %9 : java.lang.Object = constant @null;
                        yield %9;
                    };
                %10 : java.lang.String = cast %4 @"java.lang.String";
                return %10;
            };
            """)
    String test10(int cond) {
        return (String)switch(cond) {
            case 1  -> null;
            default -> null;
        };
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @"";
                        java.yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @null;
                        java.yield %9;
                    };
                return %4;
            };
            """)
    String test11(int cond) {
        return switch(cond) {
            case 1  -> { yield ""; }
            default -> { yield null; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @null;
                        java.yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @"";
                        java.yield %9;
                    };
                return %4;
            };
            """)
    String test12(int cond) {
        return switch(cond) {
            case 1  -> { yield null; }
            default -> { yield ""; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test13" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.String = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @null;
                        java.yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @null;
                        java.yield %9;
                    };
                return %4;
            };
            """)
    String test13(int cond) {
        return switch(cond) {
            case 1  -> { yield null; }
            default -> { yield null; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test14" (%0 : NullTest, %1 : int)java.lang.String -> {
                %2 : Var<int> = var %1 @"cond";
                %3 : int = var.load %2;
                %4 : java.lang.Object = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()java.lang.Object -> {
                        %8 : java.lang.Object = constant @null;
                        java.yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %9 : java.lang.Object = constant @null;
                        java.yield %9;
                    };
                %10 : java.lang.String = cast %4 @"java.lang.String";
                return %10;
            };
            """)
    String test14(int cond) {
        return (String)switch(cond) {
            case 1  -> { yield null; }
            default -> { yield null; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : NullTest)java.util.function.Supplier<java.lang.String> -> {
                %1 : java.util.function.Supplier<java.lang.String> = lambda ()java.lang.String -> {
                    %2 : java.lang.String = constant @null;
                    return %2;
                };
                return %1;
            };
            """)
    Supplier<String> test15() {
        return () -> null;
    }

    static void m(String s, String... ss) { }

    @CodeReflection
    @IR("""
            func @"test16" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                invoke %1 @"NullTest::m(java.lang.String, java.lang.String[])void";
                return;
            };
            """)
    void test16() {
        m(null);
    }

    @CodeReflection
    @IR("""
            func @"test17" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : java.lang.String[] = constant @null;
                invoke %1 %2 @"NullTest::m(java.lang.String, java.lang.String[])void";
                return;
            };
            """)
    void test17() {
        m(null, null);
    }

    @CodeReflection
    @IR("""
            func @"test18" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : java.lang.String = constant @null;
                %3 : java.lang.String = constant @null;
                invoke %1 %2 %3 @"NullTest::m(java.lang.String, java.lang.String[])void";
                return;
            };
            """)
    void test18() {
        m(null, null, null);
    }

    static class Box {
        Box(String s, String... ss) { }
    }

    @CodeReflection
    @IR("""
            func @"test19" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : NullTest$Box = new %1 @"(java.lang.String, java.lang.String[])NullTest$Box";
                return;
            };
            """)
    void test19() {
        new Box(null);
    }

    @CodeReflection
    @IR("""
            func @"test20" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : java.lang.String[] = constant @null;
                %3 : NullTest$Box = new %1 %2 @"(java.lang.String, java.lang.String[])NullTest$Box";
                return;
            };
            """)
    void test20() {
        new Box(null, null);
    }

    @CodeReflection
    @IR("""
            func @"test21" (%0 : NullTest)void -> {
                %1 : java.lang.String = constant @null;
                %2 : java.lang.String = constant @null;
                %3 : java.lang.String = constant @null;
                %4 : NullTest$Box = new %1 %2 %3 @"(java.lang.String, java.lang.String[])NullTest$Box";
                return;
            };
            """)
    void test21() {
        new Box(null, null, null);
    }
}
