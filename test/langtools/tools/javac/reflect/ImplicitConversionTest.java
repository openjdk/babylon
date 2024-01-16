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
import java.util.function.LongSupplier;


/*
 * @test
 * @summary Smoke test for code reflection with implicit conversions.
 * @enablePreview
 * @build ImplicitConversionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ImplicitConversionTest
 */

public class ImplicitConversionTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0: ImplicitConversionTest)void -> {
                %1 : int = constant @"1";
                %2 : long = conv %1;
                %3 : Var<long> = var %2 @"x";
                return;
            };
            """)
    void test1() {
        long x = 1;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0: ImplicitConversionTest)void -> {
                %1 : long = constant @"0";
                %2 : Var<long> = var %1 @"x";
                %3 : int = constant @"1";
                %4 : long = conv %3;
                var.store %2 %4;
                return;
            };
            """)
    void test2() {
        long x;
        x = 1;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0: ImplicitConversionTest)void -> {
                %1 : long = constant @"0";
                %2 : Var<long> = var %1 @"x";
                %3 : long = var.load %2;
                %4 : int = constant @"1";
                %5 : long = conv %4;
                %6 : long = add %3 %5;
                var.store %2 %6;
                return;
            };
            """)
    void test3() {
        long x = 0L;
        x += 1;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0: ImplicitConversionTest, %1 : boolean)void -> {
                %2 : Var<boolean> = var %1 @"cond";
                %3 : long = constant @"0";
                %4 : Var<long> = var %3 @"x";
                %5 : long = java.cexpression
                    ^cond()boolean -> {
                        %6 : boolean = var.load %2;
                        yield %6;
                    }
                    ^truepart()long -> {
                        %7 : long = constant @"1";
                        yield %7;
                    }
                    ^falsepart()int -> {
                        %8 : int = constant @"2";
                        %9 : long = conv %8;
                        yield %9;
                    };
                var.store %4 %5;
                return;
            };
            """)
    void test4(boolean cond) {
        long x;
        x = cond ? 1L : 2;
    }

    @CodeReflection
    @IR("""
           func @"test5" (%0: ImplicitConversionTest, %1 : boolean)void -> {
               %2 : Var<boolean> = var %1 @"cond";
               %3 : long = constant @"0";
               %4 : Var<long> = var %3 @"x";
               %5 : long = java.cexpression
                   ^cond()boolean -> {
                       %6 : boolean = var.load %2;
                       yield %6;
                   }
                   ^truepart()int -> {
                       %7 : int = constant @"1";
                       %8 : long = conv %7;
                       yield %8;
                   }
                   ^falsepart()long -> {
                       %9 : long = constant @"2";
                       yield %9;
                   };
               var.store %4 %5;
               return;
           };
           """)
    void test5(boolean cond) {
        long x;
        x = cond ? 1 : 2L;
    }

    @CodeReflection
    @IR("""
           func @"test6" (%0: ImplicitConversionTest, %1 : boolean)void -> {
               %2 : Var<boolean> = var %1 @"cond";
               %3 : long = constant @"0";
               %4 : Var<long> = var %3 @"x";
               %5 : int = java.cexpression
                   ^cond()boolean -> {
                       %6 : boolean = var.load %2;
                       yield %6;
                   }
                   ^truepart()int -> {
                       %7 : int = constant @"1";
                       yield %7;
                   }
                   ^falsepart()int -> {
                       %8 : int = constant @"2";
                       yield %8;
                   };
               %9 : long = conv %5;
               var.store %4 %9;
               return;
           };
           """)
    void test6(boolean cond) {
        long x;
        x = cond ? 1 : 2;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0: ImplicitConversionTest)long -> {
                %1 : int = constant @"1";
                %2 : long = conv %1;
                return %2;
            };
            """)
    long test7() {
        return 1;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0: ImplicitConversionTest)void -> {
                %1 : java.util.function.LongSupplier = lambda ()long -> {
                    %2 : int = constant @"1";
                    %3 : long = conv %2;
                    return %3;
                };
                %4 : Var<java.util.function.LongSupplier> = var %1 @"s";
                return;
            };
            """)
    void test8() {
        LongSupplier s = () -> { return 1; };
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0: ImplicitConversionTest)void -> {
                %1 : java.util.function.LongSupplier = lambda ()long -> {
                    %2 : int = constant @"1";
                    %3 : long = conv %2;
                    return %3;
                };
                %4 : Var<java.util.function.LongSupplier> = var %1 @"s";
                return;
            };
            """)
    void test9() {
        LongSupplier s = () -> 1;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : long = constant @"1";
                        yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %9 : int = constant @"0";
                        %10 : long = conv %9;
                        yield %10;
                    };
                %11 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test10(int i) {
        long l = switch (i) {
            case 1 -> 1L;
            default -> 0;
        };
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : int = constant @"1";
                        %9 : long = conv %8;
                        yield %9;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %10 : long = constant @"0";
                        yield %10;
                    };
                %11 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test11(int i) {
        long l = switch (i) {
            case 1 -> 1;
            default -> 0L;
        };
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : int = constant @"1";
                        %9 : long = conv %8;
                        yield %9;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %10 : int = constant @"0";
                        %11 : long = conv %10;
                        yield %11;
                    };
                %12 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test12(int i) {
        long l = switch (i) {
            case 1 -> 1;
            default -> 0;
        };
    }

    @CodeReflection
    @IR("""
            func @"test13" (%0 : ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : long = constant @"1";
                        java.yield %8;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %9 : int = constant @"0";
                        %10 : long = conv %9;
                        java.yield %10;
                    };
                %11 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test13(int i) {
        long l = switch (i) {
            case 1 -> { yield 1L; }
            default -> { yield 0; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test14" (%0 : ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : int = constant @"1";
                        %9 : long = conv %8;
                        java.yield %9;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %10 : long = constant @"0";
                        java.yield %10;
                    };
                %11 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test14(int i) {
        long l = switch (i) {
            case 1 -> { yield 1; }
            default -> { yield 0L; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = java.switch.expression %3
                    ^constantCaseLabel(%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()long -> {
                        %8 : int = constant @"1";
                        %9 : long = conv %8;
                        java.yield %9;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()long -> {
                        %10 : int = constant @"0";
                        %11 : long = conv %10;
                        java.yield %11;
                    };
                %12 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test15(int i) {
        long l = switch (i) {
            case 1 -> { yield 1; }
            default -> { yield 0; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test16" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = conv %3;
                %5 : long = constant @"2";
                %6 : long = add %4 %5;
                %7 : Var<long> = var %6 @"l";
                return;
            };
            """)
    void test16(int i) {
        long l = i + 2L;
    }

    void m(long l) { }

    @CodeReflection
    @IR("""
            func @"test17" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = conv %3;
                invoke %0 %4 @"ImplicitConversionTest::m(long)void";
                return;
            };
            """)
    void test17(int i) {
        m(i);
    }

    void m(int i1, int i2, long... l) { }

    @CodeReflection
    @IR("""
            func @"test18" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                invoke %0 %3 %4 @"ImplicitConversionTest::m(int, int, long[])void";
                return;
            };
            """)
    void test18(int i) {
        m(i, i);
    }

    @CodeReflection
    @IR(""" 
           func @"test19" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                %5 : int = var.load %2;
                %6 : long = conv %5;
                invoke %0 %3 %4 %6 @"ImplicitConversionTest::m(int, int, long[])void";
                return;
           };
           """)
    void test19(int i) {
        m(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test20" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                %5 : int = var.load %2;
                %6 : long = conv %5;
                %7 : int = var.load %2;
                %8 : long = conv %7;
                invoke %0 %3 %4 %6 %8 @"ImplicitConversionTest::m(int, int, long[])void";
                return;
            };
            """)
    void test20(int i) {
        m(i, i, i, i);
    }

    static class Box {
        Box(long l) { }
        Box(int i1, int i2, long... longs) { }
    }

    @CodeReflection
    @IR("""
            func @"test21" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = conv %3;
                %5 : ImplicitConversionTest$Box = new %4 @"(long)ImplicitConversionTest$Box";
                return;
            };
            """)
    void test21(int i) {
        new Box(i);
    }

    @CodeReflection
    @IR("""
            func @"test22" (%0: ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                %5 : ImplicitConversionTest$Box = new %3 %4 @"(int, int, long[])ImplicitConversionTest$Box";
                return;
            };
            """)
    void test22(int i) {
        new Box(i, i);
    }

    @CodeReflection
    @IR("""
           func @"test23" (%0 : ImplicitConversionTest, %1 : int)void -> {
               %2 : Var<int> = var %1 @"i";
               %3 : int = var.load %2;
               %4 : int = var.load %2;
               %5 : int = var.load %2;
               %6 : long = conv %5;
               %7 : ImplicitConversionTest$Box = new %3 %4 %6 @"(int, int, long[])ImplicitConversionTest$Box";
               return;
           };           
           """)
    void test23(int i) {
        new Box(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test24" (%0 : ImplicitConversionTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                %5 : int = var.load %2;
                %6 : long = conv %5;
                %7 : int = var.load %2;
                %8 : long = conv %7;
                %9 : ImplicitConversionTest$Box = new %3 %4 %6 %8 @"(int, int, long[])ImplicitConversionTest$Box";
                return;
            };
            """)
    void test24(int i) {
        new Box(i, i, i, i);
    }
}
