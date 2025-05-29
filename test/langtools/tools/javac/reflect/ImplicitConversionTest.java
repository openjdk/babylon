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
import java.util.function.LongSupplier;


/*
 * @test
 * @summary Smoke test for code reflection with implicit conversions.
 * @modules jdk.incubator.code
 * @enablePreview
 * @build ImplicitConversionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ImplicitConversionTest
 */

public class ImplicitConversionTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"ImplicitConversionTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @1;
                %2 : java.type:"long" = conv %1;
                %3 : Var<java.type:"long"> = var %2 @"x";
                return;
            };
            """)
    void test1() {
        long x = 1;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"ImplicitConversionTest")java.type:"void" -> {
                %1 : Var<java.type:"long"> = var @"x";
                %2 : java.type:"int" = constant @1;
                %3 : java.type:"long" = conv %2;
                var.store %1 %3;
                return;
            };
            """)
    void test2() {
        long x;
        x = 1;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"ImplicitConversionTest")java.type:"void" -> {
                %1 : java.type:"long" = constant @0;
                %2 : Var<java.type:"long"> = var %1 @"x";
                %3 : java.type:"long" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"long" = conv %4;
                %6 : java.type:"long" = add %3 %5;
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
            func @"test4" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : Var<java.type:"long"> = var @"x";
                %4 : java.type:"long" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"long" -> {
                        %6 : java.type:"long" = constant @1;
                        yield %6;
                    }
                    ()java.type:"long" -> {
                        %7 : java.type:"int" = constant @2;
                        %8 : java.type:"long" = conv %7;
                        yield %8;
                    };
                var.store %3 %4;
                return;
            };
            """)
    void test4(boolean cond) {
        long x;
        x = cond ? 1L : 2;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : Var<java.type:"long"> = var @"x";
                %4 : java.type:"long" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"long" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"long" = conv %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"long" = constant @2;
                        yield %8;
                    };
                var.store %3 %4;
                return;
            };
           """)
    void test5(boolean cond) {
        long x;
        x = cond ? 1 : 2L;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : Var<java.type:"long"> = var @"x";
                %4 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %5 : java.type:"boolean" = var.load %2;
                        yield %5;
                    }
                    ()java.type:"int" -> {
                        %6 : java.type:"int" = constant @1;
                        yield %6;
                    }
                    ()java.type:"int" -> {
                        %7 : java.type:"int" = constant @2;
                        yield %7;
                    };
                %8 : java.type:"long" = conv %4;
                var.store %3 %8;
                return;
            };
           """)
    void test6(boolean cond) {
        long x;
        x = cond ? 1 : 2;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"ImplicitConversionTest")java.type:"long" -> {
                %1 : java.type:"int" = constant @1;
                %2 : java.type:"long" = conv %1;
                return %2;
            };
            """)
    long test7() {
        return 1;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"ImplicitConversionTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.LongSupplier" = lambda ()java.type:"long" -> {
                    %2 : java.type:"int" = constant @1;
                    %3 : java.type:"long" = conv %2;
                    return %3;
                };
                %4 : Var<java.type:"java.util.function.LongSupplier"> = var %1 @"s";
                return;
            };
            """)
    void test8() {
        LongSupplier s = () -> { return 1; };
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"ImplicitConversionTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.LongSupplier" = lambda ()java.type:"long" -> {
                    %2 : java.type:"int" = constant @1;
                    %3 : java.type:"long" = conv %2;
                    return %3;
                };
                %4 : Var<java.type:"java.util.function.LongSupplier"> = var %1 @"s";
                return;
            };
            """)
    void test9() {
        LongSupplier s = () -> 1;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"long" = constant @1;
                        yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"long" -> {
                        %10 : java.type:"int" = constant @0;
                        %11 : java.type:"long" = conv %10;
                        yield %11;
                    };
                %12 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test11" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"long" = conv %8;
                        yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"long" -> {
                        %11 : java.type:"long" = constant @0;
                        yield %11;
                    };
                %12 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test12" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"long" = conv %8;
                        yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"long" -> {
                        %11 : java.type:"int" = constant @0;
                        %12 : java.type:"long" = conv %11;
                        yield %12;
                    };
                %13 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test13" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"long" = constant @1;
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"long" -> {
                        %10 : java.type:"int" = constant @0;
                        %11 : java.type:"long" = conv %10;
                        java.yield %11;
                    };
                %12 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test14" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"long" = conv %8;
                        java.yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"long" -> {
                        %11 : java.type:"long" = constant @0;
                        java.yield %11;
                    };
                %12 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test15" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"long" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"long" = conv %8;
                        java.yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"long" -> {
                        %11 : java.type:"int" = constant @0;
                        %12 : java.type:"long" = conv %11;
                        java.yield %12;
                    };
                %13 : Var<java.type:"long"> = var %4 @"l";
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
            func @"test16" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = conv %3;
                %5 : java.type:"long" = constant @2;
                %6 : java.type:"long" = add %4 %5;
                %7 : Var<java.type:"long"> = var %6 @"l";
                return;
            };
            """)
    void test16(int i) {
        long l = i + 2L;
    }

    void m(long l) { }

    @CodeReflection
    @IR("""
            func @"test17" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = conv %3;
                invoke %0 %4 @java.ref:"ImplicitConversionTest::m(long):void";
                return;
            };
            """)
    void test17(int i) {
        m(i);
    }

    void m(int i1, int i2, long... l) { }

    @CodeReflection
    @IR("""
            func @"test18" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                invoke %0 %3 %4 @java.ref:"ImplicitConversionTest::m(int, int, long[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
                return;
            };
            """)
    void test18(int i) {
        m(i, i);
    }

    @CodeReflection
    @IR("""
            func @"test19" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"long" = conv %5;
                invoke %0 %3 %4 %6 @java.ref:"ImplicitConversionTest::m(int, int, long[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
                return;
            };
           """)
    void test19(int i) {
        m(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test20" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"long" = conv %5;
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"long" = conv %7;
                invoke %0 %3 %4 %6 %8 @java.ref:"ImplicitConversionTest::m(int, int, long[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
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
            func @"test21" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"long" = conv %3;
                %5 : java.type:"ImplicitConversionTest$Box" = new %4 @java.ref:"ImplicitConversionTest$Box::(long)";
                return;
            };
            """)
    void test21(int i) {
        new Box(i);
    }

    @CodeReflection
    @IR("""
            func @"test22" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"ImplicitConversionTest$Box" = new %3 %4 @java.ref:"ImplicitConversionTest$Box::(int, int, long[])" @new.varargs=true;
                return;
            };
            """)
    void test22(int i) {
        new Box(i, i);
    }

    @CodeReflection
    @IR("""
            func @"test23" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"long" = conv %5;
                %7 : java.type:"ImplicitConversionTest$Box" = new %3 %4 %6 @java.ref:"ImplicitConversionTest$Box::(int, int, long[])" @new.varargs=true;
                return;
            };
           """)
    void test23(int i) {
        new Box(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test24" (%0 : java.type:"ImplicitConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"long" = conv %5;
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"long" = conv %7;
                %9 : java.type:"ImplicitConversionTest$Box" = new %3 %4 %6 %8 @java.ref:"ImplicitConversionTest$Box::(int, int, long[])" @new.varargs=true;
                return;
            };
            """)
    void test24(int i) {
        new Box(i, i, i, i);
    }
}
