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
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with boxing conversions.
 * @modules jdk.incubator.code
 * @build BoxingConversionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BoxingConversionTest
 */

public class BoxingConversionTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"long" = constant @1;
                %2 : java.type:"java.lang.Long" = invoke %1 @java.ref:"java.lang.Long::valueOf(long):java.lang.Long";
                %3 : Var<java.type:"java.lang.Long"> = var %2 @"x";
                return;
            };
            """)
    void test1() {
        Long x = 1L;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"java.lang.Long")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Long"> = var %1 @"L";
                %3 : java.type:"java.lang.Long" = var.load %2;
                %4 : java.type:"long" = invoke %3 @java.ref:"java.lang.Long::longValue():long";
                %5 : Var<java.type:"long"> = var %4 @"l";
                return;
            };
            """)
    void test2(Long L) {
        long l = L;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"long" = constant @0;
                %2 : java.type:"java.lang.Long" = invoke %1 @java.ref:"java.lang.Long::valueOf(long):java.lang.Long";
                %3 : Var<java.type:"java.lang.Object"> = var %2 @"o";
                return;
            };
            """)
    void test3() {
        Object o = 0L;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"java.lang.Object")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %1 @"o";
                %3 : java.type:"java.lang.Object" = var.load %2;
                %4 : java.type:"java.lang.Long" = cast %3 @java.type:"java.lang.Long";
                %5 : java.type:"long" = invoke %4 @java.ref:"java.lang.Long::longValue():long";
                %6 : Var<java.type:"long"> = var %5 @"l";
                return;
            };
            """)
    void test4(Object o) {
        long l = (long)o;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"java.lang.Integer")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Integer"> = var %1 @"i2";
                %3 : java.type:"java.lang.Integer" = var.load %2;
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"int" = invoke %3 @java.ref:"java.lang.Integer::intValue():int";
                %6 : java.type:"int" = add %5 %4;
                %7 : java.type:"java.lang.Integer" = invoke %6 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                var.store %2 %7;
                return;
            };
            """)
    void test5(Integer i2) {
        i2++;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"java.lang.Integer")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.Integer"> = var %1 @"i2";
                %3 : java.type:"java.lang.Integer" = var.load %2;
                %4 : java.type:"int" = constant @3;
                %5 : java.type:"int" = invoke %3 @java.ref:"java.lang.Integer::intValue():int";
                %6 : java.type:"int" = add %5 %4;
                %7 : java.type:"java.lang.Integer" = invoke %6 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                var.store %2 %7;
                return;
            };
            """)
    void test6(Integer i2) {
        i2 += 3;
    }

    static class Box {
        Integer i;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"BoxingConversionTest$Box" = new @java.ref:"BoxingConversionTest$Box::()";
                %2 : java.type:"java.lang.Integer" = field.load %1 @java.ref:"BoxingConversionTest$Box::i:java.lang.Integer";
                %3 : java.type:"int" = constant @1;
                %4 : java.type:"int" = invoke %2 @java.ref:"java.lang.Integer::intValue():int";
                %5 : java.type:"int" = add %4 %3;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                field.store %1 %6 @java.ref:"BoxingConversionTest$Box::i:java.lang.Integer";
                return;
            };
            """)
    void test7() {
        new Box().i++;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"BoxingConversionTest$Box" = new @java.ref:"BoxingConversionTest$Box::()";
                %2 : java.type:"java.lang.Integer" = field.load %1 @java.ref:"BoxingConversionTest$Box::i:java.lang.Integer";
                %3 : java.type:"int" = constant @3;
                %4 : java.type:"int" = invoke %2 @java.ref:"java.lang.Integer::intValue():int";
                %5 : java.type:"int" = add %4 %3;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                field.store %1 %6 @java.ref:"BoxingConversionTest$Box::i:java.lang.Integer";
                return;
            };
            """)
    void test8() {
        new Box().i += 3;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int[]", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int[]"> = var %1 @"ia";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"i";
                %5 : java.type:"int[]" = var.load %3;
                %6 : java.type:"int" = constant @0;
                %7 : java.type:"int" = array.load %5 %6;
                %8 : java.type:"java.lang.Integer" = var.load %4;
                %9 : java.type:"int" = invoke %8 @java.ref:"java.lang.Integer::intValue():int";
                %10 : java.type:"int" = add %7 %9;
                array.store %5 %6 %10;
                return;
            };
            """)
    void test9(int[] ia, Integer i) {
        ia[0] += i;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"boolean", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"boolean"> = var %1 @"cond";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %6 : java.type:"boolean" = var.load %3;
                        yield %6;
                    }
                    ()java.type:"int" -> {
                        %7 : java.type:"java.lang.Integer" = var.load %4;
                        %8 : java.type:"int" = invoke %7 @java.ref:"java.lang.Integer::intValue():int";
                        yield %8;
                    }
                    ()java.type:"int" -> {
                        %9 : java.type:"int" = constant @2;
                        yield %9;
                    };
                %10 : Var<java.type:"int"> = var %5 @"res";
                return;
            };
            """)
    void test10(boolean cond, Integer I) {
        int res = cond ? I : 2;
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"boolean", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"boolean"> = var %1 @"cond";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %6 : java.type:"boolean" = var.load %3;
                        yield %6;
                    }
                    ()java.type:"int" -> {
                        %7 : java.type:"int" = constant @2;
                        yield %7;
                    }
                    ()java.type:"int" -> {
                        %8 : java.type:"java.lang.Integer" = var.load %4;
                        %9 : java.type:"int" = invoke %8 @java.ref:"java.lang.Integer::intValue():int";
                        yield %9;
                    };
                %10 : Var<java.type:"int"> = var %5 @"res";
                return;
            };
            """)
    void test11(boolean cond, Integer I) {
        int res = cond ? 2 : I;
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"boolean")java.type:"void" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : java.type:"int" = java.cexpression
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = var.load %2;
                        yield %4;
                    }
                    ()java.type:"int" -> {
                        %5 : java.type:"int" = constant @1;
                        yield %5;
                    }
                    ()java.type:"int" -> {
                        %6 : java.type:"int" = constant @2;
                        yield %6;
                    };
                %7 : java.type:"java.lang.Integer" = invoke %3 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %8 : Var<java.type:"java.lang.Integer"> = var %7 @"x";
                return;
            };
            """)
    void test12(boolean cond) {
        Integer x = cond ? 1 : 2;
    }

    @CodeReflection
    @IR("""
            func @"test13" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.Integer>" = lambda ()java.type:"java.lang.Integer" -> {
                    %2 : java.type:"int" = constant @1;
                    %3 : java.type:"java.lang.Integer" = invoke %2 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                    return %3;
                };
                %4 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %1 @"s";
                return;
            };
            """)
    void test13() {
        Supplier<Integer> s = () -> { return 1; };
    }

    @CodeReflection
    @IR("""
            func @"test14" (%0 : java.type:"BoxingConversionTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.Integer>" = lambda ()java.type:"java.lang.Integer" -> {
                    %2 : java.type:"int" = constant @1;
                    %3 : java.type:"java.lang.Integer" = invoke %2 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                    return %3;
                };
                %4 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %1 @"s";
                return;
            };
            """)
    void test14() {
        Supplier<Integer> s = () -> 1;
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = java.switch.expression %5
                    (%7 : java.type:"int")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"java.lang.Integer" = var.load %4;
                        %11 : java.type:"int" = invoke %10 @java.ref:"java.lang.Integer::intValue():int";
                        yield %11;
                    }
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = constant @true;
                        yield %12;
                    }
                    ()java.type:"int" -> {
                        %13 : java.type:"int" = constant @0;
                        yield %13;
                    };
                %14 : Var<java.type:"int"> = var %6 @"x";
                return;
            };
            """)
    void test15(int i, Integer I) {
        int x = switch (i) {
            case 1 -> I;
            default -> 0;
        };
    }

    @CodeReflection
    @IR("""
            func @"test16" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = java.switch.expression %5
                    (%7 : java.type:"int")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"int" = constant @1;
                        yield %10;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"int" -> {
                        %12 : java.type:"java.lang.Integer" = var.load %4;
                        %13 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::intValue():int";
                        yield %13;
                    };
                %14 : Var<java.type:"int"> = var %6 @"x";
                return;
            };
            """)
    void test16(int i, Integer I) {
        int x = switch (i) {
            case 1 -> 1;
            default -> I;
        };
    }

    @CodeReflection
    @IR("""
            func @"test17" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Integer" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.Integer" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"java.lang.Integer" = invoke %8 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"java.lang.Integer" -> {
                        %11 : java.type:"int" = constant @0;
                        %12 : java.type:"java.lang.Integer" = invoke %11 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        yield %12;
                    };
                %13 : Var<java.type:"java.lang.Integer"> = var %4 @"x";
                return;
            };
            """)
    void test17(int i) {
        Integer x = switch (i) {
            case 1 -> 1;
            default -> 0;
        };
    }

    @CodeReflection
    @IR("""
            func @"test18" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = java.switch.expression %5
                    (%7 : java.type:"int")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"java.lang.Integer" = var.load %4;
                        %11 : java.type:"int" = invoke %10 @java.ref:"java.lang.Integer::intValue():int";
                        java.yield %11;
                    }
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = constant @true;
                        yield %12;
                    }
                    ()java.type:"int" -> {
                        %13 : java.type:"int" = constant @0;
                        java.yield %13;
                    };
                %14 : Var<java.type:"int"> = var %6 @"x";
                return;
            };
            """)
    void test18(int i, Integer I) {
        int x = switch (i) {
            case 1 -> { yield I; }
            default -> { yield 0; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test19" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = java.switch.expression %5
                    (%7 : java.type:"int")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"int" -> {
                        %10 : java.type:"int" = constant @1;
                        java.yield %10;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"int" -> {
                        %12 : java.type:"java.lang.Integer" = var.load %4;
                        %13 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::intValue():int";
                        java.yield %13;
                    };
                %14 : Var<java.type:"int"> = var %6 @"x";
                return;
            };
            """)
    void test19(int i, Integer I) {
        int x = switch (i) {
            case 1 -> { yield 1; }
            default -> { yield I; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test20" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Integer" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.Integer" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"java.lang.Integer" = invoke %8 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        java.yield %9;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"java.lang.Integer" -> {
                        %11 : java.type:"int" = constant @0;
                        %12 : java.type:"java.lang.Integer" = invoke %11 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        java.yield %12;
                    };
                %13 : Var<java.type:"java.lang.Integer"> = var %4 @"x";
                return;
            };
            """)
    void test20(int i) {
        Integer x = switch (i) {
            case 1 -> { yield 1; }
            default -> { yield 0; }
        };
    }

    @CodeReflection
    @IR("""
            func @"test21" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int", %2 : java.type:"java.lang.Integer")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"java.lang.Integer"> = var %2 @"I";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"java.lang.Integer" = var.load %4;
                %7 : java.type:"int" = invoke %6 @java.ref:"java.lang.Integer::intValue():int";
                %8 : java.type:"int" = add %5 %7;
                %9 : Var<java.type:"int"> = var %8 @"l";
                return;
            };
            """)
    void test21(int i, Integer I) {
        int l = i + I;
    }

    void m(Integer I) { }

    @CodeReflection
    @IR("""
            func @"test22" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Integer" = invoke %3 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                invoke %0 %4 @java.ref:"BoxingConversionTest::m(java.lang.Integer):void";
                return;
            };
            """)
    void test22(int i) {
        m(i);
    }

    void m(int i1, int i2, Integer... I) { }

    @CodeReflection
    @IR("""
            func @"test23" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                invoke %0 %3 %4 @java.ref:"BoxingConversionTest::m(int, int, java.lang.Integer[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
                return;
            };
            """)
    void test23(int i) {
        m(i, i);
    }

    @CodeReflection
    @IR("""
            func @"test24" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                invoke %0 %3 %4 %6 @java.ref:"BoxingConversionTest::m(int, int, java.lang.Integer[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
                return;
            };
            """)
    void test24(int i) {
        m(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test25" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"java.lang.Integer" = invoke %7 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                invoke %0 %3 %4 %6 %8 @java.ref:"BoxingConversionTest::m(int, int, java.lang.Integer[]):void" @invoke.kind="INSTANCE" @invoke.varargs=true;
                return;
            };
            """)
    void test25(int i) {
        m(i, i, i, i);
    }

    static class Box2 {
        Box2(Integer I) { }
        Box2(int i1, int i2, Integer... Is) { }
    }

    @CodeReflection
    @IR("""
            func @"test26" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Integer" = invoke %3 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %5 : java.type:"BoxingConversionTest$Box2" = new %4 @java.ref:"BoxingConversionTest$Box2::(java.lang.Integer)";
                return;
            };
            """)
    void test26(int i) {
        new Box2(i);
    }

    @CodeReflection
    @IR("""
            func @"test27" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"BoxingConversionTest$Box2" = new %3 %4 @java.ref:"BoxingConversionTest$Box2::(int, int, java.lang.Integer[])" @new.varargs=true;
                return;
            };
            """)
    void test27(int i) {
        new Box2(i, i);
    }

    @CodeReflection
    @IR("""
            func @"test28" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %7 : java.type:"BoxingConversionTest$Box2" = new %3 %4 %6 @java.ref:"BoxingConversionTest$Box2::(int, int, java.lang.Integer[])" @new.varargs=true;
                return;
            };
            """)
    void test28(int i) {
        new Box2(i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test29" (%0 : java.type:"BoxingConversionTest", %1 : java.type:"int")java.type:"void" -> {
                %2 : Var<java.type:"int"> = var %1 @"i";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"int" = var.load %2;
                %5 : java.type:"int" = var.load %2;
                %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %7 : java.type:"int" = var.load %2;
                %8 : java.type:"java.lang.Integer" = invoke %7 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %9 : java.type:"BoxingConversionTest$Box2" = new %3 %4 %6 %8 @java.ref:"BoxingConversionTest$Box2::(int, int, java.lang.Integer[])" @new.varargs=true;
                return;
            };
            """)
    void test29(int i) {
        new Box2(i, i, i, i);
    }

    @CodeReflection
    @IR("""
            func @"test30" (%0 : java.type:"java.lang.Integer")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                %2 : java.type:"java.lang.Integer" = var.load %1;
                %3 : java.type:"int" = invoke %2 @java.ref:"java.lang.Integer::intValue():int";
                %4 : java.type:"int" = neg %3;
                %5 : Var<java.type:"int"> = var %4 @"j";
                return;
            };
            """)
    static void test30(Integer i) {
        int j = -i;
    }

    @CodeReflection
    @IR("""
            func @"test31" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = neg %2;
                %4 : java.type:"java.lang.Integer" = invoke %3 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %5 : Var<java.type:"java.lang.Integer"> = var %4 @"j";
                return;
            };
            """)
    static void test31(int i) {
        Integer j = -i;
    }

    @CodeReflection
    @IR("""
            func @"test32" (%0 : java.type:"boolean")java.type:"void" -> {
                %1 : Var<java.type:"boolean"> = var %0 @"i";
                %2 : java.type:"boolean" = var.load %1;
                %3 : java.type:"boolean" = not %2;
                %4 : java.type:"java.lang.Boolean" = invoke %3 @java.ref:"java.lang.Boolean::valueOf(boolean):java.lang.Boolean";
                %5 : Var<java.type:"java.lang.Boolean"> = var %4 @"j";
                return;
            };
            """)
    static void test32(boolean i) {
        Boolean j = !i;
    }

    @CodeReflection
    @IR("""
            func @"test33" (%0 : java.type:"java.lang.Boolean")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Boolean"> = var %0 @"i";
                %2 : java.type:"java.lang.Boolean" = var.load %1;
                %3 : java.type:"boolean" = invoke %2 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                %4 : java.type:"boolean" = not %3;
                %5 : Var<java.type:"boolean"> = var %4 @"j";
                return;
            };
            """)
    static void test33(Boolean i) {
        boolean j = !i;
    }
}
