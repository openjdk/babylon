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

import jdk.incubator.code.Reflect;
import java.util.function.Supplier;


/*
 * @test
 * @summary Smoke test for code reflection with null constants.
 * @modules jdk.incubator.code
 * @enablePreview
 * @build NullTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester NullTest
 */

public class NullTest {

    @Reflect
    @IR("""
            func @"test1" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                return;
            };
            """)
    void test1() {
        String s = null;
    }

    @Reflect
    @IR("""
            func @"test2" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Object" = constant @null;
                %2 : java.type:"java.lang.String" = cast %1 @java.type:"java.lang.String";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"s";
                return;
            };
            """)
    void test2() {
        String s = (String)null;
    }

    @Reflect
    @IR("""
            func @"test3" (%0 : java.type:"NullTest", %1 : java.type:"boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = var.load %2;
                        yield %4;
                    }
                    ()java.type:"java.lang.String" -> {
                        %5 : java.type:"java.lang.String" = constant @null;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"java.lang.String" = constant @"";
                        yield %6;
                    };
                return %3;
            };
            """)
    String test3(boolean cond) {
        return cond ? null : "";
    }

    @Reflect
    @IR("""
            func @"test4" (%0 : java.type:"NullTest", %1 : java.type:"boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = var.load %2;
                        yield %4;
                    }
                    ()java.type:"java.lang.String" -> {
                        %5 : java.type:"java.lang.String" = constant @"";
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"java.lang.String" = constant @null;
                        yield %6;
                    };
                return %3;
            };
            """)
    String test4(boolean cond) {
        return cond ? "" : null;
    }

    @Reflect
    @IR("""
            func @"test5" (%0 : java.type:"NullTest", %1 : java.type:"boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : java.type:"java.lang.String" = java.cexpression
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = var.load %2;
                        yield %4;
                    }
                    ()java.type:"java.lang.String" -> {
                        %5 : java.type:"java.lang.String" = constant @null;
                        yield %5;
                    }
                    ()java.type:"java.lang.String" -> {
                        %6 : java.type:"java.lang.String" = constant @null;
                        yield %6;
                    };
                return %3;
            };
            """)
    String test5(boolean cond) {
        return cond ? null : null;
    }

    @Reflect
    @IR("""
            func @"test6" (%0 : java.type:"NullTest", %1 : java.type:"boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"boolean"> = var %1 @"cond";
                %3 : java.type:"java.lang.Object" = java.cexpression
                    ()java.type:"boolean" -> {
                        %4 : java.type:"boolean" = var.load %2;
                        yield %4;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %5 : java.type:"java.lang.Object" = constant @null;
                        yield %5;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %6 : java.type:"java.lang.Object" = constant @null;
                        yield %6;
                    };
                %7 : java.type:"java.lang.String" = cast %3 @java.type:"java.lang.String";
                return %7;
            };
            """)
    String test6(boolean cond) {
        return (String)(cond ? null : null);
    }

    @Reflect
    @IR("""
            func @"test7" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @"";
                        yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @null;
                        yield %10;
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

    @Reflect
    @IR("""
            func @"test8" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @null;
                        yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @"";
                        yield %10;
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

    @Reflect
    @IR("""
            func @"test9" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @null;
                        yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @null;
                        yield %10;
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

    @Reflect
    @IR("""
            func @"test10" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Object" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %8 : java.type:"java.lang.Object" = constant @null;
                        yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %10 : java.type:"java.lang.Object" = constant @null;
                        yield %10;
                    };
                %11 : java.type:"java.lang.String" = cast %4 @java.type:"java.lang.String";
                return %11;
            };
            """)
    String test10(int cond) {
        return (String)switch(cond) {
            case 1  -> null;
            default -> null;
        };
    }

    @Reflect
    @IR("""
            func @"test11" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @"";
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @null;
                        java.yield %10;
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

    @Reflect
    @IR("""
            func @"test12" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @null;
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @"";
                        java.yield %10;
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

    @Reflect
    @IR("""
            func @"test13" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @null;
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @null;
                        java.yield %10;
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

    @Reflect
    @IR("""
            func @"test14" (%0 : java.type:"NullTest", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"int"> = var %1 @"cond";
                %3 : java.type:"int" = var.load %2;
                %4 : java.type:"java.lang.Object" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %8 : java.type:"java.lang.Object" = constant @null;
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %10 : java.type:"java.lang.Object" = constant @null;
                        java.yield %10;
                    };
                %11 : java.type:"java.lang.String" = cast %4 @java.type:"java.lang.String";
                return %11;
            };
            """)
    String test14(int cond) {
        return (String)switch(cond) {
            case 1  -> { yield null; }
            default -> { yield null; }
        };
    }

    @Reflect
    @IR("""
            func @"test15" (%0 : java.type:"NullTest")java.type:"java.util.function.Supplier<java.lang.String>" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.String>" = lambda @lambda.isReflectable=true ()java.type:"java.lang.String" -> {
                    %2 : java.type:"java.lang.String" = constant @null;
                    return %2;
                };
                return %1;
            };
            """)
    Supplier<String> test15() {
        return () -> null;
    }

    static void m(String s, String... ss) { }

    @Reflect
    @IR("""
            func @"test16" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                invoke %1 @java.ref:"NullTest::m(java.lang.String, java.lang.String[]):void" @invoke.kind="STATIC" @invoke.varargs=true;
                return;
            };
            """)
    void test16() {
        m(null);
    }

    @Reflect
    @IR("""
            func @"test17" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"java.lang.String[]" = constant @null;
                invoke %1 %2 @java.ref:"NullTest::m(java.lang.String, java.lang.String[]):void";
                return;
            };
            """)
    void test17() {
        m(null, null);
    }

    @Reflect
    @IR("""
            func @"test18" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"java.lang.String" = constant @null;
                %3 : java.type:"java.lang.String" = constant @null;
                invoke %1 %2 %3 @java.ref:"NullTest::m(java.lang.String, java.lang.String[]):void" @invoke.kind="STATIC" @invoke.varargs=true;
                return;
            };
            """)
    void test18() {
        m(null, null, null);
    }

    static class Box {
        Box(String s, String... ss) { }
    }

    @Reflect
    @IR("""
            func @"test19" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"NullTest$Box" = new %1 @java.ref:"NullTest$Box::(java.lang.String, java.lang.String[])" @new.varargs=true;
                return;
            };
            """)
    void test19() {
        new Box(null);
    }

    @Reflect
    @IR("""
            func @"test20" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"java.lang.String[]" = constant @null;
                %3 : java.type:"NullTest$Box" = new %1 %2 @java.ref:"NullTest$Box::(java.lang.String, java.lang.String[])";
                return;
            };
            """)
    void test20() {
        new Box(null, null);
    }

    @Reflect
    @IR("""
            func @"test21" (%0 : java.type:"NullTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"java.lang.String" = constant @null;
                %3 : java.type:"java.lang.String" = constant @null;
                %4 : java.type:"NullTest$Box" = new %1 %2 %3 @java.ref:"NullTest$Box::(java.lang.String, java.lang.String[])" @new.varargs=true;
                return;
            };
            """)
    void test21() {
        new Box(null, null, null);
    }
}
