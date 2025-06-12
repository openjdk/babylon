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
import java.util.function.Consumer;
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with switch expressions.
 * @modules jdk.incubator.code
 * @enablePreview
 * @build SwitchExpressionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchExpressionTest
 */

public class SwitchExpressionTest {

    @CodeReflection
    @IR("""
            func @"constantCaseLabelRule" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %7 : java.type:"java.lang.String" = constant @"FOO";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %11 : java.type:"java.lang.String" = constant @"FOO";
                        yield %11;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"java.lang.String" = constant @"BAZ";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %15 : java.type:"java.lang.String" = constant @"FOO";
                        yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %17 : java.type:"java.lang.String" = constant @"";
                        yield %17;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelRule(String r) {
        return switch (r) {
            case "FOO" -> "FOO";
            case "BAR" -> "FOO";
            case "BAZ" -> "FOO";
            default -> "";
        };
    }

    @CodeReflection
    @IR("""
            func @"constantCaseLabelsRule" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %6 : java.type:"java.lang.String" = constant @"FOO";
                                %7 : java.type:"boolean" = invoke %4 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"java.lang.String" = constant @"BAR";
                                %9 : java.type:"boolean" = invoke %4 %8 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %10 : java.type:"java.lang.String" = constant @"BAZ";
                                %11 : java.type:"boolean" = invoke %4 %10 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %11;
                            };
                        yield %5;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %12 : java.type:"java.lang.String" = constant @"FOO";
                        yield %12;
                    }
                    ()java.type:"boolean" -> {
                        %13 : java.type:"boolean" = constant @true;
                        yield %13;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %14 : java.type:"java.lang.String" = constant @"";
                        java.yield %14;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelsRule(String r) {
        return switch (r) {
            case "FOO", "BAR", "BAZ" -> "FOO";
            default -> {
                yield "";
            }
        };
    }

    @CodeReflection
    @IR("""
            func @"constantCaseLabelStatement" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %7 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %11 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %11;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"java.lang.String" = constant @"BAZ";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %15 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %17 : java.type:"java.lang.String" = constant @"";
                        java.yield %17;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelStatement(String r) {
        return switch (r) {
            case "FOO" : yield "FOO";
            case "BAR" : yield "FOO";
            case "BAZ" : yield "FOO";
            default : yield "";
        };
    }

    @CodeReflection
    @IR("""
            func @"constantCaseLabelsStatement" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %6 : java.type:"java.lang.String" = constant @"FOO";
                                %7 : java.type:"boolean" = invoke %4 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"java.lang.String" = constant @"BAR";
                                %9 : java.type:"boolean" = invoke %4 %8 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %10 : java.type:"java.lang.String" = constant @"BAZ";
                                %11 : java.type:"boolean" = invoke %4 %10 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %11;
                            };
                        yield %5;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %12 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %12;
                    }
                    ()java.type:"boolean" -> {
                        %13 : java.type:"boolean" = constant @true;
                        yield %13;
                    }
                    ()java.type:"java.lang.Object" -> {
                        java.block ()java.type:"void" -> {
                            %14 : java.type:"java.lang.String" = constant @"";
                            java.yield %14;
                        };
                        unreachable;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelsStatement(String r) {
        return switch (r) {
            case "FOO", "BAR", "BAZ" : yield "FOO";
            default : { yield ""; }
        };
    }

    @CodeReflection
    @IR("""
            func @"constantCaseLabelStatements" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.Object" -> {
                        java.block ()java.type:"void" -> {
                            %7 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                            %8 : java.type:"java.lang.String" = constant @"FOO";
                            invoke %7 %8 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                            yield;
                        };
                        java.block ()java.type:"void" -> {
                            %9 : java.type:"java.lang.String" = constant @"FOO";
                            java.yield %9;
                        };
                        unreachable;
                    }
                    ()java.type:"boolean" -> {
                        %10 : java.type:"boolean" = constant @true;
                        yield %10;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %11 : java.type:"java.lang.String" = constant @"";
                        java.yield %11;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelStatements(String r) {
        return switch (r) {
            case "FOO" : {
                System.out.println("FOO");
            }
            {
                yield "FOO";
            }
            default : yield "";
        };
    }

    @CodeReflection
    @IR("""
            func @"constantCaseLabelFallthrough" (%0 : java.type:"java.lang.String")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.Object" -> {
                        java.block ()java.type:"void" -> {
                            %7 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                            %8 : java.type:"java.lang.String" = constant @"FOO";
                            invoke %7 %8 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                            yield;
                        };
                        java.switch.fallthrough;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %10 : java.type:"java.lang.String" = constant @"";
                        java.yield %10;
                    };
                return %3;
            };
            """)
    public static Object constantCaseLabelFallthrough(String r) {
        return switch (r) {
            case "FOO" : {
                System.out.println("FOO");
            }
            default : yield "";
        };
    }

    record A(Number n) {
    }

    @CodeReflection
    @IR("""
            func @"patternCaseLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"r";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.Number" = constant @null;
                %4 : Var<java.type:"java.lang.Number"> = var %3 @"n";
                %5 : java.type:"java.lang.String" = constant @null;
                %6 : Var<java.type:"java.lang.String"> = var %5 @"s";
                %7 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%8 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %9 : java.type:"boolean" = pattern.match %8
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" -> {
                                %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" = pattern.record %10 @java.ref:"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %4 %12;
                                yield;
                            };
                        yield %9;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %13 : java.type:"java.lang.Number" = var.load %4;
                        java.yield %13;
                    }
                    (%14 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %15 : java.type:"boolean" = pattern.match %14
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %16;
                            }
                            (%17 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %6 %17;
                                yield;
                            };
                        yield %15;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %18 : java.type:"java.lang.String" = var.load %6;
                        java.yield %18;
                    }
                    ()java.type:"boolean" -> {
                        %19 : java.type:"boolean" = constant @true;
                        yield %19;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %20 : java.type:"java.lang.String" = constant @"";
                        java.yield %20;
                    };
                return %7;
            };
            """)
    public static Object patternCaseLabel(Object r) {
        return switch (r) {
            case A(Number n) -> {
                yield n;
            }
            case String s -> {
                yield s;
            }
            default -> {
                yield "";
            }
        };
    }

    @CodeReflection
    @IR("""
            func @"patternCaseLabelGuard" (%0 : java.type:"java.lang.Object")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"r";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.Number" = constant @null;
                %4 : Var<java.type:"java.lang.Number"> = var %3 @"n";
                %5 : java.type:"java.lang.String" = constant @null;
                %6 : Var<java.type:"java.lang.String"> = var %5 @"s";
                %7 : java.type:"java.lang.String" = constant @null;
                %8 : Var<java.type:"java.lang.String"> = var %7 @"s";
                %9 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%10 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %11 : java.type:"boolean" = pattern.match %10
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" -> {
                                %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" = pattern.record %12 @java.ref:"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %13;
                            }
                            (%14 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %4 %14;
                                yield;
                            };
                        yield %11;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %15 : java.type:"java.lang.Number" = var.load %4;
                        java.yield %15;
                    }
                    (%16 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %18 : java.type:"boolean" = pattern.match %16
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %19 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %19;
                                    }
                                    (%20 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %6 %20;
                                        yield;
                                    };
                                yield %18;
                            }
                            ()java.type:"boolean" -> {
                                %21 : java.type:"java.lang.String" = var.load %6;
                                %22 : java.type:"int" = invoke %21 @java.ref:"java.lang.String::length():int";
                                %23 : java.type:"int" = constant @5;
                                %24 : java.type:"boolean" = lt %22 %23;
                                yield %24;
                            };
                        yield %17;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %25 : java.type:"java.lang.String" = var.load %6;
                        java.yield %25;
                    }
                    (%26 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %27 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %28 : java.type:"boolean" = pattern.match %26
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %29 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %29;
                                    }
                                    (%30 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %8 %30;
                                        yield;
                                    };
                                yield %28;
                            }
                            ()java.type:"boolean" -> {
                                %31 : java.type:"java.lang.String" = var.load %8;
                                %32 : java.type:"int" = invoke %31 @java.ref:"java.lang.String::length():int";
                                %33 : java.type:"int" = constant @10;
                                %34 : java.type:"boolean" = lt %32 %33;
                                yield %34;
                            };
                        yield %27;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %35 : java.type:"java.lang.String" = var.load %8;
                        java.yield %35;
                    }
                    ()java.type:"boolean" -> {
                        %36 : java.type:"boolean" = constant @true;
                        yield %36;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %37 : java.type:"java.lang.String" = constant @"";
                        java.yield %37;
                    };
                return %9;
            };
            """)
    public static Object patternCaseLabelGuard(Object r) {
        return switch (r) {
            case A(Number n) -> {
                yield n;
            }
            case String s when s.length() < 5 -> {
                yield s;
            }
            case String s when s.length() < 10 -> {
                yield s;
            }
            default -> {
                yield "";
            }
        };
    }
}
