/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
    @IR("""
            func @"patternCaseLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"r";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : Var<java.type:"java.lang.Number"> = var @"n";
                %4 : Var<java.type:"java.lang.String"> = var @"s";
                %5 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%6 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %7 : java.type:"boolean" = pattern.match %6
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" -> {
                                %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" = pattern.record %8 @java.ref:"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %3 %10;
                                yield;
                            };
                        yield %7;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %11 : java.type:"java.lang.Number" = var.load %3;
                        java.yield %11;
                    }
                    (%12 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %13 : java.type:"boolean" = pattern.match %12
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %14;
                            }
                            (%15 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %4 %15;
                                yield;
                            };
                        yield %13;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %16 : java.type:"java.lang.String" = var.load %4;
                        java.yield %16;
                    }
                    ()java.type:"boolean" -> {
                        %17 : java.type:"boolean" = constant @true;
                        yield %17;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %18 : java.type:"java.lang.String" = constant @"";
                        java.yield %18;
                    };
                return %5;
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

    @Reflect
    @IR("""
            func @"patternCaseLabelGuard" (%0 : java.type:"java.lang.Object")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"r";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : Var<java.type:"java.lang.Number"> = var @"n";
                %4 : Var<java.type:"java.lang.String"> = var @"s";
                %5 : Var<java.type:"java.lang.String"> = var @"s";
                %6 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchExpressionTest$A>" = pattern.record %9 @java.ref:"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %10;
                            }
                            (%11 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %3 %11;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %12 : java.type:"java.lang.Number" = var.load %3;
                        java.yield %12;
                    }
                    (%13 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %14 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %15 : java.type:"boolean" = pattern.match %13
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %16;
                                    }
                                    (%17 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %4 %17;
                                        yield;
                                    };
                                yield %15;
                            }
                            ()java.type:"boolean" -> {
                                %18 : java.type:"java.lang.String" = var.load %4;
                                %19 : java.type:"int" = invoke %18 @java.ref:"java.lang.String::length():int";
                                %20 : java.type:"int" = constant @5;
                                %21 : java.type:"boolean" = lt %19 %20;
                                yield %21;
                            };
                        yield %14;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %22 : java.type:"java.lang.String" = var.load %4;
                        java.yield %22;
                    }
                    (%23 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %24 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %25 : java.type:"boolean" = pattern.match %23
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %26 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %26;
                                    }
                                    (%27 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %5 %27;
                                        yield;
                                    };
                                yield %25;
                            }
                            ()java.type:"boolean" -> {
                                %28 : java.type:"java.lang.String" = var.load %5;
                                %29 : java.type:"int" = invoke %28 @java.ref:"java.lang.String::length():int";
                                %30 : java.type:"int" = constant @10;
                                %31 : java.type:"boolean" = lt %29 %30;
                                yield %31;
                            };
                        yield %24;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %32 : java.type:"java.lang.String" = var.load %5;
                        java.yield %32;
                    }
                    ()java.type:"boolean" -> {
                        %33 : java.type:"boolean" = constant @true;
                        yield %33;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %34 : java.type:"java.lang.String" = constant @"";
                        java.yield %34;
                    };
                return %6;
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

    @IR("""
            func @"caseBoxedGuard" (%0 : java.type:"java.lang.Object", %1 : java.type:"java.lang.Boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %3 : Var<java.type:"java.lang.Boolean"> = var %1 @"B";
                %4 : java.type:"java.lang.Object" = var.load %2;
                %5 : Var<java.type:"java.lang.Object"> = var;
                %6 : java.type:"java.lang.String" = java.switch.expression %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %9 : java.type:"boolean" = pattern.match %7
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                        %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type;
                                        yield %10;
                                    }
                                    (%11 : java.type:"java.lang.Object")java.type:"void" -> {
                                        var.store %5 %11;
                                        yield;
                                    };
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %12 : java.type:"java.lang.Boolean" = var.load %3;
                                %13 : java.type:"boolean" = invoke %12 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                                yield %13;
                            };
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %14 : java.type:"java.lang.String" = constant @"match";
                        yield %14;
                    }
                    ()java.type:"boolean" -> {
                        %15 : java.type:"boolean" = constant @true;
                        yield %15;
                    }
                    ()java.type:"java.lang.String" -> {
                        %16 : java.type:"java.lang.String" = constant @"no match";
                        yield %16;
                    };
                return %6;
            };
            """)
    @Reflect
    private static String caseBoxedGuard(Object o, Boolean B) {
        return switch (o) {
            case Object _ when B -> "match";
            default -> "no match";
        };
    }

    @IR("""
            func @"statementGroupsYieldingBoolean" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"value";
                java.if
                    ()java.type:"boolean" -> {
                        %2 : java.type:"int" = var.load %1;
                        %3 : java.type:"boolean" = java.switch.expression %2
                            (%4 : java.type:"int")java.type:"boolean" -> {
                                %5 : java.type:"int" = constant @0;
                                %6 : java.type:"boolean" = eq %4 %5;
                                yield %6;
                            }
                            ()java.type:"boolean" -> {
                                %7 : java.type:"boolean" = constant @false;
                                java.yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"boolean" = constant @true;
                                yield %8;
                            }
                            ()java.type:"boolean" -> {
                                %9 : java.type:"int" = var.load %1;
                                %10 : java.type:"int" = constant @0;
                                %11 : java.type:"boolean" = gt %9 %10;
                                java.yield %11;
                            };
                        yield %3;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"int" = constant @1;
                        return %12;
                    };
                %13 : java.type:"int" = constant @0;
                return %13;
            };
            """)
    @Reflect
    private static int statementGroupsYieldingBoolean(int value) {
        if (switch (value) {
            case 0: yield false;
            default: yield value > 0;}) {
            return 1;
        }
        return 0;
    }

    @IR("""
            func @"statementGroupsYieldingObjects" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"value";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"java.lang.Object" = java.switch.expression %2
                    (%4 : java.type:"int")java.type:"boolean" -> {
                        %5 : java.type:"int" = constant @0;
                        %6 : java.type:"boolean" = eq %4 %5;
                        yield %6;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %7 : java.type:"int" = constant @2147483647;
                        %8 : java.type:"java.lang.Integer" = invoke %7 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        java.yield %8;
                    }
                    ()java.type:"boolean" -> {
                        %9 : java.type:"boolean" = constant @true;
                        yield %9;
                    }
                    ()java.type:"java.lang.Object" -> {
                        %10 : java.type:"java.lang.String" = constant @"default";
                        java.yield %10;
                    };
                %11 : Var<java.type:"java.lang.Object"> = var %3 @"r";
                return;
            };
            """)
    @Reflect
    private static void statementGroupsYieldingObjects(int value) {
        Object r = switch (value) {
            case 0: yield Integer.MAX_VALUE;
            default: yield "default";
        };
    }
}
