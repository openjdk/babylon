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

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @build SwitchExpressionTest2
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchExpressionTest2
 */
public class SwitchExpressionTest2 {

    @IR("""
            func @"caseConstantRuleExpression" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"BAR";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"BAZ";
                        yield %11;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"java.lang.String" = constant @"BAZ";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"FOO";
                        yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"";
                        yield %17;
                    };
                return %3;
            };
            """)
    @Reflect
    public static String caseConstantRuleExpression(String r) {
        return switch (r) {
            case "FOO" -> "BAR";
            case "BAR" -> "BAZ";
            case "BAZ" -> "FOO";
            default -> "";
        };
    }

    @IR("""
            func @"caseConstantRuleBlock" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"BAR";
                        java.yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"BAZ";
                        java.yield %11;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"java.lang.String" = constant @"BAZ";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"";
                        java.yield %17;
                    };
                return %3;
            };
            """)
    @Reflect
    public static String caseConstantRuleBlock(String r) {
        return switch (r) {
            case "FOO" -> {
                yield "BAR";
            }
            case "BAR" -> {
                yield "BAZ";
            }
            case "BAZ" -> {
                yield "FOO";
            }
            default -> {
                yield "";
            }
        };
    }

    @IR("""
            func @"caseConstantStatement" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"FOO";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"BAR";
                        java.yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"BAZ";
                        java.yield %11;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"java.lang.String" = constant @"BAZ";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"FOO";
                        java.yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"";
                        java.yield %17;
                    };
                return %3;
            };
            """)
    @Reflect
    private static String caseConstantStatement(String s) {
        return switch (s) {
            case "FOO": yield "BAR";
            case "BAR": yield "BAZ";
            case "BAZ": yield "FOO";
            default: yield "";
        };
    }

    @IR("""
            func @"caseConstantMultiLabels" (%0 : java.type:"char")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"char"> = var %0 @"c";
                %2 : java.type:"char" = var.load %1;
                %3 : java.type:"char" = invoke %2 @java.ref:"java.lang.Character::toLowerCase(char):char";
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"char")java.type:"boolean" -> {
                        %6 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %7 : java.type:"char" = constant @'a';
                                %8 : java.type:"boolean" = eq %5 %7;
                                yield %8;
                            }
                            ()java.type:"boolean" -> {
                                %9 : java.type:"char" = constant @'e';
                                %10 : java.type:"boolean" = eq %5 %9;
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %11 : java.type:"char" = constant @'i';
                                %12 : java.type:"boolean" = eq %5 %11;
                                yield %12;
                            }
                            ()java.type:"boolean" -> {
                                %13 : java.type:"char" = constant @'o';
                                %14 : java.type:"boolean" = eq %5 %13;
                                yield %14;
                            }
                            ()java.type:"boolean" -> {
                                %15 : java.type:"char" = constant @'u';
                                %16 : java.type:"boolean" = eq %5 %15;
                                yield %16;
                            };
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"vowel";
                        java.yield %17;
                    }
                    ()java.type:"boolean" -> {
                        %18 : java.type:"boolean" = constant @true;
                        yield %18;
                    }
                    ()java.type:"java.lang.String" -> {
                        %19 : java.type:"java.lang.String" = constant @"consonant";
                        java.yield %19;
                    };
                return %4;
            };
            """)
    @Reflect
    private static String caseConstantMultiLabels(char c) {
        return switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u': yield "vowel";
            default: yield "consonant";
        };
    }

    @IR("""
            func @"casePatternMultiLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : Var<java.type:"java.lang.Integer"> = var;
                %4 : Var<java.type:"java.lang.Long"> = var;
                %5 : Var<java.type:"java.lang.Character"> = var;
                %6 : Var<java.type:"java.lang.Byte"> = var;
                %7 : Var<java.type:"java.lang.Short"> = var;
                %8 : java.type:"java.lang.String" = java.switch.expression %2
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %11 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type;
                                        yield %12;
                                    }
                                    (%13 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %3 %13;
                                        yield;
                                    };
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %14 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" -> {
                                        %15 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" = pattern.type;
                                        yield %15;
                                    }
                                    (%16 : java.type:"java.lang.Long")java.type:"void" -> {
                                        var.store %4 %16;
                                        yield;
                                    };
                                yield %14;
                            }
                            ()java.type:"boolean" -> {
                                %17 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Character>" -> {
                                        %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Character>" = pattern.type;
                                        yield %18;
                                    }
                                    (%19 : java.type:"java.lang.Character")java.type:"void" -> {
                                        var.store %5 %19;
                                        yield;
                                    };
                                yield %17;
                            }
                            ()java.type:"boolean" -> {
                                %20 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" -> {
                                        %21 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" = pattern.type;
                                        yield %21;
                                    }
                                    (%22 : java.type:"java.lang.Byte")java.type:"void" -> {
                                        var.store %6 %22;
                                        yield;
                                    };
                                yield %20;
                            }
                            ()java.type:"boolean" -> {
                                %23 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" -> {
                                        %24 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" = pattern.type;
                                        yield %24;
                                    }
                                    (%25 : java.type:"java.lang.Short")java.type:"void" -> {
                                        var.store %7 %25;
                                        yield;
                                    };
                                yield %23;
                            };
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %26 : java.type:"java.lang.String" = constant @"integral type";
                        yield %26;
                    }
                    ()java.type:"boolean" -> {
                        %27 : java.type:"boolean" = constant @true;
                        yield %27;
                    }
                    ()java.type:"java.lang.String" -> {
                        %28 : java.type:"java.lang.String" = constant @"non integral type";
                        yield %28;
                    };
                return %8;
            };
            """)
    @Reflect
    private static String casePatternMultiLabel(Object o) {
        return switch (o) {
            case Integer _, Long _, Character _, Byte _, Short _-> "integral type";
            default -> "non integral type";
        };
    }

    @IR("""
            func @"casePatternGuardedMultiLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : Var<java.type:"java.lang.Integer"> = var;
                %4 : Var<java.type:"java.lang.Long"> = var;
                %5 : Var<java.type:"java.lang.Byte"> = var;
                %6 : Var<java.type:"java.lang.Short"> = var;
                %7 : java.type:"java.lang.String" = java.switch.expression %2
                    (%8 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %9 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %10 : java.type:"boolean" = java.cor
                                    ()java.type:"boolean" -> {
                                        %11 : java.type:"boolean" = pattern.match %8
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                                %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type;
                                                yield %12;
                                            }
                                            (%13 : java.type:"java.lang.Integer")java.type:"void" -> {
                                                var.store %3 %13;
                                                yield;
                                            };
                                        yield %11;
                                    }
                                    ()java.type:"boolean" -> {
                                        %14 : java.type:"boolean" = pattern.match %8
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" -> {
                                                %15 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" = pattern.type;
                                                yield %15;
                                            }
                                            (%16 : java.type:"java.lang.Long")java.type:"void" -> {
                                                var.store %4 %16;
                                                yield;
                                            };
                                        yield %14;
                                    }
                                    ()java.type:"boolean" -> {
                                        %17 : java.type:"boolean" = pattern.match %8
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" -> {
                                                %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" = pattern.type;
                                                yield %18;
                                            }
                                            (%19 : java.type:"java.lang.Byte")java.type:"void" -> {
                                                var.store %5 %19;
                                                yield;
                                            };
                                        yield %17;
                                    }
                                    ()java.type:"boolean" -> {
                                        %20 : java.type:"boolean" = pattern.match %8
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" -> {
                                                %21 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" = pattern.type;
                                                yield %21;
                                            }
                                            (%22 : java.type:"java.lang.Short")java.type:"void" -> {
                                                var.store %6 %22;
                                                yield;
                                            };
                                        yield %20;
                                    };
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %23 : java.type:"java.lang.Object" = var.load %1;
                                %24 : java.type:"java.lang.Number" = cast %23 @java.type:"java.lang.Number";
                                %25 : java.type:"int" = invoke %24 @java.ref:"java.lang.Number::intValue():int";
                                %26 : java.type:"int" = constant @0;
                                %27 : java.type:"boolean" = gt %25 %26;
                                yield %27;
                            };
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %28 : java.type:"java.lang.String" = constant @"integral type";
                        yield %28;
                    }
                    ()java.type:"boolean" -> {
                        %29 : java.type:"boolean" = constant @true;
                        yield %29;
                    }
                    ()java.type:"java.lang.String" -> {
                        %30 : java.type:"java.lang.String" = constant @"non integral type";
                        yield %30;
                    };
                return %7;
            };
            """)
    @Reflect
    private static String casePatternGuardedMultiLabel(Object o) {
        return switch (o) {
            case Integer _, Long _, Byte _, Short _ when ((Number)o).intValue() > 0 -> "integral type";
            default -> "non integral type";
        };
    }

    @IR("""
            func @"caseConstantThrow" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                %2 : java.type:"java.lang.Integer" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.Integer::intValue():int";
                        %6 : java.type:"int" = constant @8;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %8;
                    }
                    (%9 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %10 : java.type:"int" = invoke %9 @java.ref:"java.lang.Integer::intValue():int";
                        %11 : java.type:"int" = constant @9;
                        %12 : java.type:"boolean" = eq %10 %11;
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @"NINE";
                        yield %13;
                    }
                    ()java.type:"boolean" -> {
                        %14 : java.type:"boolean" = constant @true;
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"An integer";
                        yield %15;
                    };
                return %3;
            };
            """)
    @Reflect
    private static String caseConstantThrow(Integer i) {
        return switch (i) {
            case 8 -> throw new IllegalArgumentException();
            case 9 -> "NINE";
            default -> "An integer";
        };
    }

    @IR("""
            func @"caseConstantNullLabel" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2 @switch.handle.nulls=true
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.Object" = constant @null;
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"null";
                        yield %7;
                    }
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = constant @true;
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"non null";
                        yield %9;
                    };
                return %3;
            };
            """)
    @Reflect
    private static String caseConstantNullLabel(String s) {
        return switch (s) {
            case null -> "null";
            default -> "non null";
        };
    }

    @IR("""
            func @"caseConstantNullAndDefault" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2 @switch.handle.nulls=true
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"abc";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"alphabet";
                        yield %7;
                    }
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = constant @true;
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"null or default";
                        yield %9;
                    };
                return %3;
            };
            """)
    @Reflect
    private static String caseConstantNullAndDefault(String s) {
        return switch (s) {
            case "abc" -> "alphabet";
            case null, default -> "null or default";
        };
    }

    @IR("""
            func @"caseConstantFallThrough" (%0 : java.type:"char")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"char"> = var %0 @"c";
                %2 : java.type:"char" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"char")java.type:"boolean" -> {
                        %5 : java.type:"char" = constant @'A';
                        %6 : java.type:"boolean" = eq %4 %5;
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        java.switch.fallthrough;
                    }
                    (%7 : java.type:"char")java.type:"boolean" -> {
                        %8 : java.type:"char" = constant @'B';
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @"A or B";
                        java.yield %10;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"java.lang.String" -> {
                        %12 : java.type:"java.lang.String" = constant @"Neither A nor B";
                        java.yield %12;
                    };
                return %3;
            };
            """)
    @Reflect
    private static String caseConstantFallThrough(char c) {
        return switch (c) {
            case 'A':
            case 'B':
                yield "A or B";
            default:
                yield "Neither A nor B";
        };
    }

    enum Day {
        MON, TUE, WED, THU, FRI, SAT, SUN
    }
    @IR("""
            func @"caseConstantEnum" (%0 : java.type:"SwitchExpressionTest2$Day")java.type:"int" -> {
                %1 : Var<java.type:"SwitchExpressionTest2$Day"> = var %0 @"d";
                %2 : java.type:"SwitchExpressionTest2$Day" = var.load %1;
                %3 : java.type:"int" = java.switch.expression %2
                    (%4 : java.type:"SwitchExpressionTest2$Day")java.type:"boolean" -> {
                        %5 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %6 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::MON:SwitchExpressionTest2$Day";
                                %7 : java.type:"boolean" = eq %4 %6;
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::FRI:SwitchExpressionTest2$Day";
                                %9 : java.type:"boolean" = eq %4 %8;
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %10 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::SUN:SwitchExpressionTest2$Day";
                                %11 : java.type:"boolean" = eq %4 %10;
                                yield %11;
                            };
                        yield %5;
                    }
                    ()java.type:"int" -> {
                        %12 : java.type:"int" = constant @6;
                        yield %12;
                    }
                    (%13 : java.type:"SwitchExpressionTest2$Day")java.type:"boolean" -> {
                        %14 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::TUE:SwitchExpressionTest2$Day";
                        %15 : java.type:"boolean" = eq %13 %14;
                        yield %15;
                    }
                    ()java.type:"int" -> {
                        %16 : java.type:"int" = constant @7;
                        yield %16;
                    }
                    (%17 : java.type:"SwitchExpressionTest2$Day")java.type:"boolean" -> {
                        %18 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %19 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::THU:SwitchExpressionTest2$Day";
                                %20 : java.type:"boolean" = eq %17 %19;
                                yield %20;
                            }
                            ()java.type:"boolean" -> {
                                %21 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::SAT:SwitchExpressionTest2$Day";
                                %22 : java.type:"boolean" = eq %17 %21;
                                yield %22;
                            };
                        yield %18;
                    }
                    ()java.type:"int" -> {
                        %23 : java.type:"int" = constant @8;
                        yield %23;
                    }
                    (%24 : java.type:"SwitchExpressionTest2$Day")java.type:"boolean" -> {
                        %25 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::WED:SwitchExpressionTest2$Day";
                        %26 : java.type:"boolean" = eq %24 %25;
                        yield %26;
                    }
                    ()java.type:"int" -> {
                        %27 : java.type:"int" = constant @9;
                        yield %27;
                    }
                    ()java.type:"boolean" -> {
                        %28 : java.type:"boolean" = constant @true;
                        yield %28;
                    }
                    ()java.type:"int" -> {
                        %29 : java.type:"java.lang.String" = constant @null;
                        %30 : java.type:"java.lang.Throwable" = constant @null;
                        %31 : java.type:"java.lang.MatchException" = new %29 %30 @java.ref:"java.lang.MatchException::(java.lang.String, java.lang.Throwable)";
                        throw %31;
                    };
                return %3;
            };
            """)
    @Reflect
    private static int caseConstantEnum(Day d) {
        return switch (d) {
            case MON, FRI, SUN -> 6;
            case TUE -> 7;
            case THU, SAT -> 8;
            case WED -> 9;
        };
    }

    static class Constants {
        static final int c1 = 12;
    }
    @IR("""
            func @"caseConstantOtherKindsOfExpr" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = constant @11;
                %3 : java.type:"int" = var.load %1;
                %4 : java.type:"java.lang.String" = java.switch.expression %3
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @"1";
                        yield %8;
                    }
                    (%9 : java.type:"int")java.type:"boolean" -> {
                        %10 : java.type:"int" = constant @2;
                        %11 : java.type:"boolean" = eq %9 %10;
                        yield %11;
                    }
                    ()java.type:"java.lang.String" -> {
                        %12 : java.type:"java.lang.String" = constant @"2";
                        yield %12;
                    }
                    (%13 : java.type:"int")java.type:"boolean" -> {
                        %14 : java.type:"int" = constant @3;
                        %15 : java.type:"boolean" = eq %13 %14;
                        yield %15;
                    }
                    ()java.type:"java.lang.String" -> {
                        %16 : java.type:"java.lang.String" = constant @"3";
                        yield %16;
                    }
                    (%17 : java.type:"int")java.type:"boolean" -> {
                        %18 : java.type:"int" = constant @4;
                        %19 : java.type:"boolean" = eq %17 %18;
                        yield %19;
                    }
                    ()java.type:"java.lang.String" -> {
                        %20 : java.type:"java.lang.String" = constant @"4";
                        yield %20;
                    }
                    (%21 : java.type:"int")java.type:"boolean" -> {
                        %22 : java.type:"int" = constant @5;
                        %23 : java.type:"boolean" = eq %21 %22;
                        yield %23;
                    }
                    ()java.type:"java.lang.String" -> {
                        %24 : java.type:"java.lang.String" = constant @"5";
                        yield %24;
                    }
                    (%25 : java.type:"int")java.type:"boolean" -> {
                        %26 : java.type:"int" = constant @6;
                        %27 : java.type:"boolean" = eq %25 %26;
                        yield %27;
                    }
                    ()java.type:"java.lang.String" -> {
                        %28 : java.type:"java.lang.String" = constant @"6";
                        yield %28;
                    }
                    (%29 : java.type:"int")java.type:"boolean" -> {
                        %30 : java.type:"int" = constant @7;
                        %31 : java.type:"boolean" = eq %29 %30;
                        yield %31;
                    }
                    ()java.type:"java.lang.String" -> {
                        %32 : java.type:"java.lang.String" = constant @"7";
                        yield %32;
                    }
                    (%33 : java.type:"int")java.type:"boolean" -> {
                        %34 : java.type:"int" = constant @8;
                        %35 : java.type:"boolean" = eq %33 %34;
                        yield %35;
                    }
                    ()java.type:"java.lang.String" -> {
                        %36 : java.type:"java.lang.String" = constant @"8";
                        yield %36;
                    }
                    (%37 : java.type:"int")java.type:"boolean" -> {
                        %38 : java.type:"int" = constant @9;
                        %39 : java.type:"boolean" = eq %37 %38;
                        yield %39;
                    }
                    ()java.type:"java.lang.String" -> {
                        %40 : java.type:"java.lang.String" = constant @"9";
                        yield %40;
                    }
                    (%41 : java.type:"int")java.type:"boolean" -> {
                        %42 : java.type:"int" = constant @10;
                        %43 : java.type:"boolean" = eq %41 %42;
                        yield %43;
                    }
                    ()java.type:"java.lang.String" -> {
                        %44 : java.type:"java.lang.String" = constant @"10";
                        yield %44;
                    }
                    (%45 : java.type:"int")java.type:"boolean" -> {
                        %46 : java.type:"int" = constant @11;
                        %47 : java.type:"boolean" = eq %45 %46;
                        yield %47;
                    }
                    ()java.type:"java.lang.String" -> {
                        %48 : java.type:"java.lang.String" = constant @"11";
                        yield %48;
                    }
                    (%49 : java.type:"int")java.type:"boolean" -> {
                        %50 : java.type:"int" = constant @12;
                        %51 : java.type:"boolean" = eq %49 %50;
                        yield %51;
                    }
                    ()java.type:"java.lang.String" -> {
                        %52 : java.type:"int" = constant @12;
                        %53 : java.type:"java.lang.String" = invoke %52 @java.ref:"java.lang.String::valueOf(int):java.lang.String";
                        yield %53;
                    }
                    (%54 : java.type:"int")java.type:"boolean" -> {
                        %55 : java.type:"int" = constant @13;
                        %56 : java.type:"boolean" = eq %54 %55;
                        yield %56;
                    }
                    ()java.type:"java.lang.String" -> {
                        %57 : java.type:"java.lang.String" = constant @"13";
                        yield %57;
                    }
                    ()java.type:"boolean" -> {
                        %58 : java.type:"boolean" = constant @true;
                        yield %58;
                    }
                    ()java.type:"java.lang.String" -> {
                        %59 : java.type:"java.lang.String" = constant @"an int";
                        yield %59;
                    };
                return %4;
            };
            """)
    @Reflect
    private static String caseConstantOtherKindsOfExpr(int i) {
        final int eleven = 11;
        return switch (i) {
            case 1 & 0xF -> "1";
            case 4>>1 -> "2";
            case (int) 3L -> "3";
            case 2<<1 -> "4";
            case 10 / 2 -> "5";
            case 12 - 6 -> "6";
            case 3 + 4 -> "7";
            case 2 * 2 * 2 -> "8";
            case 8 | 1 -> "9";
            case (10) -> "10";
            case eleven -> "11";
            case Constants.c1 -> String.valueOf(Constants.c1);
            case 1 > 0 ? 13 : 133 -> "13";
            default -> "an int";
        };
    }

    // these are the conversions that applies in switch

    @IR("""
            func @"caseConstantConv" (%0 : java.type:"short")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"short"> = var %0 @"a";
                %2 : java.type:"int" = constant @1;
                %3 : java.type:"short" = conv %2;
                %4 : java.type:"int" = constant @2;
                %5 : java.type:"byte" = conv %4;
                %6 : java.type:"short" = var.load %1;
                %7 : java.type:"java.lang.String" = java.switch.expression %6
                    (%8 : java.type:"short")java.type:"boolean" -> {
                        %9 : java.type:"short" = constant @1;
                        %10 : java.type:"boolean" = eq %8 %9;
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"one";
                        yield %11;
                    }
                    (%12 : java.type:"short")java.type:"boolean" -> {
                        %13 : java.type:"short" = constant @2;
                        %14 : java.type:"boolean" = eq %12 %13;
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"three";
                        yield %15;
                    }
                    (%16 : java.type:"short")java.type:"boolean" -> {
                        %17 : java.type:"short" = constant @3;
                        %18 : java.type:"boolean" = eq %16 %17;
                        yield %18;
                    }
                    ()java.type:"java.lang.String" -> {
                        %19 : java.type:"java.lang.String" = constant @"two";
                        yield %19;
                    }
                    ()java.type:"boolean" -> {
                        %20 : java.type:"boolean" = constant @true;
                        yield %20;
                    }
                    ()java.type:"java.lang.String" -> {
                        %21 : java.type:"java.lang.String" = constant @"default";
                        yield %21;
                    };
                return %7;
            };
            """)
    @Reflect
    static String caseConstantConv(short a) {
        final short s = 1;
        final byte b = 2;
        return switch (a) {
            case s -> "one"; // identity
            case b -> "three"; // widening primitive conversion
            case 3 -> "two"; // narrowing primitive conversion
            default -> "default";
        };
    }

    @IR("""
            func @"caseConstantConv2" (%0 : java.type:"java.lang.Byte")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Byte"> = var %0 @"a";
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"byte" = conv %2;
                %4 : java.type:"java.lang.Byte" = var.load %1;
                %5 : java.type:"java.lang.String" = java.switch.expression %4
                    (%6 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %7 : java.type:"byte" = invoke %6 @java.ref:"java.lang.Byte::byteValue():byte";
                        %8 : java.type:"byte" = constant @1;
                        %9 : java.type:"boolean" = eq %7 %8;
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @"one";
                        yield %10;
                    }
                    (%11 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %12 : java.type:"byte" = invoke %11 @java.ref:"java.lang.Byte::byteValue():byte";
                        %13 : java.type:"byte" = constant @2;
                        %14 : java.type:"boolean" = eq %12 %13;
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"two";
                        yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"default";
                        yield %17;
                    };
                return %5;
            };
            """)
    @Reflect
    static String caseConstantConv2(Byte a) {
        final byte b = 2;
        return switch (a) {
            // narrowing conv is missing in the code model
            case 1 -> "one"; // narrowing primitive conversion followed by a boxing conversion
            case b -> "two"; // boxing
            default -> "default";
        };
    }

    enum E { F, G }
    @IR("""
            func @"noDefaultLabelEnum" (%0 : java.type:"SwitchExpressionTest2$E")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"SwitchExpressionTest2$E"> = var %0 @"e";
                %2 : java.type:"SwitchExpressionTest2$E" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"SwitchExpressionTest2$E")java.type:"boolean" -> {
                        %5 : java.type:"SwitchExpressionTest2$E" = field.load @java.ref:"SwitchExpressionTest2$E::F:SwitchExpressionTest2$E";
                        %6 : java.type:"boolean" = eq %4 %5;
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"f";
                        yield %7;
                    }
                    (%8 : java.type:"SwitchExpressionTest2$E")java.type:"boolean" -> {
                        %9 : java.type:"SwitchExpressionTest2$E" = field.load @java.ref:"SwitchExpressionTest2$E::G:SwitchExpressionTest2$E";
                        %10 : java.type:"boolean" = eq %8 %9;
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"g";
                        yield %11;
                    }
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = constant @true;
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @null;
                        %14 : java.type:"java.lang.Throwable" = constant @null;
                        %15 : java.type:"java.lang.MatchException" = new %13 %14 @java.ref:"java.lang.MatchException::(java.lang.String, java.lang.Throwable)";
                        throw %15;
                    };
                return %3;
            };
            """)
    @Reflect
    static String noDefaultLabelEnum(E e) {
        return switch (e) {
            case F -> "f";
            case G -> "g";
        };
    }

    @IR("""
            func @"unconditionalPattern" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : Var<java.type:"java.lang.Object"> = var @"o";
                %4 : java.type:"java.lang.String" = java.switch.expression %2
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"A";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.String" = constant @"Alphabet";
                        yield %8;
                    }
                    (%9 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type @"o";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Object")java.type:"void" -> {
                                var.store %3 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @"default";
                        yield %13;
                    };
                return %4;
            };
            """)
    @Reflect
    static String unconditionalPattern(String s) {
        return switch (s) {
            case "A" -> "Alphabet";
            case Object o -> "default";
        };
    }

    sealed interface A permits B, C {}
    record B() implements A {}
    final class C implements A {}
    @IR("""
            func @"noDefault" (%0 : java.type:"SwitchExpressionTest2$A")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"SwitchExpressionTest2$A"> = var %0 @"a";
                %2 : java.type:"SwitchExpressionTest2$A" = var.load %1;
                %3 : Var<java.type:"SwitchExpressionTest2$B"> = var @"b";
                %4 : Var<java.type:"SwitchExpressionTest2::C"> = var @"c";
                %5 : java.type:"java.lang.String" = java.switch.expression %2
                    (%6 : java.type:"SwitchExpressionTest2$A")java.type:"boolean" -> {
                        %7 : java.type:"boolean" = pattern.match %6
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2$B>" -> {
                                %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2$B>" = pattern.type @"b";
                                yield %8;
                            }
                            (%9 : java.type:"SwitchExpressionTest2$B")java.type:"void" -> {
                                var.store %3 %9;
                                yield;
                            };
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %10 : java.type:"java.lang.String" = constant @"B";
                        yield %10;
                    }
                    (%11 : java.type:"SwitchExpressionTest2$A")java.type:"boolean" -> {
                        %12 : java.type:"boolean" = pattern.match %11
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2::C>" -> {
                                %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2::C>" = pattern.type @"c";
                                yield %13;
                            }
                            (%14 : java.type:"SwitchExpressionTest2::C")java.type:"void" -> {
                                var.store %4 %14;
                                yield;
                            };
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %15 : java.type:"java.lang.String" = constant @"C";
                        yield %15;
                    }
                    ()java.type:"boolean" -> {
                        %16 : java.type:"boolean" = constant @true;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @null;
                        %18 : java.type:"java.lang.Throwable" = constant @null;
                        %19 : java.type:"java.lang.MatchException" = new %17 %18 @java.ref:"java.lang.MatchException::(java.lang.String, java.lang.Throwable)";
                        throw %19;
                    };
                return %5;
            };
            """)
    @Reflect
    static String noDefault(A a) {
        return switch (a) {
            case B b -> "B";
            case C c -> "C";
        };
    }

    @IR("""
            func @"defaultNotTheLastLabel" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = constant @true;
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @"else";
                        yield %13;
                    }
                    (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %5 : java.type:"java.lang.String" = constant @"M";
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"Mow";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %9 : java.type:"java.lang.String" = constant @"A";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"Aow";
                        yield %11;
                    };
                return %3;
            };
            """)
    @Reflect
    static String defaultNotTheLastLabel(String s) {
        return switch (s) {
            default -> "else";
            case "M" -> "Mow";
            case "A" -> "Aow";
        };
    }

    @IR("""
            func @"caseConstantPrimitiveWrapperSelector" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                  %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                  %2 : java.type:"java.lang.Integer" = var.load %1;
                  %3 : java.type:"java.lang.String" = java.switch.expression %2
                      (%4 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                          %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.Integer::intValue():int";
                          %6 : java.type:"int" = constant @1;
                          %7 : java.type:"boolean" = eq %5 %6;
                          yield %7;
                      }
                      ()java.type:"java.lang.String" -> {
                          %8 : java.type:"java.lang.String" = constant @"one";
                          yield %8;
                      }
                      (%9 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                          %10 : java.type:"boolean" = java.cor
                              ()java.type:"boolean" -> {
                                  %11 : java.type:"int" = invoke %9 @java.ref:"java.lang.Integer::intValue():int";
                                  %12 : java.type:"int" = constant @2;
                                  %13 : java.type:"boolean" = eq %11 %12;
                                  yield %13;
                              }
                              ()java.type:"boolean" -> {
                                  %14 : java.type:"int" = invoke %9 @java.ref:"java.lang.Integer::intValue():int";
                                  %15 : java.type:"int" = constant @3;
                                  %16 : java.type:"boolean" = eq %14 %15;
                                  yield %16;
                              };
                          yield %10;
                      }
                      ()java.type:"java.lang.String" -> {
                          %17 : java.type:"java.lang.String" = constant @"two or three";
                          yield %17;
                      }
                      ()java.type:"boolean" -> {
                          %18 : java.type:"boolean" = constant @true;
                          yield %18;
                      }
                      ()java.type:"java.lang.String" -> {
                          %19 : java.type:"java.lang.String" = constant @"else";
                          yield %19;
                      };
                  return %3;
            };
            """)
    @Reflect
    static String caseConstantPrimitiveWrapperSelector(Integer i) {
        return switch (i) {
            case 1 -> "one";
            case 2, 3 -> "two or three";
            default -> "else";
        };
    }

    @IR("""
            func @"constantLabelCasted" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"int")java.type:"boolean" -> {
                        %5 : java.type:"int" = constant @1;
                        %6 : java.type:"boolean" = eq %4 %5;
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"one";
                        yield %7;
                    }
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = constant @true;
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"not one";
                        yield %9;
                    };
                return %3;
            };
            """)
    @Reflect
    static String constantLabelCasted(int i) {
        return switch (i) {
            case (byte) 1 -> "one";
            default -> "not one";
        };
    }

    @IR("""
            func @"caseConstantStringLiteral" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                  %2 : java.type:"java.lang.String" = var.load %1;
                  %3 : java.type:"java.lang.String" = java.switch.expression %2
                      (%4 : java.type:"java.lang.String")java.type:"boolean" -> {
                          %5 : java.type:"java.lang.String" = constant @"1";
                          %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                          yield %6;
                      }
                      ()java.type:"java.lang.String" -> {
                          %7 : java.type:"java.lang.String" = constant @"one";
                          yield %7;
                      }
                      (%8 : java.type:"java.lang.String")java.type:"boolean" -> {
                          %9 : java.type:"boolean" = java.cor
                              ()java.type:"boolean" -> {
                                  %10 : java.type:"java.lang.String" = constant @"2";
                                  %11 : java.type:"boolean" = invoke %8 %10 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                  yield %11;
                              }
                              ()java.type:"boolean" -> {
                                  %12 : java.type:"java.lang.String" = constant @"3";
                                  %13 : java.type:"boolean" = invoke %8 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                  yield %13;
                              };
                          yield %9;
                      }
                      ()java.type:"java.lang.String" -> {
                          %14 : java.type:"java.lang.String" = constant @"two or three";
                          yield %14;
                      }
                      ()java.type:"boolean" -> {
                          %15 : java.type:"boolean" = constant @true;
                          yield %15;
                      }
                      ()java.type:"java.lang.String" -> {
                          %16 : java.type:"java.lang.String" = constant @"else";
                          yield %16;
                      };
                  return %3;
              };
            """)
    @Reflect
    static String caseConstantStringLiteral(String s) {
        return switch (s) {
            case "1" -> "one";
            case "2", "3" -> "two or three";
            default -> "else";
        };
    }

    @IR("""
            func @"casePatternWithCaseConstant" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : Var<java.type:"java.lang.Integer"> = var @"j";
                %4 : Var<java.type:"java.lang.Integer"> = var;
                %5 : java.type:"java.lang.String" = java.switch.expression %2
                    (%6 : java.type:"int")java.type:"boolean" -> {
                        %7 : java.type:"int" = constant @0;
                        %8 : java.type:"boolean" = eq %6 %7;
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"zero";
                        yield %9;
                    }
                    (%10 : java.type:"int")java.type:"boolean" -> {
                        %11 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %12 : java.type:"java.lang.Integer" = invoke %10 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                                %13 : java.type:"boolean" = pattern.match %12
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"j";
                                        yield %14;
                                    }
                                    (%15 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %3 %15;
                                        yield;
                                    };
                                yield %13;
                            }
                            ()java.type:"boolean" -> {
                                %16 : java.type:"java.lang.Integer" = var.load %3;
                                %17 : java.type:"int" = invoke %16 @java.ref:"java.lang.Integer::intValue():int";
                                %18 : java.type:"int" = constant @0;
                                %19 : java.type:"boolean" = gt %17 %18;
                                yield %19;
                            };
                        yield %11;
                    }
                    ()java.type:"java.lang.String" -> {
                        %20 : java.type:"java.lang.String" = constant @"positive";
                        yield %20;
                    }
                    (%21 : java.type:"int")java.type:"boolean" -> {
                        %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %23 : java.type:"boolean" = pattern.match %22
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %24 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type;
                                yield %24;
                            }
                            (%25 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %4 %25;
                                yield;
                            };
                        yield %23;
                    }
                    ()java.type:"java.lang.String" -> {
                        %26 : java.type:"java.lang.String" = constant @"negative";
                        yield %26;
                    };
                return %5;
            };
            """)
    @Reflect
    static String casePatternWithCaseConstant(int i) {
        return switch (i) {
            case 0 -> "zero";
            case Integer j when j > 0 -> "positive";
            case Integer _ -> "negative";
        };
    }
}
