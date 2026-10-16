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
import java.util.Collection;
import java.util.RandomAccess;
import java.util.Stack;

/*
 * @test
 * @modules jdk.incubator.code
 * @build SwitchStatementTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchStatementTest
 */
public class SwitchStatementTest {

    @IR("""
            func @"caseConstantRuleExpression" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"s";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"FOO";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %12 : java.type:"java.lang.String" = constant @"BAR";
                        %13 : java.type:"boolean" = invoke %11 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"java.lang.String" = constant @"BAZ";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    }
                    (%17 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %18 : java.type:"java.lang.String" = constant @"BAZ";
                        %19 : java.type:"boolean" = invoke %17 %18 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"FOO";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = var.load %3;
                        %25 : java.type:"java.lang.String" = constant @"else";
                        %26 : java.type:"java.lang.String" = concat %24 %25;
                        var.store %3 %26;
                        yield;
                    };
                %27 : java.type:"java.lang.String" = var.load %3;
                return %27;
            };
            """)
    @Reflect
    public static String caseConstantRuleExpression(String r) {
        String s = "";
        switch (r) {
            case "FOO" -> s += "BAR";
            case "BAR" -> s += "BAZ";
            case "BAZ" -> s += "FOO";
            default -> s += "else";
        }
        return s;
    }

    @IR("""
            func @"caseConstantRuleBlock" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"r";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"s";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"FOO";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %12 : java.type:"java.lang.String" = constant @"BAR";
                        %13 : java.type:"boolean" = invoke %11 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"java.lang.String" = constant @"BAZ";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    }
                    (%17 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %18 : java.type:"java.lang.String" = constant @"BAZ";
                        %19 : java.type:"boolean" = invoke %17 %18 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"FOO";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = var.load %3;
                        %25 : java.type:"java.lang.String" = constant @"else";
                        %26 : java.type:"java.lang.String" = concat %24 %25;
                        var.store %3 %26;
                        yield;
                    };
                %27 : java.type:"java.lang.String" = var.load %3;
                return %27;
            };
            """)
    @Reflect
    public static String caseConstantRuleBlock(String r) {
        String s = "";
        switch (r) {
            case "FOO" -> {
                s += "BAR";
            }
            case "BAR" -> {
                s += "BAZ";
            }
            case "BAZ" -> {
                s += "FOO";
            }
            default -> {
                s += "else";
            }
        }
        return s;
    }

    @IR("""
            func @"caseConstantStatement" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"FOO";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"BAR";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        java.break;
                    }
                    (%11 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %12 : java.type:"java.lang.String" = constant @"BAR";
                        %13 : java.type:"boolean" = invoke %11 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"java.lang.String" = constant @"BAZ";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        java.break;
                    }
                    (%17 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %18 : java.type:"java.lang.String" = constant @"BAZ";
                        %19 : java.type:"boolean" = invoke %17 %18 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"FOO";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        java.break;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = var.load %3;
                        %25 : java.type:"java.lang.String" = constant @"else";
                        %26 : java.type:"java.lang.String" = concat %24 %25;
                        var.store %3 %26;
                        yield;
                    };
                %27 : java.type:"java.lang.String" = var.load %3;
                return %27;
            };
            """)
    @Reflect
    private static String caseConstantStatement(String s) {
        String r = "";
        switch (s) {
            case "FOO":
                r += "BAR";
                break;
            case "BAR":
                r += "BAZ";
                break;
            case "BAZ":
                r += "FOO";
                break;
            default:
                r += "else";
        }
        return r;
    }

    @IR("""
            func @"caseConstantMultiLabels" (%0 : java.type:"char")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"char"> = var %0 @"c";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"char" = var.load %1;
                %5 : java.type:"char" = invoke %4 @java.ref:"java.lang.Character::toLowerCase(char):char";
                java.switch.statement %5
                    (%6 : java.type:"char")java.type:"boolean" -> {
                        %7 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %8 : java.type:"char" = constant @'a';
                                %9 : java.type:"boolean" = eq %6 %8;
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %10 : java.type:"char" = constant @'e';
                                %11 : java.type:"boolean" = eq %6 %10;
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %12 : java.type:"char" = constant @'i';
                                %13 : java.type:"boolean" = eq %6 %12;
                                yield %13;
                            }
                            ()java.type:"boolean" -> {
                                %14 : java.type:"char" = constant @'o';
                                %15 : java.type:"boolean" = eq %6 %14;
                                yield %15;
                            }
                            ()java.type:"boolean" -> {
                                %16 : java.type:"char" = constant @'u';
                                %17 : java.type:"boolean" = eq %6 %16;
                                yield %17;
                            };
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"vowel";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        java.break;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"consonant";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    private static String caseConstantMultiLabels(char c) {
        String r = "";
        switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u':
                r += "vowel";
                break;
            default:
                r += "consonant";
        }
        return r;
    }

    @IR("""
            func @"caseConstantThrow" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Integer" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %6 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::intValue():int";
                        %7 : java.type:"int" = constant @8;
                        %8 : java.type:"boolean" = eq %6 %7;
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %9;
                    }
                    (%10 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %11 : java.type:"int" = invoke %10 @java.ref:"java.lang.Integer::intValue():int";
                        %12 : java.type:"int" = constant @9;
                        %13 : java.type:"boolean" = eq %11 %12;
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"java.lang.String" = constant @"Nine";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %17 : java.type:"boolean" = constant @true;
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"An integer";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    };
                %21 : java.type:"java.lang.String" = var.load %3;
                return %21;
            };
            """)
    @Reflect
    private static String caseConstantThrow(Integer i) {
        String r = "";
        switch (i) {
            case 8 -> throw new IllegalArgumentException();
            case 9 -> r += "Nine";
            default -> r += "An integer";
        }
        return r;
    }

    @IR("""
            func @"caseConstantNullLabel" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4 @switch.handle.nulls=true
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.Object" = constant @null;
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"null";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %3;
                        %13 : java.type:"java.lang.String" = constant @"non null";
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    };
                %15 : java.type:"java.lang.String" = var.load %3;
                return %15;
            };
            """)
    @Reflect
    private static String caseConstantNullLabel(String s) {
        String r = "";
        switch (s) {
            case null -> r += "null";
            default -> r += "non null";
        }
        return r;
    }

    @IR("""
            func @"caseConstantNullAndDefault" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4 @switch.handle.nulls=true
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"abc";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"alphabet";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %3;
                        %13 : java.type:"java.lang.String" = constant @"null or default";
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    };
                %15 : java.type:"java.lang.String" = var.load %3;
                return %15;
            };
            """)
    @Reflect
    private static String caseConstantNullAndDefault(String s) {
        String r = "";
        switch (s) {
            case "abc" -> r += "alphabet";
            case null, default -> r += "null or default";
        }
        return r;
    }

    @IR("""
            func @"caseConstantFallThrough" (%0 : java.type:"char")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"char"> = var %0 @"c";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"char" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"char")java.type:"boolean" -> {
                        %6 : java.type:"char" = constant @'A';
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        java.switch.fallthrough;
                    }
                    (%8 : java.type:"char")java.type:"boolean" -> {
                        %9 : java.type:"char" = constant @'B';
                        %10 : java.type:"boolean" = eq %8 %9;
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"A or B";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        java.break;
                    }
                    ()java.type:"boolean" -> {
                        %14 : java.type:"boolean" = constant @true;
                        yield %14;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %3;
                        %16 : java.type:"java.lang.String" = constant @"Neither A nor B";
                        %17 : java.type:"java.lang.String" = concat %15 %16;
                        var.store %3 %17;
                        yield;
                    };
                %18 : java.type:"java.lang.String" = var.load %3;
                return %18;
            };
            """)
    @Reflect
    private static String caseConstantFallThrough(char c) {
        String r = "";
        switch (c) {
            case 'A':
            case 'B':
                r += "A or B";
                break;
            default:
                r += "Neither A nor B";
        }
        return r;
    }

    enum Day {
        MON, TUE, WED, THU, FRI, SAT, SUN
    }
    @IR("""
            func @"caseConstantEnum" (%0 : java.type:"SwitchStatementTest$Day")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"SwitchStatementTest$Day"> = var %0 @"d";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"SwitchStatementTest$Day" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"SwitchStatementTest$Day")java.type:"boolean" -> {
                        %6 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %7 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::MON:SwitchStatementTest$Day";
                                %8 : java.type:"boolean" = eq %5 %7;
                                yield %8;
                            }
                            ()java.type:"boolean" -> {
                                %9 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::FRI:SwitchStatementTest$Day";
                                %10 : java.type:"boolean" = eq %5 %9;
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %11 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::SUN:SwitchStatementTest$Day";
                                %12 : java.type:"boolean" = eq %5 %11;
                                yield %12;
                            };
                        yield %6;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"int" = constant @6;
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : java.type:"SwitchStatementTest$Day")java.type:"boolean" -> {
                        %17 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::TUE:SwitchStatementTest$Day";
                        %18 : java.type:"boolean" = eq %16 %17;
                        yield %18;
                    }
                    ()java.type:"void" -> {
                        %19 : java.type:"java.lang.String" = var.load %3;
                        %20 : java.type:"int" = constant @7;
                        %21 : java.type:"java.lang.String" = concat %19 %20;
                        var.store %3 %21;
                        yield;
                    }
                    (%22 : java.type:"SwitchStatementTest$Day")java.type:"boolean" -> {
                        %23 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %24 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::THU:SwitchStatementTest$Day";
                                %25 : java.type:"boolean" = eq %22 %24;
                                yield %25;
                            }
                            ()java.type:"boolean" -> {
                                %26 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::SAT:SwitchStatementTest$Day";
                                %27 : java.type:"boolean" = eq %22 %26;
                                yield %27;
                            };
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %28 : java.type:"java.lang.String" = var.load %3;
                        %29 : java.type:"int" = constant @8;
                        %30 : java.type:"java.lang.String" = concat %28 %29;
                        var.store %3 %30;
                        yield;
                    }
                    (%31 : java.type:"SwitchStatementTest$Day")java.type:"boolean" -> {
                        %32 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::WED:SwitchStatementTest$Day";
                        %33 : java.type:"boolean" = eq %31 %32;
                        yield %33;
                    }
                    ()java.type:"void" -> {
                        %34 : java.type:"java.lang.String" = var.load %3;
                        %35 : java.type:"int" = constant @9;
                        %36 : java.type:"java.lang.String" = concat %34 %35;
                        var.store %3 %36;
                        yield;
                    };
                %37 : java.type:"java.lang.String" = var.load %3;
                return %37;
            };
            """)
    @Reflect
    private static String caseConstantEnum(Day d) {
        String r = "";
        switch (d) {
            case MON, FRI, SUN -> r += 6;
            case TUE -> r += 7;
            case THU, SAT -> r += 8;
            case WED -> r += 9;
        }
        return r;
    }

    static class Constants {
        static final int c1 = 12;
    }
    @IR("""
            func @"caseConstantOtherKindsOfExpr" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"int" = constant @11;
                %5 : java.type:"int" = var.load %1;
                java.switch.statement %5
                    (%6 : java.type:"int")java.type:"boolean" -> {
                        %7 : java.type:"int" = constant @1;
                        %8 : java.type:"boolean" = eq %6 %7;
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = var.load %3;
                        %10 : java.type:"int" = constant @1;
                        %11 : java.type:"java.lang.String" = concat %9 %10;
                        var.store %3 %11;
                        yield;
                    }
                    (%12 : java.type:"int")java.type:"boolean" -> {
                        %13 : java.type:"int" = constant @2;
                        %14 : java.type:"boolean" = eq %12 %13;
                        yield %14;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %3;
                        %16 : java.type:"java.lang.String" = constant @"2";
                        %17 : java.type:"java.lang.String" = concat %15 %16;
                        var.store %3 %17;
                        yield;
                    }
                    (%18 : java.type:"int")java.type:"boolean" -> {
                        %19 : java.type:"int" = constant @3;
                        %20 : java.type:"boolean" = eq %18 %19;
                        yield %20;
                    }
                    ()java.type:"void" -> {
                        %21 : java.type:"java.lang.String" = var.load %3;
                        %22 : java.type:"int" = constant @3;
                        %23 : java.type:"java.lang.String" = concat %21 %22;
                        var.store %3 %23;
                        yield;
                    }
                    (%24 : java.type:"int")java.type:"boolean" -> {
                        %25 : java.type:"int" = constant @4;
                        %26 : java.type:"boolean" = eq %24 %25;
                        yield %26;
                    }
                    ()java.type:"void" -> {
                        %27 : java.type:"java.lang.String" = var.load %3;
                        %28 : java.type:"int" = constant @4;
                        %29 : java.type:"java.lang.String" = concat %27 %28;
                        var.store %3 %29;
                        yield;
                    }
                    (%30 : java.type:"int")java.type:"boolean" -> {
                        %31 : java.type:"int" = constant @5;
                        %32 : java.type:"boolean" = eq %30 %31;
                        yield %32;
                    }
                    ()java.type:"void" -> {
                        %33 : java.type:"java.lang.String" = var.load %3;
                        %34 : java.type:"int" = constant @5;
                        %35 : java.type:"java.lang.String" = concat %33 %34;
                        var.store %3 %35;
                        yield;
                    }
                    (%36 : java.type:"int")java.type:"boolean" -> {
                        %37 : java.type:"int" = constant @6;
                        %38 : java.type:"boolean" = eq %36 %37;
                        yield %38;
                    }
                    ()java.type:"void" -> {
                        %39 : java.type:"java.lang.String" = var.load %3;
                        %40 : java.type:"int" = constant @6;
                        %41 : java.type:"java.lang.String" = concat %39 %40;
                        var.store %3 %41;
                        yield;
                    }
                    (%42 : java.type:"int")java.type:"boolean" -> {
                        %43 : java.type:"int" = constant @7;
                        %44 : java.type:"boolean" = eq %42 %43;
                        yield %44;
                    }
                    ()java.type:"void" -> {
                        %45 : java.type:"java.lang.String" = var.load %3;
                        %46 : java.type:"int" = constant @7;
                        %47 : java.type:"java.lang.String" = concat %45 %46;
                        var.store %3 %47;
                        yield;
                    }
                    (%48 : java.type:"int")java.type:"boolean" -> {
                        %49 : java.type:"int" = constant @8;
                        %50 : java.type:"boolean" = eq %48 %49;
                        yield %50;
                    }
                    ()java.type:"void" -> {
                        %51 : java.type:"java.lang.String" = var.load %3;
                        %52 : java.type:"int" = constant @8;
                        %53 : java.type:"java.lang.String" = concat %51 %52;
                        var.store %3 %53;
                        yield;
                    }
                    (%54 : java.type:"int")java.type:"boolean" -> {
                        %55 : java.type:"int" = constant @9;
                        %56 : java.type:"boolean" = eq %54 %55;
                        yield %56;
                    }
                    ()java.type:"void" -> {
                        %57 : java.type:"java.lang.String" = var.load %3;
                        %58 : java.type:"int" = constant @9;
                        %59 : java.type:"java.lang.String" = concat %57 %58;
                        var.store %3 %59;
                        yield;
                    }
                    (%60 : java.type:"int")java.type:"boolean" -> {
                        %61 : java.type:"int" = constant @10;
                        %62 : java.type:"boolean" = eq %60 %61;
                        yield %62;
                    }
                    ()java.type:"void" -> {
                        %63 : java.type:"java.lang.String" = var.load %3;
                        %64 : java.type:"int" = constant @10;
                        %65 : java.type:"java.lang.String" = concat %63 %64;
                        var.store %3 %65;
                        yield;
                    }
                    (%66 : java.type:"int")java.type:"boolean" -> {
                        %67 : java.type:"int" = constant @11;
                        %68 : java.type:"boolean" = eq %66 %67;
                        yield %68;
                    }
                    ()java.type:"void" -> {
                        %69 : java.type:"java.lang.String" = var.load %3;
                        %70 : java.type:"int" = constant @11;
                        %71 : java.type:"java.lang.String" = concat %69 %70;
                        var.store %3 %71;
                        yield;
                    }
                    (%72 : java.type:"int")java.type:"boolean" -> {
                        %73 : java.type:"int" = constant @12;
                        %74 : java.type:"boolean" = eq %72 %73;
                        yield %74;
                    }
                    ()java.type:"void" -> {
                        %75 : java.type:"java.lang.String" = var.load %3;
                        %76 : java.type:"int" = constant @12;
                        %77 : java.type:"java.lang.String" = concat %75 %76;
                        var.store %3 %77;
                        yield;
                    }
                    (%78 : java.type:"int")java.type:"boolean" -> {
                        %79 : java.type:"int" = constant @13;
                        %80 : java.type:"boolean" = eq %78 %79;
                        yield %80;
                    }
                    ()java.type:"void" -> {
                        %81 : java.type:"java.lang.String" = var.load %3;
                        %82 : java.type:"int" = constant @13;
                        %83 : java.type:"java.lang.String" = concat %81 %82;
                        var.store %3 %83;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %84 : java.type:"boolean" = constant @true;
                        yield %84;
                    }
                    ()java.type:"void" -> {
                        %85 : java.type:"java.lang.String" = var.load %3;
                        %86 : java.type:"java.lang.String" = constant @"an int";
                        %87 : java.type:"java.lang.String" = concat %85 %86;
                        var.store %3 %87;
                        yield;
                    };
                %88 : java.type:"java.lang.String" = var.load %3;
                return %88;
            };
            """)
    @Reflect
    private static String caseConstantOtherKindsOfExpr(int i) {
        String r = "";
        final int eleven = 11;
        switch (i) {
            case 1 & 0xF -> r += 1;
            case 4>>1 -> r += "2";
            case (int) 3L -> r += 3;
            case 2<<1 -> r += 4;
            case 10 / 2 -> r += 5;
            case 12 - 6 -> r += 6;
            case 3 + 4 -> r += 7;
            case 2 * 2 * 2 -> r += 8;
            case 8 | 1 -> r += 9;
            case (10) -> r += 10;
            case eleven -> r += 11;
            case Constants.c1 -> r += Constants.c1;
            case 1 > 0 ? 13 : 133 -> r += 13;
            default -> r += "an int";
        }
        return r;
    }

    @IR("""
            func @"caseConstantConv" (%0 : java.type:"short")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"short"> = var %0 @"a";
                %2 : java.type:"int" = constant @1;
                %3 : java.type:"short" = conv %2;
                %4 : java.type:"int" = constant @2;
                %5 : java.type:"byte" = conv %4;
                %6 : java.type:"java.lang.String" = constant @"";
                %7 : Var<java.type:"java.lang.String"> = var %6 @"r";
                %8 : java.type:"short" = var.load %1;
                java.switch.statement %8
                    (%9 : java.type:"short")java.type:"boolean" -> {
                        %10 : java.type:"short" = constant @1;
                        %11 : java.type:"boolean" = eq %9 %10;
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %7;
                        %13 : java.type:"java.lang.String" = constant @"one";
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %7 %14;
                        yield;
                    }
                    (%15 : java.type:"short")java.type:"boolean" -> {
                        %16 : java.type:"short" = constant @2;
                        %17 : java.type:"boolean" = eq %15 %16;
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %7;
                        %19 : java.type:"java.lang.String" = constant @"two";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %7 %20;
                        yield;
                    }
                    (%21 : java.type:"short")java.type:"boolean" -> {
                        %22 : java.type:"short" = constant @3;
                        %23 : java.type:"boolean" = eq %21 %22;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = var.load %7;
                        %25 : java.type:"java.lang.String" = constant @"three";
                        %26 : java.type:"java.lang.String" = concat %24 %25;
                        var.store %7 %26;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %27 : java.type:"boolean" = constant @true;
                        yield %27;
                    }
                    ()java.type:"void" -> {
                        %28 : java.type:"java.lang.String" = var.load %7;
                        %29 : java.type:"java.lang.String" = constant @"else";
                        %30 : java.type:"java.lang.String" = concat %28 %29;
                        var.store %7 %30;
                        yield;
                    };
                %31 : java.type:"java.lang.String" = var.load %7;
                return %31;
            };
            """)
    @Reflect
    static String caseConstantConv(short a) {
        final short s = 1;
        final byte b = 2;
        String r = "";
        switch (a) {
            case s -> r += "one"; // identity, short -> short
            case b -> r += "two"; // widening primitive conversion, byte -> short
            case 3 -> r += "three"; // narrowing primitive conversion, int -> short
            default -> r += "else";
        }
        return r;
    }

    @IR("""
            func @"caseConstantConv2" (%0 : java.type:"java.lang.Byte")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Byte"> = var %0 @"a";
                %2 : java.type:"int" = constant @2;
                %3 : java.type:"byte" = conv %2;
                %4 : java.type:"java.lang.String" = constant @"";
                %5 : Var<java.type:"java.lang.String"> = var %4 @"r";
                %6 : java.type:"java.lang.Byte" = var.load %1;
                java.switch.statement %6
                    (%7 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %8 : java.type:"byte" = invoke %7 @java.ref:"java.lang.Byte::byteValue():byte";
                        %9 : java.type:"byte" = constant @1;
                        %10 : java.type:"boolean" = eq %8 %9;
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %5;
                        %12 : java.type:"java.lang.String" = constant @"one";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %5 %13;
                        yield;
                    }
                    (%14 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %15 : java.type:"byte" = invoke %14 @java.ref:"java.lang.Byte::byteValue():byte";
                        %16 : java.type:"byte" = constant @2;
                        %17 : java.type:"boolean" = eq %15 %16;
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %5;
                        %19 : java.type:"java.lang.String" = constant @"two";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %5 %20;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %5;
                        %23 : java.type:"java.lang.String" = constant @"default";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %5 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %5;
                return %25;
            };
            """)
    @Reflect
    static String caseConstantConv2(Byte a) {
        final byte b = 2;
        String r = "";
        switch (a) {
            case 1 -> r+= "one";
            case b -> r+= "two";
            default -> r+= "default";
        }
        return r;
    }

    @IR("""
            func @"nonEnhancedSwStatNoDefault" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"a";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"int" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"1";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : java.type:"int")java.type:"boolean" -> {
                        %12 : java.type:"int" = constant @2;
                        %13 : java.type:"boolean" = eq %11 %12;
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"int" = constant @2;
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    };
                %17 : java.type:"java.lang.String" = var.load %3;
                return %17;
            };
            """)
    @Reflect
    static String nonEnhancedSwStatNoDefault(int a) {
        String r = "";
        switch (a) {
            case 1 -> r += "1";
            case 2 -> r += 2;
        }
        return r;
    }

    enum E {A, B}
    @IR("""
            func @"enhancedSwStatNoDefault1" (%0 : java.type:"SwitchStatementTest$E")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"SwitchStatementTest$E"> = var %0 @"e";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"SwitchStatementTest$E" = var.load %1;
                java.switch.statement %4 @switch.handle.nulls=true
                    (%5 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %6 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::A:SwitchStatementTest$E";
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::A:SwitchStatementTest$E";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %12 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::B:SwitchStatementTest$E";
                        %13 : java.type:"boolean" = eq %11 %12;
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::B:SwitchStatementTest$E";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    }
                    (%17 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %18 : java.type:"java.lang.Object" = constant @null;
                        %19 : java.type:"boolean" = eq %17 %18;
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"null";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = constant @null;
                        %25 : java.type:"java.lang.Throwable" = constant @null;
                        %26 : java.type:"java.lang.MatchException" = new %24 %25 @java.ref:"java.lang.MatchException::(java.lang.String, java.lang.Throwable)";
                        throw %26;
                    };
            %27 : java.type:"java.lang.String" = var.load %3;
            return %27;
            };
            """)
    @Reflect
    static String enhancedSwStatNoDefault1(E e) {
        String r = "";
        switch (e) {
            case A -> r += E.A;
            case B -> r += E.B;
            case null -> r += "null";
        }
        return r;
    }

    sealed interface I permits K, J {}
    record K() implements I {}
    static final class J implements I {}
    @IR("""
            func @"enhancedSwStatNoDefault2" (%0 : java.type:"SwitchStatementTest$I")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"SwitchStatementTest$I"> = var %0 @"i";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"SwitchStatementTest$I" = var.load %1;
                %5 : Var<java.type:"SwitchStatementTest$K"> = var @"k";
                %6 : Var<java.type:"SwitchStatementTest$J"> = var @"j";
                java.switch.statement %4
                    (%7 : java.type:"SwitchStatementTest$I")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$K>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$K>" = pattern.type @"k";
                                yield %9;
                            }
                            (%10 : java.type:"SwitchStatementTest$K")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"K";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        yield;
                    }
                    (%14 : java.type:"SwitchStatementTest$I")java.type:"boolean" -> {
                        %15 : java.type:"boolean" = pattern.match %14
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$J>" -> {
                                %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$J>" = pattern.type @"j";
                                yield %16;
                            }
                            (%17 : java.type:"SwitchStatementTest$J")java.type:"void" -> {
                                var.store %6 %17;
                                yield;
                            };
                        yield %15;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"J";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = constant @null;
                        %23 : java.type:"java.lang.Throwable" = constant @null;
                        %24 : java.type:"java.lang.MatchException" = new %22 %23 @java.ref:"java.lang.MatchException::(java.lang.String, java.lang.Throwable)";
                        throw %24;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    static String enhancedSwStatNoDefault2(I i) {
        String r = "";
        switch (i) {
            case K k -> r += "K";
            case J j -> r += "J";
        }
        return r;
    }

    @IR("""
            func @"enhancedSwStatUnconditionalPattern" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.String" = var.load %1;
                %5 : Var<java.type:"java.lang.Object"> = var @"o";
                java.switch.statement %4
                    (%6 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %7 : java.type:"java.lang.String" = constant @"A";
                        %8 : java.type:"boolean" = invoke %6 %7 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = var.load %3;
                        %10 : java.type:"java.lang.String" = constant @"A";
                        %11 : java.type:"java.lang.String" = concat %9 %10;
                        var.store %3 %11;
                        yield;
                    }
                    (%12 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %13 : java.type:"boolean" = pattern.match %12
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type @"o";
                                yield %14;
                            }
                            (%15 : java.type:"java.lang.Object")java.type:"void" -> {
                                var.store %5 %15;
                                yield;
                            };
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %16 : java.type:"java.lang.String" = var.load %3;
                        %17 : java.type:"java.lang.String" = constant @"obj";
                        %18 : java.type:"java.lang.String" = concat %16 %17;
                        var.store %3 %18;
                        yield;
                    };
                %19 : java.type:"java.lang.String" = var.load %3;
                return %19;
            };
            """)
    @Reflect
    static String enhancedSwStatUnconditionalPattern(String s) {
        String r = "";
        switch (s) {
            case "A" -> r += "A";
            case Object o -> r += "obj";
        }
        return r;
    }

    @IR("""
            func @"casePatternRuleExpression" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                %6 : Var<java.type:"java.lang.String"> = var @"s";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"integer";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        yield;
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
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"string";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"else";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    private static String casePatternRuleExpression(Object o) {
        String r = "";
        switch (o) {
            case Integer i -> r += "integer";
            case String s -> r+= "string";
            default -> r+= "else";
        }
        return r;
    }

    @IR("""
            func @"casePatternRuleBlock" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                %6 : Var<java.type:"java.lang.String"> = var @"s";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"integer";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        yield;
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
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"string";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"else";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    private static String casePatternRuleBlock(Object o) {
        String r = "";
        switch (o) {
            case Integer i -> {
                r += "integer";
            }
            case String s -> {
                r += "string";
            }
            default -> {
                r += "else";
            }
        }
        return r;
    }

    @IR("""
            func @"casePatternStatement" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                %6 : Var<java.type:"java.lang.String"> = var @"s";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"integer";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        java.break;
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
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"string";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        java.break;
                    }
                    ()java.type:"boolean" -> {
                        %21 : java.type:"boolean" = constant @true;
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"else";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    private static String casePatternStatement(Object o) {
        String r = "";
        switch (o) {
            case Integer i:
                r += "integer";
                break;
            case String s:
                r += "string";
                break;
            default:
                r += "else";
        }
        return r;
    }

    @IR("""
            func @"casePatternThrow" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Number"> = var @"n";
                %6 : Var<java.type:"java.lang.String"> = var @"s";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %11;
                    }
                    (%12 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %13 : java.type:"boolean" = pattern.match %12
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %14 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %14;
                            }
                            (%15 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %6 %15;
                                yield;
                            };
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %16 : java.type:"java.lang.String" = var.load %3;
                        %17 : java.type:"java.lang.String" = constant @"a string";
                        %18 : java.type:"java.lang.String" = concat %16 %17;
                        var.store %3 %18;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %19 : java.type:"boolean" = constant @true;
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.Object" = var.load %1;
                        %22 : java.type:"java.lang.Class<?>" = invoke %21 @java.ref:"java.lang.Object::getClass():java.lang.Class";
                        %23 : java.type:"java.lang.String" = invoke %22 @java.ref:"java.lang.Class::getName():java.lang.String";
                        %24 : java.type:"java.lang.String" = concat %20 %23;
                        var.store %3 %24;
                        yield;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @Reflect
    private static String casePatternThrow(Object o) {
        String r = "";
        switch (o) {
            case Number n -> throw new IllegalArgumentException();
            case String s -> r += "a string";
            default -> r += o.getClass().getName();
        }
        return r;
    }

    @IR("""
            func @"casePatternMultiLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var;
                %6 : Var<java.type:"java.lang.Long"> = var;
                %7 : Var<java.type:"java.lang.Character"> = var;
                %8 : Var<java.type:"java.lang.Byte"> = var;
                %9 : Var<java.type:"java.lang.Short"> = var;
                java.switch.statement %4
                    (%10 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %11 : java.type:"boolean" = java.cor
                            ()java.type:"boolean" -> {
                                %12 : java.type:"boolean" = pattern.match %10
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type;
                                        yield %13;
                                    }
                                    (%14 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %5 %14;
                                        yield;
                                    };
                                yield %12;
                            }
                            ()java.type:"boolean" -> {
                                %15 : java.type:"boolean" = pattern.match %10
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" -> {
                                        %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" = pattern.type;
                                        yield %16;
                                    }
                                    (%17 : java.type:"java.lang.Long")java.type:"void" -> {
                                        var.store %6 %17;
                                        yield;
                                    };
                                yield %15;
                            }
                            ()java.type:"boolean" -> {
                                %18 : java.type:"boolean" = pattern.match %10
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Character>" -> {
                                        %19 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Character>" = pattern.type;
                                        yield %19;
                                    }
                                    (%20 : java.type:"java.lang.Character")java.type:"void" -> {
                                        var.store %7 %20;
                                        yield;
                                    };
                                yield %18;
                            }
                            ()java.type:"boolean" -> {
                                %21 : java.type:"boolean" = pattern.match %10
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" -> {
                                        %22 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" = pattern.type;
                                        yield %22;
                                    }
                                    (%23 : java.type:"java.lang.Byte")java.type:"void" -> {
                                        var.store %8 %23;
                                        yield;
                                    };
                                yield %21;
                            }
                            ()java.type:"boolean" -> {
                                %24 : java.type:"boolean" = pattern.match %10
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" -> {
                                        %25 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" = pattern.type;
                                        yield %25;
                                    }
                                    (%26 : java.type:"java.lang.Short")java.type:"void" -> {
                                        var.store %9 %26;
                                        yield;
                                    };
                                yield %24;
                            };
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %27 : java.type:"java.lang.String" = var.load %3;
                        %28 : java.type:"java.lang.String" = constant @"integral type";
                        %29 : java.type:"java.lang.String" = concat %27 %28;
                        var.store %3 %29;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %30 : java.type:"boolean" = constant @true;
                        yield %30;
                    }
                    ()java.type:"void" -> {
                        %31 : java.type:"java.lang.String" = var.load %3;
                        %32 : java.type:"java.lang.String" = constant @"non integral type";
                        %33 : java.type:"java.lang.String" = concat %31 %32;
                        var.store %3 %33;
                        yield;
                    };
                %34 : java.type:"java.lang.String" = var.load %3;
                return %34;
            };
            """)
    @Reflect
    private static String casePatternMultiLabel(Object o) {
        String r = "";
        switch (o) {
            case Integer _, Long _, Character _, Byte _, Short _-> r += "integral type";
            default -> r += "non integral type";
        }
        return r;
    }

    @IR("""
            func @"casePatternGuardedMultiLabel" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var;
                %6 : Var<java.type:"java.lang.Long"> = var;
                %7 : Var<java.type:"java.lang.Byte"> = var;
                %8 : Var<java.type:"java.lang.Short"> = var;
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %11 : java.type:"boolean" = java.cor
                                    ()java.type:"boolean" -> {
                                        %12 : java.type:"boolean" = pattern.match %9
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                                %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type;
                                                yield %13;
                                            }
                                            (%14 : java.type:"java.lang.Integer")java.type:"void" -> {
                                                var.store %5 %14;
                                                yield;
                                            };
                                        yield %12;
                                    }
                                    ()java.type:"boolean" -> {
                                        %15 : java.type:"boolean" = pattern.match %9
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" -> {
                                                %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Long>" = pattern.type;
                                                yield %16;
                                            }
                                            (%17 : java.type:"java.lang.Long")java.type:"void" -> {
                                                var.store %6 %17;
                                                yield;
                                            };
                                        yield %15;
                                    }
                                    ()java.type:"boolean" -> {
                                        %18 : java.type:"boolean" = pattern.match %9
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" -> {
                                                %19 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Byte>" = pattern.type;
                                                yield %19;
                                            }
                                            (%20 : java.type:"java.lang.Byte")java.type:"void" -> {
                                                var.store %7 %20;
                                                yield;
                                            };
                                        yield %18;
                                    }
                                    ()java.type:"boolean" -> {
                                        %21 : java.type:"boolean" = pattern.match %9
                                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" -> {
                                                %22 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Short>" = pattern.type;
                                                yield %22;
                                            }
                                            (%23 : java.type:"java.lang.Short")java.type:"void" -> {
                                                var.store %8 %23;
                                                yield;
                                            };
                                        yield %21;
                                    };
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %24 : java.type:"java.lang.Object" = var.load %1;
                                %25 : java.type:"java.lang.Number" = cast %24 @java.type:"java.lang.Number";
                                %26 : java.type:"int" = invoke %25 @java.ref:"java.lang.Number::intValue():int";
                                %27 : java.type:"int" = constant @0;
                                %28 : java.type:"boolean" = gt %26 %27;
                                yield %28;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %29 : java.type:"java.lang.String" = var.load %3;
                        %30 : java.type:"java.lang.String" = constant @"integral type";
                        %31 : java.type:"java.lang.String" = concat %29 %30;
                        var.store %3 %31;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %32 : java.type:"boolean" = constant @true;
                        yield %32;
                    }
                    ()java.type:"void" -> {
                        %33 : java.type:"java.lang.String" = var.load %3;
                        %34 : java.type:"java.lang.String" = constant @"non integral type";
                        %35 : java.type:"java.lang.String" = concat %33 %34;
                        var.store %3 %35;
                        yield;
                    };
                %36 : java.type:"java.lang.String" = var.load %3;
                return %36;
            };
            """)
    @Reflect
    private static String casePatternGuardedMultiLabel(Object o) {
        String r = "";
        switch (o) {
            case Integer _, Long _, Byte _, Short _ when ((Number)o).intValue() > 0 -> r += "integral type";
            default -> r += "non integral type";
        }
        return r;
    }

    @IR("""
            func @"casePatternWithCaseConstant" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"a";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Integer" = var.load %1;
                %5 : Var<java.type:"java.lang.Integer"> = var @"i";
                %6 : Var<java.type:"java.lang.Integer"> = var @"i";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %8 : java.type:"int" = invoke %7 @java.ref:"java.lang.Integer::intValue():int";
                        %9 : java.type:"int" = constant @42;
                        %10 : java.type:"boolean" = eq %8 %9;
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"forty two";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        yield;
                    }
                    (%14 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %15 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %16 : java.type:"boolean" = pattern.match %14
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %17 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                        yield %17;
                                    }
                                    (%18 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %5 %18;
                                        yield;
                                    };
                                yield %16;
                            }
                            ()java.type:"boolean" -> {
                                %19 : java.type:"java.lang.Integer" = var.load %5;
                                %20 : java.type:"int" = invoke %19 @java.ref:"java.lang.Integer::intValue():int";
                                %21 : java.type:"int" = constant @0;
                                %22 : java.type:"boolean" = gt %20 %21;
                                yield %22;
                            };
                        yield %15;
                    }
                    ()java.type:"void" -> {
                        %23 : java.type:"java.lang.String" = var.load %3;
                        %24 : java.type:"java.lang.String" = constant @"positive int";
                        %25 : java.type:"java.lang.String" = concat %23 %24;
                        var.store %3 %25;
                        yield;
                    }
                    (%26 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %27 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %28 : java.type:"boolean" = pattern.match %26
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %29 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                        yield %29;
                                    }
                                    (%30 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %6 %30;
                                        yield;
                                    };
                                yield %28;
                            }
                            ()java.type:"boolean" -> {
                                %31 : java.type:"java.lang.Integer" = var.load %6;
                                %32 : java.type:"int" = invoke %31 @java.ref:"java.lang.Integer::intValue():int";
                                %33 : java.type:"int" = constant @0;
                                %34 : java.type:"boolean" = lt %32 %33;
                                yield %34;
                            };
                        yield %27;
                    }
                    ()java.type:"void" -> {
                        %35 : java.type:"java.lang.String" = var.load %3;
                        %36 : java.type:"java.lang.String" = constant @"negative int";
                        %37 : java.type:"java.lang.String" = concat %35 %36;
                        var.store %3 %37;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %38 : java.type:"boolean" = constant @true;
                        yield %38;
                    }
                    ()java.type:"void" -> {
                        %39 : java.type:"java.lang.String" = var.load %3;
                        %40 : java.type:"java.lang.String" = constant @"zero";
                        %41 : java.type:"java.lang.String" = concat %39 %40;
                        var.store %3 %41;
                        yield;
                    };
                %42 : java.type:"java.lang.String" = var.load %3;
                return %42;
            };
            """)
    @Reflect
    static String casePatternWithCaseConstant(Integer a) {
        String r = "";
        switch (a) {
            case 42 -> r += "forty two";
            case Integer i when i > 0 -> r += "positive int";
            case Integer i when i < 0 -> r += "negative int";
            default -> r += "zero";
        }
        return r;
    }

    @IR("""
            func @"caseTypePattern" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.String"> = var;
                %6 : Var<java.type:"java.util.RandomAccess"> = var;
                %7 : Var<java.type:"int[]"> = var;
                %8 : Var<java.type:"java.util.Stack[][]"> = var;
                %9 : Var<java.type:"java.util.Collection[][][]"> = var;
                %10 : Var<java.type:"java.lang.Number"> = var @"n";
                java.switch.statement %4
                    (%11 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %12 : java.type:"boolean" = pattern.match %11
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %13 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type;
                                yield %13;
                            }
                            (%14 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %5 %14;
                                yield;
                            };
                        yield %12;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %3;
                        %16 : java.type:"java.lang.String" = constant @"String";
                        %17 : java.type:"java.lang.String" = concat %15 %16;
                        var.store %3 %17;
                        yield;
                    }
                    (%18 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %19 : java.type:"boolean" = pattern.match %18
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.RandomAccess>" -> {
                                %20 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.RandomAccess>" = pattern.type;
                                yield %20;
                            }
                            (%21 : java.type:"java.util.RandomAccess")java.type:"void" -> {
                                var.store %6 %21;
                                yield;
                            };
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"RandomAccess";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    }
                    (%25 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %26 : java.type:"boolean" = pattern.match %25
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int[]>" -> {
                                %27 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int[]>" = pattern.type;
                                yield %27;
                            }
                            (%28 : java.type:"int[]")java.type:"void" -> {
                                var.store %7 %28;
                                yield;
                            };
                        yield %26;
                    }
                    ()java.type:"void" -> {
                        %29 : java.type:"java.lang.String" = var.load %3;
                        %30 : java.type:"java.lang.String" = constant @"int[]";
                        %31 : java.type:"java.lang.String" = concat %29 %30;
                        var.store %3 %31;
                        yield;
                    }
                    (%32 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %33 : java.type:"boolean" = pattern.match %32
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Stack[][]>" -> {
                                %34 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Stack[][]>" = pattern.type;
                                yield %34;
                            }
                            (%35 : java.type:"java.util.Stack[][]")java.type:"void" -> {
                                var.store %8 %35;
                                yield;
                            };
                        yield %33;
                    }
                    ()java.type:"void" -> {
                        %36 : java.type:"java.lang.String" = var.load %3;
                        %37 : java.type:"java.lang.String" = constant @"Stack[][]";
                        %38 : java.type:"java.lang.String" = concat %36 %37;
                        var.store %3 %38;
                        yield;
                    }
                    (%39 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %40 : java.type:"boolean" = pattern.match %39
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Collection[][][]>" -> {
                                %41 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Collection[][][]>" = pattern.type;
                                yield %41;
                            }
                            (%42 : java.type:"java.util.Collection[][][]")java.type:"void" -> {
                                var.store %9 %42;
                                yield;
                            };
                        yield %40;
                    }
                    ()java.type:"void" -> {
                        %43 : java.type:"java.lang.String" = var.load %3;
                        %44 : java.type:"java.lang.String" = constant @"Collection[][][]";
                        %45 : java.type:"java.lang.String" = concat %43 %44;
                        var.store %3 %45;
                        yield;
                    }
                    (%46 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %47 : java.type:"boolean" = pattern.match %46
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                %48 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                yield %48;
                            }
                            (%49 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %10 %49;
                                yield;
                            };
                        yield %47;
                    }
                    ()java.type:"void" -> {
                        %50 : java.type:"java.lang.String" = var.load %3;
                        %51 : java.type:"java.lang.String" = constant @"Number";
                        %52 : java.type:"java.lang.String" = concat %50 %51;
                        var.store %3 %52;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %53 : java.type:"boolean" = constant @true;
                        yield %53;
                    }
                    ()java.type:"void" -> {
                        %54 : java.type:"java.lang.String" = var.load %3;
                        %55 : java.type:"java.lang.String" = constant @"something else";
                        %56 : java.type:"java.lang.String" = concat %54 %55;
                        var.store %3 %56;
                        yield;
                    };
                %57 : java.type:"java.lang.String" = var.load %3;
                return %57;
            };
            """)
    @Reflect
    static String caseTypePattern(Object o) {
        String r = "";
        switch (o) {
            case String _ -> r+= "String"; // class
            case RandomAccess _ -> r+= "RandomAccess"; // interface
            case int[] _ -> r+= "int[]"; // array primitive
            case Stack[][] _ -> r+= "Stack[][]"; // array class
            case Collection[][][] _ -> r+= "Collection[][][]"; // array interface
            case final Number n -> r+= "Number"; // final modifier
            default -> r+= "something else";
        }
        return r;
    }

    record R(Number n) {}
    @IR("""
            func @"caseRecordPattern" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.Number"> = var @"n";
                java.switch.statement %4
                    (%6 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %7 : java.type:"boolean" = pattern.match %6
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" -> {
                                %8 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" = pattern.record %8 @java.ref:"(java.lang.Number n)SwitchStatementTest$R";
                                yield %9;
                            }
                            (%10 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %5 %10;
                                yield;
                            };
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %11 : java.type:"java.lang.String" = var.load %3;
                        %12 : java.type:"java.lang.String" = constant @"R(_)";
                        %13 : java.type:"java.lang.String" = concat %11 %12;
                        var.store %3 %13;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %14 : java.type:"boolean" = constant @true;
                        yield %14;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %3;
                        %16 : java.type:"java.lang.String" = constant @"else";
                        %17 : java.type:"java.lang.String" = concat %15 %16;
                        var.store %3 %17;
                        yield;
                    };
                %18 : java.type:"java.lang.String" = var.load %3;
                return %18;
            };
            """)
    @Reflect
    static String caseRecordPattern(Object o) {
        String r = "";
        switch (o) {
            case R(Number n) -> r += "R(_)";
            default -> r+= "else";
        }
        return r;
    }

    @IR("""
            func @"casePatternGuard" (%0 : java.type:"java.lang.Object")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"obj";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : Var<java.type:"java.lang.String"> = var @"s";
                %6 : Var<java.type:"java.lang.Number"> = var @"n";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %9 : java.type:"boolean" = pattern.match %7
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %10;
                                    }
                                    (%11 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %5 %11;
                                        yield;
                                    };
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %12 : java.type:"java.lang.String" = var.load %5;
                                %13 : java.type:"int" = invoke %12 @java.ref:"java.lang.String::length():int";
                                %14 : java.type:"int" = constant @3;
                                %15 : java.type:"boolean" = gt %13 %14;
                                yield %15;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %16 : java.type:"java.lang.String" = var.load %3;
                        %17 : java.type:"java.lang.String" = constant @"str with length > %d";
                        %18 : java.type:"java.lang.String" = var.load %5;
                        %19 : java.type:"int" = invoke %18 @java.ref:"java.lang.String::length():int";
                        %20 : java.type:"java.lang.Integer" = invoke %19 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %21 : java.type:"java.lang.String" = invoke %17 %20 @java.ref:"java.lang.String::formatted(java.lang.Object[]):java.lang.String" @invoke.kind="INSTANCE" @invoke.varargs=true;
                        %22 : java.type:"java.lang.String" = concat %16 %21;
                        var.store %3 %22;
                        yield;
                    }
                    (%23 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %24 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %25 : java.type:"boolean" = pattern.match %23
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" -> {
                                        %26 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                        %27 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" = pattern.record %26 @java.ref:"(java.lang.Number n)SwitchStatementTest$R";
                                        yield %27;
                                    }
                                    (%28 : java.type:"java.lang.Number")java.type:"void" -> {
                                        var.store %6 %28;
                                        yield;
                                    };
                                yield %25;
                            }
                            ()java.type:"boolean" -> {
                                %29 : java.type:"java.lang.Number" = var.load %6;
                                %30 : java.type:"java.lang.Class<? extends java.lang.Number>" = invoke %29 @java.ref:"java.lang.Object::getClass():java.lang.Class";
                                %31 : java.type:"java.lang.Class" = constant @java.type:"java.lang.Double";
                                %32 : java.type:"boolean" = invoke %30 %31 @java.ref:"java.lang.Object::equals(java.lang.Object):boolean";
                                yield %32;
                            };
                        yield %24;
                    }
                    ()java.type:"void" -> {
                        %33 : java.type:"java.lang.String" = var.load %3;
                        %34 : java.type:"java.lang.String" = constant @"R(Double)";
                        %35 : java.type:"java.lang.String" = concat %33 %34;
                        var.store %3 %35;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %36 : java.type:"boolean" = constant @true;
                        yield %36;
                    }
                    ()java.type:"void" -> {
                        %37 : java.type:"java.lang.String" = var.load %3;
                        %38 : java.type:"java.lang.String" = constant @"else";
                        %39 : java.type:"java.lang.String" = concat %37 %38;
                        var.store %3 %39;
                        yield;
                    };
                %40 : java.type:"java.lang.String" = var.load %3;
                return %40;
            };
            """)
    @Reflect
    static String casePatternGuard(Object obj) {
        String r = "";
        switch (obj) {
            case String s when s.length() > 3 -> r += "str with length > %d".formatted(s.length());
            case R(Number n) when n.getClass().equals(Double.class) -> r += "R(Double)";
            default -> r += "else";
        }
        return r;
    }

    @IR("""
            func @"defaultCaseNotTheLast" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"java.lang.String" = var.load %1;
                java.switch.statement %4
                    ()java.type:"boolean" -> {
                        %17 : java.type:"boolean" = constant @true;
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"else";
                        %20 : java.type:"java.lang.String" = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    }
                    (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %6 : java.type:"java.lang.String" = constant @"M";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"Mow";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %12 : java.type:"java.lang.String" = constant @"A";
                        %13 : java.type:"boolean" = invoke %11 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %3;
                        %15 : java.type:"java.lang.String" = constant @"Aow";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    };
                %21 : java.type:"java.lang.String" = var.load %3;
                return %21;
            };
            """)
    @Reflect
    static String defaultCaseNotTheLast(String s) {
        String r = "";
        switch (s) {
            default -> r += "else";
            case "M" -> r += "Mow";
            case "A" -> r += "Aow";
        }
        return r;
    }

    @IR("""
            func @"f" (%0 : java.type:"int")java.type:"int" -> {
                  %1 : Var<java.type:"int"> = var %0 @"i";
                  %2 : java.type:"int" = var.load %1;
                  java.switch.statement %2
                      (%3 : java.type:"int")java.type:"boolean" -> {
                          %4 : java.type:"int" = constant @0;
                          %5 : java.type:"boolean" = eq %3 %4;
                          yield %5;
                      }
                      ()java.type:"void" -> {
                          %6 : java.type:"int" = constant @0;
                          return %6;
                      };
                  %7 : java.type:"int" = constant @0;
                  return %7;
              };
            """)
    @Reflect
    static int f(int i) {
        switch (i) {
            case 0 -> {
                return 0;
            }
        }
        return 0;
    }

    @IR("""
            func @"outOfOrderFallThrought" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                  %1 : Var<java.type:"int"> = var %0 @"i";
                  %2 : java.type:"java.lang.String" = constant @"";
                  %3 : Var<java.type:"java.lang.String"> = var %2 @"ret";
                  %4 : java.type:"int" = var.load %1;
                  java.switch.statement %4
                      ()java.type:"boolean" -> {
                          %5 : java.type:"boolean" = constant @true;
                          yield %5;
                      }
                      ()java.type:"void" -> {
                          %6 : java.type:"java.lang.String" = var.load %3;
                          %7 : java.type:"java.lang.String" = constant @"? ";
                          %8 : java.type:"java.lang.String" = concat %6 %7;
                          var.store %3 %8;
                          java.switch.fallthrough;
                      }
                      (%9 : java.type:"int")java.type:"boolean" -> {
                          %10 : java.type:"int" = constant @4;
                          %11 : java.type:"boolean" = eq %9 %10;
                          yield %11;
                      }
                      ()java.type:"void" -> {
                          %12 : java.type:"java.lang.String" = var.load %3;
                          %13 : java.type:"java.lang.String" = constant @"four ";
                          %14 : java.type:"java.lang.String" = concat %12 %13;
                          var.store %3 %14;
                          java.switch.fallthrough;
                      }
                      (%15 : java.type:"int")java.type:"boolean" -> {
                          %16 : java.type:"int" = constant @2;
                          %17 : java.type:"boolean" = eq %15 %16;
                          yield %17;
                      }
                      ()java.type:"void" -> {
                          %18 : java.type:"java.lang.String" = var.load %3;
                          %19 : java.type:"java.lang.String" = constant @"two ";
                          %20 : java.type:"java.lang.String" = concat %18 %19;
                          var.store %3 %20;
                          java.switch.fallthrough;
                      }
                      (%21 : java.type:"int")java.type:"boolean" -> {
                          %22 : java.type:"int" = constant @3;
                          %23 : java.type:"boolean" = eq %21 %22;
                          yield %23;
                      }
                      ()java.type:"void" -> {
                          %24 : java.type:"java.lang.String" = var.load %3;
                          %25 : java.type:"java.lang.String" = constant @"three ";
                          %26 : java.type:"java.lang.String" = concat %24 %25;
                          var.store %3 %26;
                          java.switch.fallthrough;
                      }
                      (%27 : java.type:"int")java.type:"boolean" -> {
                          %28 : java.type:"int" = constant @1;
                          %29 : java.type:"boolean" = eq %27 %28;
                          yield %29;
                      }
                      ()java.type:"void" -> {
                          %30 : java.type:"java.lang.String" = var.load %3;
                          %31 : java.type:"java.lang.String" = constant @"one";
                          %32 : java.type:"java.lang.String" = concat %30 %31;
                          var.store %3 %32;
                          yield;
                      };
                  %33 : java.type:"java.lang.String" = var.load %3;
                  return %33;
              };
            """)
    @Reflect
    static String outOfOrderFallThrought(int i) {
        String ret = "";
        switch (i) {
            default:
                ret += "? ";
            case 4:
                ret += "four ";
            case 2:
                ret += "two ";
            case 3:
                ret += "three ";
            case 1:
                ret += "one";
        }
        return ret;
    }

    @IR("""
            func @"caseConstantPrimitiveWrapperSelector" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                  %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                  %2 : java.type:"java.lang.String" = constant @"";
                  %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                  %4 : java.type:"java.lang.Integer" = var.load %1;
                  java.switch.statement %4
                      (%5 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                          %6 : java.type:"int" = invoke %5 @java.ref:"java.lang.Integer::intValue():int";
                          %7 : java.type:"int" = constant @1;
                          %8 : java.type:"boolean" = eq %6 %7;
                          yield %8;
                      }
                      ()java.type:"void" -> {
                          %9 : java.type:"java.lang.String" = var.load %3;
                          %10 : java.type:"java.lang.String" = constant @"one";
                          %11 : java.type:"java.lang.String" = concat %9 %10;
                          var.store %3 %11;
                          yield;
                      }
                      (%12 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                          %13 : java.type:"boolean" = java.cor
                              ()java.type:"boolean" -> {
                                  %14 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::intValue():int";
                                  %15 : java.type:"int" = constant @2;
                                  %16 : java.type:"boolean" = eq %14 %15;
                                  yield %16;
                              }
                              ()java.type:"boolean" -> {
                                  %17 : java.type:"int" = invoke %12 @java.ref:"java.lang.Integer::intValue():int";
                                  %18 : java.type:"int" = constant @3;
                                  %19 : java.type:"boolean" = eq %17 %18;
                                  yield %19;
                              };
                          yield %13;
                      }
                      ()java.type:"void" -> {
                          %20 : java.type:"java.lang.String" = var.load %3;
                          %21 : java.type:"java.lang.String" = constant @"two or three";
                          %22 : java.type:"java.lang.String" = concat %20 %21;
                          var.store %3 %22;
                          yield;
                      }
                      ()java.type:"boolean" -> {
                          %23 : java.type:"boolean" = constant @true;
                          yield %23;
                      }
                      ()java.type:"void" -> {
                          %24 : java.type:"java.lang.String" = var.load %3;
                          %25 : java.type:"java.lang.String" = constant @"else";
                          %26 : java.type:"java.lang.String" = concat %24 %25;
                          var.store %3 %26;
                          yield;
                      };
                  %27 : java.type:"java.lang.String" = var.load %3;
                  return %27;
              };
            """)
    @Reflect
    static String caseConstantPrimitiveWrapperSelector(Integer i) {
        String r = "";
        switch (i) {
            case 1 -> r += "one";
            case 2, 3 -> r += "two or three";
            default -> r += "else";
        };
        return r;
    }

    @IR("""
            func @"constantLabelCasted" (%0 : java.type:"int")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"java.lang.String" = constant @"";
                %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                %4 : java.type:"int" = var.load %1;
                java.switch.statement %4
                    (%5 : java.type:"int")java.type:"boolean" -> {
                        %6 : java.type:"int" = constant @1;
                        %7 : java.type:"boolean" = eq %5 %6;
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"java.lang.String" = constant @"one";
                        %10 : java.type:"java.lang.String" = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %11 : java.type:"boolean" = constant @true;
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %3;
                        %13 : java.type:"java.lang.String" = constant @"not one";
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    };
                %15 : java.type:"java.lang.String" = var.load %3;
                return %15;
            };
            """)
    @Reflect
    static String constantLabelCasted(int i) {
        String r = "";
        switch (i) {
            case (byte) 1 -> r += "one";
            default -> r += "not one";
        };
        return r;
    }

    @IR("""
            func @"caseConstantStringLiteral" (%0 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %1 : Var<java.type:"java.lang.String"> = var %0 @"s";
                  %2 : java.type:"java.lang.String" = constant @"";
                  %3 : Var<java.type:"java.lang.String"> = var %2 @"r";
                  %4 : java.type:"java.lang.String" = var.load %1;
                  java.switch.statement %4
                      (%5 : java.type:"java.lang.String")java.type:"boolean" -> {
                          %6 : java.type:"java.lang.String" = constant @"1";
                          %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                          yield %7;
                      }
                      ()java.type:"void" -> {
                          %8 : java.type:"java.lang.String" = var.load %3;
                          %9 : java.type:"java.lang.String" = constant @"one";
                          %10 : java.type:"java.lang.String" = concat %8 %9;
                          var.store %3 %10;
                          yield;
                      }
                      (%11 : java.type:"java.lang.String")java.type:"boolean" -> {
                          %12 : java.type:"boolean" = java.cor
                              ()java.type:"boolean" -> {
                                  %13 : java.type:"java.lang.String" = constant @"2";
                                  %14 : java.type:"boolean" = invoke %11 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                  yield %14;
                              }
                              ()java.type:"boolean" -> {
                                  %15 : java.type:"java.lang.String" = constant @"3";
                                  %16 : java.type:"boolean" = invoke %11 %15 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                  yield %16;
                              };
                          yield %12;
                      }
                      ()java.type:"void" -> {
                          %17 : java.type:"java.lang.String" = var.load %3;
                          %18 : java.type:"java.lang.String" = constant @"two or three";
                          %19 : java.type:"java.lang.String" = concat %17 %18;
                          var.store %3 %19;
                          yield;
                      }
                      ()java.type:"boolean" -> {
                          %20 : java.type:"boolean" = constant @true;
                          yield %20;
                      }
                      ()java.type:"void" -> {
                          %21 : java.type:"java.lang.String" = var.load %3;
                          %22 : java.type:"java.lang.String" = constant @"else";
                          %23 : java.type:"java.lang.String" = concat %21 %22;
                          var.store %3 %23;
                          yield;
                      };
                  %24 : java.type:"java.lang.String" = var.load %3;
                  return %24;
              };
            """)
    @Reflect
    static String caseConstantStringLiteral(String s) {
        String r = "";
        switch (s) {
            case "1" -> r += "one";
            case "2", "3" -> r+= "two or three";
            default -> r += "else";
        };
        return r;
    }

    @IR("""
            func @"caseBoxedGuard" (%0 : java.type:"java.lang.Object", %1 : java.type:"java.lang.Boolean")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %3 : Var<java.type:"java.lang.Boolean"> = var %1 @"B";
                %4 : java.type:"java.lang.String" = constant @"";
                %5 : Var<java.type:"java.lang.String"> = var %4 @"r";
                %6 : java.type:"java.lang.Object" = var.load %2;
                %7 : Var<java.type:"java.lang.Object"> = var;
                java.switch.statement %6
                    (%8 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %9 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %10 : java.type:"boolean" = pattern.match %8
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                        %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type;
                                        yield %11;
                                    }
                                    (%12 : java.type:"java.lang.Object")java.type:"void" -> {
                                        var.store %7 %12;
                                        yield;
                                    };
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %13 : java.type:"java.lang.Boolean" = var.load %3;
                                %14 : java.type:"boolean" = invoke %13 @java.ref:"java.lang.Boolean::booleanValue():boolean";
                                yield %14;
                            };
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %5;
                        %16 : java.type:"java.lang.String" = constant @"match";
                        %17 : java.type:"java.lang.String" = concat %15 %16;
                        var.store %5 %17;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %18 : java.type:"boolean" = constant @true;
                        yield %18;
                    }
                    ()java.type:"void" -> {
                        %19 : java.type:"java.lang.String" = var.load %5;
                        %20 : java.type:"java.lang.String" = constant @"no match";
                        %21 : java.type:"java.lang.String" = concat %19 %20;
                        var.store %5 %21;
                        yield;
                    };
                %22 : java.type:"java.lang.String" = var.load %5;
                return %22;
            };
            """)
    @Reflect
    private static String caseBoxedGuard(Object o, Boolean B) {
        String r = "";
        switch (o) {
            case Object _ when B -> r += "match";
            default -> r += "no match";
        }
        return r;
    }

    @IR("""
        func @"caseReassignVar" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"sel";
            %2 : java.type:"int" = var.load %1;
            java.switch.statement %2
                (%3 : java.type:"int")java.type:"boolean" -> {
                    %4 : java.type:"int" = constant @0;
                    %5 : java.type:"boolean" = eq %3 %4;
                    yield %5;
                }
                ()java.type:"void" -> {
                    %6 : java.type:"int" = constant @1;
                    %7 : Var<java.type:"int"> = var %6 @"i";
                    %8 : java.type:"int" = var.load %7;
                    return %8;
                }
                (%9 : java.type:"int")java.type:"boolean" -> {
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"boolean" = eq %9 %10;
                    yield %11;
                }
                ()java.type:"void" -> {
                    %12 : Var<java.type:"int"> = var @"i";
                    %13 : java.type:"int" = constant @2;
                    var.store %12 %13;
                    %14 : java.type:"int" = var.load %12;
                    return %14;
                }
                ()java.type:"boolean" -> {
                    %15 : java.type:"boolean" = constant @true;
                    yield %15;
                }
                ()java.type:"void" -> {
                    %16 : java.type:"int" = constant @-1;
                    return %16;
                };
            unreachable;
        };
        """)
    @Reflect
    private static int caseReassignVar(int sel) {
        switch (sel) {
            case 0:
                int i = 1;
                return i;
            case 1:
                i = 2;
                return i;
            default:
                return -1;
        }
    }

    @IR("""
        func @"caseReassignVarFallThrough" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"sel";
            %2 : java.type:"int" = var.load %1;
            java.switch.statement %2
                (%3 : java.type:"int")java.type:"boolean" -> {
                    %4 : java.type:"int" = constant @0;
                    %5 : java.type:"boolean" = eq %3 %4;
                    yield %5;
                }
                ()java.type:"void" -> {
                    %6 : java.type:"int" = constant @1;
                    %7 : Var<java.type:"int"> = var %6 @"i";
                    java.switch.fallthrough;
                }
                (%9 : java.type:"int")java.type:"boolean" -> {
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"boolean" = eq %9 %10;
                    yield %11;
                }
                ()java.type:"void" -> {
                    %12 : Var<java.type:"int"> = var @"i";
                    %13 : java.type:"int" = constant @2;
                    var.store %12 %13;
                    %14 : java.type:"int" = var.load %12;
                    return %14;
                }
                ()java.type:"boolean" -> {
                    %15 : java.type:"boolean" = constant @true;
                    yield %15;
                }
                ()java.type:"void" -> {
                    %16 : java.type:"int" = constant @-1;
                    return %16;
                };
            unreachable;
        };
        """)
    @Reflect
    private static int caseReassignVarFallThrough(int sel) {
        switch (sel) {
            case 0:
                int i = 1;
            case 1:
                i = 2;
                return i;
            default:
                return -1;
        }
    }

    @IR("""
        func @"caseReassignVarNested" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"sel";
            %2 : java.type:"int" = var.load %1;
            java.switch.statement %2
                (%3 : java.type:"int")java.type:"boolean" -> {
                    %4 : java.type:"int" = constant @0;
                    %5 : java.type:"boolean" = eq %3 %4;
                    yield %5;
                }
                ()java.type:"void" -> {
                    %6 : java.type:"int" = constant @1;
                    %7 : Var<java.type:"int"> = var %6 @"i";
                    %8 : java.type:"int" = var.load %7;
                    return %8;
                }
                (%9 : java.type:"int")java.type:"boolean" -> {
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"boolean" = eq %9 %10;
                    yield %11;
                }
                ()java.type:"void" -> {
                    %12 : Var<java.type:"int"> = var @"i";
                    %13 : java.type:"int" = constant @2;
                    var.store %12 %13;
                    %14 : Var<java.type:"int"> = var %13 @"j";
                    %15 : java.type:"int" = var.load %12;
                    %16 : java.type:"int" = var.load %14;
                    %17 : java.type:"int" = add %15 %16;
                    return %17;
                }
                ()java.type:"boolean" -> {
                    %18 : java.type:"boolean" = constant @true;
                    yield %18;
                }
                ()java.type:"void" -> {
                    %19 : java.type:"int" = constant @-1;
                    return %19;
                };
            unreachable;
        };
        """)
    @Reflect
    private static int caseReassignVarNested(int sel) {
        switch (sel) {
            case 0:
                int i = 1;
                return i;
            case 1:
                int j = (i = 2);
                return i + j;
            default:
                return -1;
        }
    }

    @IR("""
        func @"caseReassignVarNestedBlock" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"sel";
            %2 : java.type:"int" = var.load %1;
            java.switch.statement %2
                (%3 : java.type:"int")java.type:"boolean" -> {
                    %4 : java.type:"int" = constant @0;
                    %5 : java.type:"boolean" = eq %3 %4;
                    yield %5;
                }
                ()java.type:"void" -> {
                    %6 : java.type:"int" = constant @1;
                    %7 : Var<java.type:"int"> = var %6 @"i";
                    %8 : java.type:"int" = var.load %7;
                    return %8;
                }
                (%9 : java.type:"int")java.type:"boolean" -> {
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"boolean" = eq %9 %10;
                    yield %11;
                }
                ()java.type:"void" -> {
                    %12 : Var<java.type:"int"> = var @"i";
                    java.block ()java.type:"void" -> {
                        %13 : java.type:"int" = constant @2;
                        var.store %12 %13;
                        yield;
                    };
                    %14 : java.type:"int" = var.load %12;
                    return %14;
                }
                ()java.type:"boolean" -> {
                    %15 : java.type:"boolean" = constant @true;
                    yield %15;
                }
                ()java.type:"void" -> {
                    %16 : java.type:"int" = constant @-1;
                    return %16;
                };
            unreachable;
        };
        """)
    @Reflect
    private static int caseReassignVarNestedBlock(int sel) {
        switch (sel) {
            case 0:
                int i = 1;
                return i;
            case 1:
                {
                    i = 2;
                }
                return i;
            default:
                return -1;
        }
    }

    @IR("""
        func @"caseReassignVarExpression" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"sel";
            %2 : java.type:"int" = var.load %1;
            java.switch.statement %2
                (%3 : java.type:"int")java.type:"boolean" -> {
                    %4 : java.type:"int" = constant @0;
                    %5 : java.type:"boolean" = eq %3 %4;
                    yield %5;
                }
                ()java.type:"void" -> {
                    %6 : java.type:"int" = constant @1;
                    %7 : Var<java.type:"int"> = var %6 @"i";
                    %8 : java.type:"int" = var.load %7;
                    return %8;
                }
                (%9 : java.type:"int")java.type:"boolean" -> {
                    %10 : java.type:"int" = constant @1;
                    %11 : java.type:"boolean" = eq %9 %10;
                    yield %11;
                }
                ()java.type:"void" -> {
                    %12 : Var<java.type:"int"> = var @"i";
                    %13 : java.type:"int" = constant @2;
                    var.store %12 %13;
                    return %13;
                }
                ()java.type:"boolean" -> {
                    %14 : java.type:"boolean" = constant @true;
                    yield %14;
                }
                ()java.type:"void" -> {
                    %15 : java.type:"int" = constant @-1;
                    return %15;
                };
            unreachable;
        };
        """)
    @Reflect
    private static int caseReassignVarExpression(int sel) {
        switch (sel) {
            case 0:
                int i = 1;
                return i;
            case 1:
                return i = 2;
            default:
                return -1;
        }
    }
}
