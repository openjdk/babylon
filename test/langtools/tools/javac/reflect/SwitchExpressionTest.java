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
import java.util.function.Consumer;
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with switch expressions.
 * @enablePreview
 * @build SwitchExpressionTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchExpressionTest
 */

public class SwitchExpressionTest {

    @CodeReflection
    @IR("""
            func @"constantCaseLabelRule" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.Object -> {
                        %7 : java.lang.String = constant @"FOO";
                        yield %7;
                    }
                    ^constantCaseLabel(%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.Object -> {
                        %11 : java.lang.String = constant @"FOO";
                        yield %11;
                    }
                    ^constantCaseLabel(%12 : java.lang.String)boolean -> {
                        %13 : java.lang.String = constant @"BAZ";
                        %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %14;
                    }
                    ()java.lang.Object -> {
                        %15 : java.lang.String = constant @"FOO";
                        yield %15;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %16 : java.lang.String = constant @"";
                        yield %16;
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
            func @"constantCaseLabelsRule" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : boolean = java.cor
                            ()boolean -> {
                                %6 : java.lang.String = constant @"FOO";
                                %7 : boolean = invoke %4 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %7;
                            }
                            ()boolean -> {
                                %8 : java.lang.String = constant @"BAR";
                                %9 : boolean = invoke %4 %8 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %9;
                            }
                            ()boolean -> {
                                %10 : java.lang.String = constant @"BAZ";
                                %11 : boolean = invoke %4 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %11;
                            };
                        yield %5;
                    }
                    ()java.lang.Object -> {
                        %12 : java.lang.String = constant @"FOO";
                        yield %12;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %13 : java.lang.String = constant @"";
                        java.yield %13;
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
            func @"constantCaseLabelStatement" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.Object -> {
                        %7 : java.lang.String = constant @"FOO";
                        java.yield %7;
                    }
                    ^constantCaseLabel(%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.Object -> {
                        %11 : java.lang.String = constant @"FOO";
                        java.yield %11;
                    }
                    ^constantCaseLabel(%12 : java.lang.String)boolean -> {
                        %13 : java.lang.String = constant @"BAZ";
                        %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %14;
                    }
                    ()java.lang.Object -> {
                        %15 : java.lang.String = constant @"FOO";
                        java.yield %15;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %16 : java.lang.String = constant @"";
                        java.yield %16;
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
            func @"constantCaseLabelsStatement" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : boolean = java.cor
                            ()boolean -> {
                                %6 : java.lang.String = constant @"FOO";
                                %7 : boolean = invoke %4 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %7;
                            }
                            ()boolean -> {
                                %8 : java.lang.String = constant @"BAR";
                                %9 : boolean = invoke %4 %8 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %9;
                            }
                            ()boolean -> {
                                %10 : java.lang.String = constant @"BAZ";
                                %11 : boolean = invoke %4 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %11;
                            };
                        yield %5;
                    }
                    ()java.lang.Object -> {
                        %12 : java.lang.String = constant @"FOO";
                        java.yield %12;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        java.block ()void -> {
                            %13 : java.lang.String = constant @"";
                            java.yield %13;
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
            func @"constantCaseLabelStatements" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.Object -> {
                        java.block ()void -> {
                            %7 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                            %8 : java.lang.String = constant @"FOO";
                            invoke %7 %8 @"java.io.PrintStream::println(java.lang.String)void";
                            yield;
                        };
                        java.block ()void -> {
                            %9 : java.lang.String = constant @"FOO";
                            java.yield %9;
                        };
                        unreachable;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %10 : java.lang.String = constant @"";
                        java.yield %10;
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
            func @"constantCaseLabelFallthrough" (%0 : java.lang.String)java.lang.Object -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = java.switch.expression %2
                    ^constantCaseLabel(%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.Object -> {
                        java.block ()void -> {
                            %7 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                            %8 : java.lang.String = constant @"FOO";
                            invoke %7 %8 @"java.io.PrintStream::println(java.lang.String)void";
                            yield;
                        };
                        java.switch.fallthrough;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %9 : java.lang.String = constant @"";
                        java.yield %9;
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
            func @"patternCaseLabel" (%0 : java.lang.Object)java.lang.Object -> {
                %1 : Var<java.lang.Object> = var %0 @"r";
                %2 : java.lang.Object = var.load %1;
                %3 : java.lang.Number = constant @null;
                %4 : Var<java.lang.Number> = var %3 @"n";
                %5 : java.lang.String = constant @null;
                %6 : Var<java.lang.String> = var %5 @"s";
                %7 : java.lang.Object = java.switch.expression %2
                    ^patternCaseLabel(%8 : java.lang.Object)boolean -> {
                        %9 : boolean = pattern.match %8
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Record<SwitchExpressionTest$A> -> {
                                %10 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                %11 : java.lang.reflect.code.ExtendedOps$Pattern$Record<SwitchExpressionTest$A> = pattern.record %10 @"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %11;
                            }
                            ^match(%12 : java.lang.Number)void -> {
                                var.store %4 %12;
                                yield;
                            };
                        yield %9;
                    }
                    ()java.lang.Object -> {
                        %13 : java.lang.Number = var.load %4;
                        java.yield %13;
                    }
                    ^patternCaseLabel(%14 : java.lang.Object)boolean -> {
                        %15 : boolean = pattern.match %14
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                %16 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %16;
                            }
                            ^match(%17 : java.lang.String)void -> {
                                var.store %6 %17;
                                yield;
                            };
                        yield %15;
                    }
                    ()java.lang.Object -> {
                        %18 : java.lang.String = var.load %6;
                        java.yield %18;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %19 : java.lang.String = constant @"";
                        java.yield %19;
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
            func @"patternCaseLabelGuard" (%0 : java.lang.Object)java.lang.Object -> {
                %1 : Var<java.lang.Object> = var %0 @"r";
                %2 : java.lang.Object = var.load %1;
                %3 : java.lang.Number = constant @null;
                %4 : Var<java.lang.Number> = var %3 @"n";
                %5 : java.lang.String = constant @null;
                %6 : Var<java.lang.String> = var %5 @"s";
                %7 : java.lang.String = constant @null;
                %8 : Var<java.lang.String> = var %7 @"s";
                %9 : java.lang.Object = java.switch.expression %2
                    ^patternCaseLabel(%10 : java.lang.Object)boolean -> {
                        %11 : boolean = pattern.match %10
                            ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Record<SwitchExpressionTest$A> -> {
                                %12 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                %13 : java.lang.reflect.code.ExtendedOps$Pattern$Record<SwitchExpressionTest$A> = pattern.record %12 @"(java.lang.Number n)SwitchExpressionTest$A";
                                yield %13;
                            }
                            ^match(%14 : java.lang.Number)void -> {
                                var.store %4 %14;
                                yield;
                            };
                        yield %11;
                    }
                    ()java.lang.Object -> {
                        %15 : java.lang.Number = var.load %4;
                        java.yield %15;
                    }
                    ^patternCaseLabel(%16 : java.lang.Object)boolean -> {
                        %17 : boolean = java.cand
                            ()boolean -> {
                                %18 : boolean = pattern.match %16
                                    ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                        %19 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                        yield %19;
                                    }
                                    ^match(%20 : java.lang.String)void -> {
                                        var.store %6 %20;
                                        yield;
                                    };
                                yield %18;
                            }
                            ()boolean -> {
                                %21 : java.lang.String = var.load %6;
                                %22 : int = invoke %21 @"java.lang.String::length()int";
                                %23 : int = constant @"5";
                                %24 : boolean = lt %22 %23;
                                yield %24;
                            };
                        yield %17;
                    }
                    ()java.lang.Object -> {
                        %25 : java.lang.String = var.load %6;
                        java.yield %25;
                    }
                    ^patternCaseLabel(%26 : java.lang.Object)boolean -> {
                        %27 : boolean = java.cand
                            ()boolean -> {
                                %28 : boolean = pattern.match %26
                                    ^pattern()java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> -> {
                                        %29 : java.lang.reflect.code.ExtendedOps$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                        yield %29;
                                    }
                                    ^match(%30 : java.lang.String)void -> {
                                        var.store %8 %30;
                                        yield;
                                    };
                                yield %28;
                            }
                            ()boolean -> {
                                %31 : java.lang.String = var.load %8;
                                %32 : int = invoke %31 @"java.lang.String::length()int";
                                %33 : int = constant @"10";
                                %34 : boolean = lt %32 %33;
                                yield %34;
                            };
                        yield %27;
                    }
                    ()java.lang.Object -> {
                        %35 : java.lang.String = var.load %8;
                        java.yield %35;
                    }
                    ^defaultCaseLabel()void -> {
                        yield;
                    }
                    ()java.lang.Object -> {
                        %36 : java.lang.String = constant @"";
                        java.yield %36;
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
