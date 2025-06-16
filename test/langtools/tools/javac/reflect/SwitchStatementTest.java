import jdk.incubator.code.CodeReflection;
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
    @CodeReflection
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
    @CodeReflection
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
    @CodeReflection
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
    @CodeReflection
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
                        %6 : java.type:"int" = constant @8;
                        %7 : java.type:"java.lang.Integer" = invoke %6 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %8 : java.type:"boolean" = invoke %5 %7 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %9 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %9;
                    }
                    (%10 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %11 : java.type:"int" = constant @9;
                        %12 : java.type:"java.lang.Integer" = invoke %11 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %13 : java.type:"boolean" = invoke %10 %12 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
    @CodeReflection
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
                java.switch.statement %4
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
    @CodeReflection
    private static String caseConstantNullLabel(String s) {
        String r = "";
        switch (s) {
            case null -> r += "null";
            default -> r += "non null";
        }
        return r;
    }

//    @CodeReflection
//    @@@ not supported
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
    @CodeReflection
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
                                %8 : java.type:"boolean" = invoke %5 %7 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %8;
                            }
                            ()java.type:"boolean" -> {
                                %9 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::FRI:SwitchStatementTest$Day";
                                %10 : java.type:"boolean" = invoke %5 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %10;
                            }
                            ()java.type:"boolean" -> {
                                %11 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::SUN:SwitchStatementTest$Day";
                                %12 : java.type:"boolean" = invoke %5 %11 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %18 : java.type:"boolean" = invoke %16 %17 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                                %25 : java.type:"boolean" = invoke %22 %24 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %25;
                            }
                            ()java.type:"boolean" -> {
                                %26 : java.type:"SwitchStatementTest$Day" = field.load @java.ref:"SwitchStatementTest$Day::SAT:SwitchStatementTest$Day";
                                %27 : java.type:"boolean" = invoke %22 %26 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %33 : java.type:"boolean" = invoke %31 %32 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
    @CodeReflection
    private static String caseConstantEnum(Day d) {
        String r = "";
        switch (d) {
            // @@@ concat of String and int is modeled as: add str Integer
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
                %5 : Var<java.type:"int"> = var %4 @"eleven";
                %6 : java.type:"int" = var.load %1;
                java.switch.statement %6
                    (%7 : java.type:"int")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"int" = constant @15;
                        %10 : java.type:"int" = and %8 %9;
                        %11 : java.type:"boolean" = eq %7 %10;
                        yield %11;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %3;
                        %13 : java.type:"int" = constant @1;
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    }
                    (%15 : java.type:"int")java.type:"boolean" -> {
                        %16 : java.type:"int" = constant @4;
                        %17 : java.type:"int" = constant @1;
                        %18 : java.type:"int" = ashr %16 %17;
                        %19 : java.type:"boolean" = eq %15 %18;
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"2";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    (%23 : java.type:"int")java.type:"boolean" -> {
                        %24 : java.type:"long" = constant @3;
                        %25 : java.type:"int" = conv %24;
                        %26 : java.type:"boolean" = eq %23 %25;
                        yield %26;
                    }
                    ()java.type:"void" -> {
                        %27 : java.type:"java.lang.String" = var.load %3;
                        %28 : java.type:"int" = constant @3;
                        %29 : java.type:"java.lang.String" = concat %27 %28;
                        var.store %3 %29;
                        yield;
                    }
                    (%30 : java.type:"int")java.type:"boolean" -> {
                        %31 : java.type:"int" = constant @2;
                        %32 : java.type:"int" = constant @1;
                        %33 : java.type:"int" = lshl %31 %32;
                        %34 : java.type:"boolean" = eq %30 %33;
                        yield %34;
                    }
                    ()java.type:"void" -> {
                        %35 : java.type:"java.lang.String" = var.load %3;
                        %36 : java.type:"int" = constant @4;
                        %37 : java.type:"java.lang.String" = concat %35 %36;
                        var.store %3 %37;
                        yield;
                    }
                    (%38 : java.type:"int")java.type:"boolean" -> {
                        %39 : java.type:"int" = constant @10;
                        %40 : java.type:"int" = constant @2;
                        %41 : java.type:"int" = div %39 %40;
                        %42 : java.type:"boolean" = eq %38 %41;
                        yield %42;
                    }
                    ()java.type:"void" -> {
                        %43 : java.type:"java.lang.String" = var.load %3;
                        %44 : java.type:"int" = constant @5;
                        %45 : java.type:"java.lang.String" = concat %43 %44;
                        var.store %3 %45;
                        yield;
                    }
                    (%46 : java.type:"int")java.type:"boolean" -> {
                        %47 : java.type:"int" = constant @12;
                        %48 : java.type:"int" = constant @6;
                        %49 : java.type:"int" = sub %47 %48;
                        %50 : java.type:"boolean" = eq %46 %49;
                        yield %50;
                    }
                    ()java.type:"void" -> {
                        %51 : java.type:"java.lang.String" = var.load %3;
                        %52 : java.type:"int" = constant @6;
                        %53 : java.type:"java.lang.String" = concat %51 %52;
                        var.store %3 %53;
                        yield;
                    }
                    (%54 : java.type:"int")java.type:"boolean" -> {
                        %55 : java.type:"int" = constant @3;
                        %56 : java.type:"int" = constant @4;
                        %57 : java.type:"int" = add %55 %56;
                        %58 : java.type:"boolean" = eq %54 %57;
                        yield %58;
                    }
                    ()java.type:"void" -> {
                        %59 : java.type:"java.lang.String" = var.load %3;
                        %60 : java.type:"int" = constant @7;
                        %61 : java.type:"java.lang.String" = concat %59 %60;
                        var.store %3 %61;
                        yield;
                    }
                    (%62 : java.type:"int")java.type:"boolean" -> {
                        %63 : java.type:"int" = constant @2;
                        %64 : java.type:"int" = constant @2;
                        %65 : java.type:"int" = mul %63 %64;
                        %66 : java.type:"int" = constant @2;
                        %67 : java.type:"int" = mul %65 %66;
                        %68 : java.type:"boolean" = eq %62 %67;
                        yield %68;
                    }
                    ()java.type:"void" -> {
                        %69 : java.type:"java.lang.String" = var.load %3;
                        %70 : java.type:"int" = constant @8;
                        %71 : java.type:"java.lang.String" = concat %69 %70;
                        var.store %3 %71;
                        yield;
                    }
                    (%72 : java.type:"int")java.type:"boolean" -> {
                        %73 : java.type:"int" = constant @8;
                        %74 : java.type:"int" = constant @1;
                        %75 : java.type:"int" = or %73 %74;
                        %76 : java.type:"boolean" = eq %72 %75;
                        yield %76;
                    }
                    ()java.type:"void" -> {
                        %77 : java.type:"java.lang.String" = var.load %3;
                        %78 : java.type:"int" = constant @9;
                        %79 : java.type:"java.lang.String" = concat %77 %78;
                        var.store %3 %79;
                        yield;
                    }
                    (%80 : java.type:"int")java.type:"boolean" -> {
                        %81 : java.type:"int" = constant @10;
                        %82 : java.type:"boolean" = eq %80 %81;
                        yield %82;
                    }
                    ()java.type:"void" -> {
                        %83 : java.type:"java.lang.String" = var.load %3;
                        %84 : java.type:"int" = constant @10;
                        %85 : java.type:"java.lang.String" = concat %83 %84;
                        var.store %3 %85;
                        yield;
                    }
                    (%86 : java.type:"int")java.type:"boolean" -> {
                        %87 : java.type:"int" = var.load %5;
                        %88 : java.type:"boolean" = eq %86 %87;
                        yield %88;
                    }
                    ()java.type:"void" -> {
                        %89 : java.type:"java.lang.String" = var.load %3;
                        %90 : java.type:"int" = constant @11;
                        %91 : java.type:"java.lang.String" = concat %89 %90;
                        var.store %3 %91;
                        yield;
                    }
                    (%92 : java.type:"int")java.type:"boolean" -> {
                        %93 : java.type:"int" = field.load @java.ref:"SwitchStatementTest$Constants::c1:int";
                        %94 : java.type:"boolean" = eq %92 %93;
                        yield %94;
                    }
                    ()java.type:"void" -> {
                        %95 : java.type:"java.lang.String" = var.load %3;
                        %96 : java.type:"int" = field.load @java.ref:"SwitchStatementTest$Constants::c1:int";
                        %97 : java.type:"java.lang.String" = concat %95 %96;
                        var.store %3 %97;
                        yield;
                    }
                    (%98 : java.type:"int")java.type:"boolean" -> {
                        %99 : java.type:"int" = java.cexpression
                            ()java.type:"boolean" -> {
                                %100 : java.type:"int" = constant @1;
                                %101 : java.type:"int" = constant @0;
                                %102 : java.type:"boolean" = gt %100 %101;
                                yield %102;
                            }
                            ()java.type:"int" -> {
                                %103 : java.type:"int" = constant @13;
                                yield %103;
                            }
                            ()java.type:"int" -> {
                                %104 : java.type:"int" = constant @133;
                                yield %104;
                            };
                        %105 : java.type:"boolean" = eq %98 %99;
                        yield %105;
                    }
                    ()java.type:"void" -> {
                        %106 : java.type:"java.lang.String" = var.load %3;
                        %107 : java.type:"int" = constant @13;
                        %108 : java.type:"java.lang.String" = concat %106 %107;
                        var.store %3 %108;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %109 : java.type:"boolean" = constant @true;
                        yield %109;
                    }
                    ()java.type:"void" -> {
                        %110 : java.type:"java.lang.String" = var.load %3;
                        %111 : java.type:"java.lang.String" = constant @"an int";
                        %112 : java.type:"java.lang.String" = concat %110 %111;
                        var.store %3 %112;
                        yield;
                    };
                %113 : java.type:"java.lang.String" = var.load %3;
                return %113;
            };
            """)
    @CodeReflection
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
                %4 : Var<java.type:"short"> = var %3 @"s";
                %5 : java.type:"int" = constant @2;
                %6 : java.type:"byte" = conv %5;
                %7 : Var<java.type:"byte"> = var %6 @"b";
                %8 : java.type:"java.lang.String" = constant @"";
                %9 : Var<java.type:"java.lang.String"> = var %8 @"r";
                %10 : java.type:"short" = var.load %1;
                java.switch.statement %10
                    (%11 : java.type:"short")java.type:"boolean" -> {
                        %12 : java.type:"short" = var.load %4;
                        %13 : java.type:"boolean" = eq %11 %12;
                        yield %13;
                    }
                    ()java.type:"void" -> {
                        %14 : java.type:"java.lang.String" = var.load %9;
                        %15 : java.type:"java.lang.String" = constant @"one";
                        %16 : java.type:"java.lang.String" = concat %14 %15;
                        var.store %9 %16;
                        yield;
                    }
                    (%17 : java.type:"short")java.type:"boolean" -> {
                        %18 : java.type:"byte" = var.load %7;
                        %19 : java.type:"short" = conv %18;
                        %20 : java.type:"boolean" = eq %17 %19;
                        yield %20;
                    }
                    ()java.type:"void" -> {
                        %21 : java.type:"java.lang.String" = var.load %9;
                        %22 : java.type:"java.lang.String" = constant @"two";
                        %23 : java.type:"java.lang.String" = concat %21 %22;
                        var.store %9 %23;
                        yield;
                    }
                    (%24 : java.type:"short")java.type:"boolean" -> {
                        %25 : java.type:"int" = constant @3;
                        %26 : java.type:"short" = conv %25;
                        %27 : java.type:"boolean" = eq %24 %26;
                        yield %27;
                    }
                    ()java.type:"void" -> {
                        %28 : java.type:"java.lang.String" = var.load %9;
                        %29 : java.type:"java.lang.String" = constant @"three";
                        %30 : java.type:"java.lang.String" = concat %28 %29;
                        var.store %9 %30;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %31 : java.type:"boolean" = constant @true;
                        yield %31;
                    }
                    ()java.type:"void" -> {
                        %32 : java.type:"java.lang.String" = var.load %9;
                        %33 : java.type:"java.lang.String" = constant @"else";
                        %34 : java.type:"java.lang.String" = concat %32 %33;
                        var.store %9 %34;
                        yield;
                    };
                %35 : java.type:"java.lang.String" = var.load %9;
                return %35;
            };
            """)
    @CodeReflection
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
                %4 : Var<java.type:"byte"> = var %3 @"b";
                %5 : java.type:"java.lang.String" = constant @"";
                %6 : Var<java.type:"java.lang.String"> = var %5 @"r";
                %7 : java.type:"java.lang.Byte" = var.load %1;
                java.switch.statement %7
                    (%8 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %9 : java.type:"int" = constant @1;
                        %10 : java.type:"byte" = conv %9;
                        %11 : java.type:"java.lang.Byte" = invoke %10 @java.ref:"java.lang.Byte::valueOf(byte):java.lang.Byte";
                        %12 : java.type:"boolean" = invoke %8 %11 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %12;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %6;
                        %14 : java.type:"java.lang.String" = constant @"one";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %6 %15;
                        yield;
                    }
                    (%16 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %17 : java.type:"byte" = var.load %4;
                        %18 : java.type:"java.lang.Byte" = invoke %17 @java.ref:"java.lang.Byte::valueOf(byte):java.lang.Byte";
                        %19 : java.type:"boolean" = invoke %16 %18 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %19;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %6;
                        %21 : java.type:"java.lang.String" = constant @"two";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %6 %22;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.String" = var.load %6;
                        %25 : java.type:"java.lang.String" = constant @"default";
                        %26 : java.type:"java.lang.String" = concat %24 %25;
                        var.store %6 %26;
                        yield;
                    };
                %27 : java.type:"java.lang.String" = var.load %6;
                return %27;
            };
            """)
    @CodeReflection
    static String caseConstantConv2(Byte a) {
        final byte b = 2;
        String r = "";
        switch (a) {
            case 1 -> r+= "one"; // narrowing primitive conversion followed by a boxing conversion, int -> bye -> Byte
            case b -> r+= "two"; // boxing, byte -> Byte
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
    @CodeReflection
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
                java.switch.statement %4
                    (%5 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %6 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::A:SwitchStatementTest$E";
                        %7 : java.type:"boolean" = invoke %5 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"void" -> {
                        %8 : java.type:"java.lang.String" = var.load %3;
                        %9 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::A:SwitchStatementTest$E";
                        %10 : java.type:"java.lang.String" = cast %9 @java.type:"java.lang.String";
                        %11 : java.type:"java.lang.String" = concat %8 %10;
                        var.store %3 %11;
                        yield;
                    }
                    (%12 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %13 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::B:SwitchStatementTest$E";
                        %14 : java.type:"boolean" = invoke %12 %13 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %14;
                    }
                    ()java.type:"void" -> {
                        %15 : java.type:"java.lang.String" = var.load %3;
                        %16 : java.type:"SwitchStatementTest$E" = field.load @java.ref:"SwitchStatementTest$E::B:SwitchStatementTest$E";
                        %17 : java.type:"java.lang.String" = cast %16 @java.type:"java.lang.String";
                        %18 : java.type:"java.lang.String" = concat %15 %17;
                        var.store %3 %18;
                        yield;
                    }
                    (%19 : java.type:"SwitchStatementTest$E")java.type:"boolean" -> {
                        %20 : java.type:"java.lang.Object" = constant @null;
                        %21 : java.type:"boolean" = invoke %19 %20 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %21;
                    }
                    ()java.type:"void" -> {
                        %22 : java.type:"java.lang.String" = var.load %3;
                        %23 : java.type:"java.lang.String" = constant @"null";
                        %24 : java.type:"java.lang.String" = concat %22 %23;
                        var.store %3 %24;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %25 : java.type:"boolean" = constant @true;
                        yield %25;
                    }
                    ()java.type:"void" -> {
                        %26 : java.type:"java.lang.MatchException" = new @java.ref:"java.lang.MatchException::()";
                        throw %26;
                    };
                %27 : java.type:"java.lang.String" = var.load %3;
                return %27;
            };
            """)
    @CodeReflection
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
                %5 : java.type:"SwitchStatementTest$K" = constant @null;
                %6 : Var<java.type:"SwitchStatementTest$K"> = var %5 @"k";
                %7 : java.type:"SwitchStatementTest$J" = constant @null;
                %8 : Var<java.type:"SwitchStatementTest$J"> = var %7 @"j";
                java.switch.statement %4
                    (%9 : java.type:"SwitchStatementTest$I")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$K>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$K>" = pattern.type @"k";
                                yield %11;
                            }
                            (%12 : java.type:"SwitchStatementTest$K")java.type:"void" -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"java.lang.String" = constant @"K";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : java.type:"SwitchStatementTest$I")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = pattern.match %16
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$J>" -> {
                                %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchStatementTest$J>" = pattern.type @"j";
                                yield %18;
                            }
                            (%19 : java.type:"SwitchStatementTest$J")java.type:"void" -> {
                                var.store %8 %19;
                                yield;
                            };
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"J";
                        %22 : java.type:"java.lang.String" = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %23 : java.type:"boolean" = constant @true;
                        yield %23;
                    }
                    ()java.type:"void" -> {
                        %24 : java.type:"java.lang.MatchException" = new @java.ref:"java.lang.MatchException::()";
                        throw %24;
                    };
                %25 : java.type:"java.lang.String" = var.load %3;
                return %25;
            };
            """)
    @CodeReflection
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
                %5 : java.type:"java.lang.Object" = constant @null;
                %6 : Var<java.type:"java.lang.Object"> = var %5 @"o";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %8 : java.type:"java.lang.String" = constant @"A";
                        %9 : java.type:"boolean" = invoke %7 %8 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %9;
                    }
                    ()java.type:"void" -> {
                        %10 : java.type:"java.lang.String" = var.load %3;
                        %11 : java.type:"java.lang.String" = constant @"A";
                        %12 : java.type:"java.lang.String" = concat %10 %11;
                        var.store %3 %12;
                        yield;
                    }
                    (%13 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %14 : java.type:"boolean" = pattern.match %13
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                %15 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type @"o";
                                yield %15;
                            }
                            (%16 : java.type:"java.lang.Object")java.type:"void" -> {
                                var.store %6 %16;
                                yield;
                            };
                        yield %14;
                    }
                    ()java.type:"void" -> {
                        %17 : java.type:"java.lang.String" = var.load %3;
                        %18 : java.type:"java.lang.String" = constant @"obj";
                        %19 : java.type:"java.lang.String" = concat %17 %18;
                        var.store %3 %19;
                        yield;
                    };
                %20 : java.type:"java.lang.String" = var.load %3;
                return %20;
            };
            """)
    @CodeReflection
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
                %5 : java.type:"java.lang.Integer" = constant @null;
                %6 : Var<java.type:"java.lang.Integer"> = var %5 @"i";
                %7 : java.type:"java.lang.String" = constant @null;
                %8 : Var<java.type:"java.lang.String"> = var %7 @"s";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"java.lang.String" = constant @"integer";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = pattern.match %16
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %18;
                            }
                            (%19 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %8 %19;
                                yield;
                            };
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"string";
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
    @CodeReflection
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
                %5 : java.type:"java.lang.Integer" = constant @null;
                %6 : Var<java.type:"java.lang.Integer"> = var %5 @"i";
                %7 : java.type:"java.lang.String" = constant @null;
                %8 : Var<java.type:"java.lang.String"> = var %7 @"s";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"java.lang.String" = constant @"integer";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = pattern.match %16
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %18;
                            }
                            (%19 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %8 %19;
                                yield;
                            };
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"string";
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
    @CodeReflection
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
                %5 : java.type:"java.lang.Integer" = constant @null;
                %6 : Var<java.type:"java.lang.Integer"> = var %5 @"i";
                %7 : java.type:"java.lang.String" = constant @null;
                %8 : Var<java.type:"java.lang.String"> = var %7 @"s";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Integer")java.type:"void" -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"java.lang.String" = constant @"integer";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        java.break;
                    }
                    (%16 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = pattern.match %16
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %18 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %18;
                            }
                            (%19 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %8 %19;
                                yield;
                            };
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %20 : java.type:"java.lang.String" = var.load %3;
                        %21 : java.type:"java.lang.String" = constant @"string";
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
    @CodeReflection
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
                %5 : java.type:"java.lang.Number" = constant @null;
                %6 : Var<java.type:"java.lang.Number"> = var %5 @"n";
                %7 : java.type:"java.lang.String" = constant @null;
                %8 : Var<java.type:"java.lang.String"> = var %7 @"s";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = pattern.match %9
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                %11 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                yield %11;
                            }
                            (%12 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %13;
                    }
                    (%14 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %15 : java.type:"boolean" = pattern.match %14
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %16 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                yield %16;
                            }
                            (%17 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %8 %17;
                                yield;
                            };
                        yield %15;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"a string";
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
                        %23 : java.type:"java.lang.Object" = var.load %1;
                        %24 : java.type:"java.lang.Class<?>" = invoke %23 @java.ref:"java.lang.Object::getClass():java.lang.Class";
                        %25 : java.type:"java.lang.String" = invoke %24 @java.ref:"java.lang.Class::getName():java.lang.String";
                        %26 : java.type:"java.lang.String" = concat %22 %25;
                        var.store %3 %26;
                        yield;
                    };
                %27 : java.type:"java.lang.String" = var.load %3;
                return %27;
            };
            """)
    @CodeReflection
    private static String casePatternThrow(Object o) {
        String r = "";
        switch (o) {
            case Number n -> throw new IllegalArgumentException();
            case String s -> r += "a string";
            default -> r += o.getClass().getName();
        }
        return r;
    }

    // @@@ code model for such as code is not supported
//    @CodeReflection
    private static String casePatternMultiLabel(Object o) {
        String r = "";
        switch (o) {
            case Integer _, Long _, Character _, Byte _, Short _-> r += "integral type";
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
                %5 : java.type:"java.lang.Integer" = constant @null;
                %6 : Var<java.type:"java.lang.Integer"> = var %5 @"i";
                %7 : java.type:"java.lang.Integer" = constant @null;
                %8 : Var<java.type:"java.lang.Integer"> = var %7 @"i";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %10 : java.type:"int" = constant @42;
                        %11 : java.type:"java.lang.Integer" = invoke %10 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %12 : java.type:"boolean" = invoke %9 %11 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %12;
                    }
                    ()java.type:"void" -> {
                        %13 : java.type:"java.lang.String" = var.load %3;
                        %14 : java.type:"java.lang.String" = constant @"forty two";
                        %15 : java.type:"java.lang.String" = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %17 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %18 : java.type:"boolean" = pattern.match %16
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %19 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                        yield %19;
                                    }
                                    (%20 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %6 %20;
                                        yield;
                                    };
                                yield %18;
                            }
                            ()java.type:"boolean" -> {
                                %21 : java.type:"java.lang.Integer" = var.load %6;
                                %22 : java.type:"int" = invoke %21 @java.ref:"java.lang.Integer::intValue():int";
                                %23 : java.type:"int" = constant @0;
                                %24 : java.type:"boolean" = gt %22 %23;
                                yield %24;
                            };
                        yield %17;
                    }
                    ()java.type:"void" -> {
                        %25 : java.type:"java.lang.String" = var.load %3;
                        %26 : java.type:"java.lang.String" = constant @"positive int";
                        %27 : java.type:"java.lang.String" = concat %25 %26;
                        var.store %3 %27;
                        yield;
                    }
                    (%28 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %29 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %30 : java.type:"boolean" = pattern.match %28
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" -> {
                                        %31 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                                        yield %31;
                                    }
                                    (%32 : java.type:"java.lang.Integer")java.type:"void" -> {
                                        var.store %8 %32;
                                        yield;
                                    };
                                yield %30;
                            }
                            ()java.type:"boolean" -> {
                                %33 : java.type:"java.lang.Integer" = var.load %8;
                                %34 : java.type:"int" = invoke %33 @java.ref:"java.lang.Integer::intValue():int";
                                %35 : java.type:"int" = constant @0;
                                %36 : java.type:"boolean" = lt %34 %35;
                                yield %36;
                            };
                        yield %29;
                    }
                    ()java.type:"void" -> {
                        %37 : java.type:"java.lang.String" = var.load %3;
                        %38 : java.type:"java.lang.String" = constant @"negative int";
                        %39 : java.type:"java.lang.String" = concat %37 %38;
                        var.store %3 %39;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %40 : java.type:"boolean" = constant @true;
                        yield %40;
                    }
                    ()java.type:"void" -> {
                        %41 : java.type:"java.lang.String" = var.load %3;
                        %42 : java.type:"java.lang.String" = constant @"zero";
                        %43 : java.type:"java.lang.String" = concat %41 %42;
                        var.store %3 %43;
                        yield;
                    };
                %44 : java.type:"java.lang.String" = var.load %3;
                return %44;
            };
            """)
    @CodeReflection
    static String casePatternWithCaseConstant(Integer a) {
        String r = "";
        switch (a) {
            case 42 -> r += "forty two";
            // @@@ case int will not match, because of the way InstanceOfOp is interpreted
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
                %5 : java.type:"java.lang.String" = constant @null;
                %6 : Var<java.type:"java.lang.String"> = var %5;
                %7 : java.type:"java.util.RandomAccess" = constant @null;
                %8 : Var<java.type:"java.util.RandomAccess"> = var %7;
                %9 : java.type:"int[]" = constant @null;
                %10 : Var<java.type:"int[]"> = var %9;
                %11 : java.type:"java.util.Stack[][]" = constant @null;
                %12 : Var<java.type:"java.util.Stack[][]"> = var %11;
                %13 : java.type:"java.util.Collection[][][]" = constant @null;
                %14 : Var<java.type:"java.util.Collection[][][]"> = var %13;
                %15 : java.type:"java.lang.Number" = constant @null;
                %16 : Var<java.type:"java.lang.Number"> = var %15 @"n";
                java.switch.statement %4
                    (%17 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %18 : java.type:"boolean" = pattern.match %17
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                %19 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type;
                                yield %19;
                            }
                            (%20 : java.type:"java.lang.String")java.type:"void" -> {
                                var.store %6 %20;
                                yield;
                            };
                        yield %18;
                    }
                    ()java.type:"void" -> {
                        %21 : java.type:"java.lang.String" = var.load %3;
                        %22 : java.type:"java.lang.String" = constant @"String";
                        %23 : java.type:"java.lang.String" = concat %21 %22;
                        var.store %3 %23;
                        yield;
                    }
                    (%24 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %25 : java.type:"boolean" = pattern.match %24
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.RandomAccess>" -> {
                                %26 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.RandomAccess>" = pattern.type;
                                yield %26;
                            }
                            (%27 : java.type:"java.util.RandomAccess")java.type:"void" -> {
                                var.store %8 %27;
                                yield;
                            };
                        yield %25;
                    }
                    ()java.type:"void" -> {
                        %28 : java.type:"java.lang.String" = var.load %3;
                        %29 : java.type:"java.lang.String" = constant @"RandomAccess";
                        %30 : java.type:"java.lang.String" = concat %28 %29;
                        var.store %3 %30;
                        yield;
                    }
                    (%31 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %32 : java.type:"boolean" = pattern.match %31
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int[]>" -> {
                                %33 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<int[]>" = pattern.type;
                                yield %33;
                            }
                            (%34 : java.type:"int[]")java.type:"void" -> {
                                var.store %10 %34;
                                yield;
                            };
                        yield %32;
                    }
                    ()java.type:"void" -> {
                        %35 : java.type:"java.lang.String" = var.load %3;
                        %36 : java.type:"java.lang.String" = constant @"int[]";
                        %37 : java.type:"java.lang.String" = concat %35 %36;
                        var.store %3 %37;
                        yield;
                    }
                    (%38 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %39 : java.type:"boolean" = pattern.match %38
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Stack[][]>" -> {
                                %40 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Stack[][]>" = pattern.type;
                                yield %40;
                            }
                            (%41 : java.type:"java.util.Stack[][]")java.type:"void" -> {
                                var.store %12 %41;
                                yield;
                            };
                        yield %39;
                    }
                    ()java.type:"void" -> {
                        %42 : java.type:"java.lang.String" = var.load %3;
                        %43 : java.type:"java.lang.String" = constant @"Stack[][]";
                        %44 : java.type:"java.lang.String" = concat %42 %43;
                        var.store %3 %44;
                        yield;
                    }
                    (%45 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %46 : java.type:"boolean" = pattern.match %45
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Collection[][][]>" -> {
                                %47 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.util.Collection[][][]>" = pattern.type;
                                yield %47;
                            }
                            (%48 : java.type:"java.util.Collection[][][]")java.type:"void" -> {
                                var.store %14 %48;
                                yield;
                            };
                        yield %46;
                    }
                    ()java.type:"void" -> {
                        %49 : java.type:"java.lang.String" = var.load %3;
                        %50 : java.type:"java.lang.String" = constant @"Collection[][][]";
                        %51 : java.type:"java.lang.String" = concat %49 %50;
                        var.store %3 %51;
                        yield;
                    }
                    (%52 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %53 : java.type:"boolean" = pattern.match %52
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" -> {
                                %54 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                yield %54;
                            }
                            (%55 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %16 %55;
                                yield;
                            };
                        yield %53;
                    }
                    ()java.type:"void" -> {
                        %56 : java.type:"java.lang.String" = var.load %3;
                        %57 : java.type:"java.lang.String" = constant @"Number";
                        %58 : java.type:"java.lang.String" = concat %56 %57;
                        var.store %3 %58;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %59 : java.type:"boolean" = constant @true;
                        yield %59;
                    }
                    ()java.type:"void" -> {
                        %60 : java.type:"java.lang.String" = var.load %3;
                        %61 : java.type:"java.lang.String" = constant @"something else";
                        %62 : java.type:"java.lang.String" = concat %60 %61;
                        var.store %3 %62;
                        yield;
                    };
                %63 : java.type:"java.lang.String" = var.load %3;
                return %63;
            };
            """)
    @CodeReflection
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
                %5 : java.type:"java.lang.Number" = constant @null;
                %6 : Var<java.type:"java.lang.Number"> = var %5 @"n";
                java.switch.statement %4
                    (%7 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %8 : java.type:"boolean" = pattern.match %7
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" -> {
                                %9 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" = pattern.record %9 @java.ref:"(java.lang.Number n)SwitchStatementTest$R";
                                yield %10;
                            }
                            (%11 : java.type:"java.lang.Number")java.type:"void" -> {
                                var.store %6 %11;
                                yield;
                            };
                        yield %8;
                    }
                    ()java.type:"void" -> {
                        %12 : java.type:"java.lang.String" = var.load %3;
                        %13 : java.type:"java.lang.String" = constant @"R(_)";
                        %14 : java.type:"java.lang.String" = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    }
                    ()java.type:"boolean" -> {
                        %15 : java.type:"boolean" = constant @true;
                        yield %15;
                    }
                    ()java.type:"void" -> {
                        %16 : java.type:"java.lang.String" = var.load %3;
                        %17 : java.type:"java.lang.String" = constant @"else";
                        %18 : java.type:"java.lang.String" = concat %16 %17;
                        var.store %3 %18;
                        yield;
                    };
                %19 : java.type:"java.lang.String" = var.load %3;
                return %19;
            };
            """)
    @CodeReflection
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
                %5 : java.type:"java.lang.String" = constant @null;
                %6 : Var<java.type:"java.lang.String"> = var %5 @"s";
                %7 : java.type:"java.lang.Number" = constant @null;
                %8 : Var<java.type:"java.lang.Number"> = var %7 @"n";
                java.switch.statement %4
                    (%9 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %10 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %11 : java.type:"boolean" = pattern.match %9
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" -> {
                                        %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.String>" = pattern.type @"s";
                                        yield %12;
                                    }
                                    (%13 : java.type:"java.lang.String")java.type:"void" -> {
                                        var.store %6 %13;
                                        yield;
                                    };
                                yield %11;
                            }
                            ()java.type:"boolean" -> {
                                %14 : java.type:"java.lang.String" = var.load %6;
                                %15 : java.type:"int" = invoke %14 @java.ref:"java.lang.String::length():int";
                                %16 : java.type:"int" = constant @3;
                                %17 : java.type:"boolean" = gt %15 %16;
                                yield %17;
                            };
                        yield %10;
                    }
                    ()java.type:"void" -> {
                        %18 : java.type:"java.lang.String" = var.load %3;
                        %19 : java.type:"java.lang.String" = constant @"str with length > %d";
                        %20 : java.type:"java.lang.String" = var.load %6;
                        %21 : java.type:"int" = invoke %20 @java.ref:"java.lang.String::length():int";
                        %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %23 : java.type:"java.lang.String" = invoke %19 %22 @java.ref:"java.lang.String::formatted(java.lang.Object[]):java.lang.String" @invoke.kind="INSTANCE" @invoke.varargs=true;
                        %24 : java.type:"java.lang.String" = concat %18 %23;
                        var.store %3 %24;
                        yield;
                    }
                    (%25 : java.type:"java.lang.Object")java.type:"boolean" -> {
                        %26 : java.type:"boolean" = java.cand
                            ()java.type:"boolean" -> {
                                %27 : java.type:"boolean" = pattern.match %25
                                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" -> {
                                        %28 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Number>" = pattern.type @"n";
                                        %29 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<SwitchStatementTest$R>" = pattern.record %28 @java.ref:"(java.lang.Number n)SwitchStatementTest$R";
                                        yield %29;
                                    }
                                    (%30 : java.type:"java.lang.Number")java.type:"void" -> {
                                        var.store %8 %30;
                                        yield;
                                    };
                                yield %27;
                            }
                            ()java.type:"boolean" -> {
                                %31 : java.type:"java.lang.Number" = var.load %8;
                                %32 : java.type:"java.lang.Class<?>" = invoke %31 @java.ref:"java.lang.Object::getClass():java.lang.Class";
                                %33 : java.type:"java.lang.Class" = constant @java.type:"java.lang.Double";
                                %34 : java.type:"boolean" = invoke %32 %33 @java.ref:"java.lang.Object::equals(java.lang.Object):boolean";
                                yield %34;
                            };
                        yield %26;
                    }
                    ()java.type:"void" -> {
                        %35 : java.type:"java.lang.String" = var.load %3;
                        %36 : java.type:"java.lang.String" = constant @"R(Double)";
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
                        %40 : java.type:"java.lang.String" = constant @"else";
                        %41 : java.type:"java.lang.String" = concat %39 %40;
                        var.store %3 %41;
                        yield;
                    };
                %42 : java.type:"java.lang.String" = var.load %3;
                return %42;
            };
            """)
    @CodeReflection
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
                    }
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
                    };
                %21 : java.type:"java.lang.String" = var.load %3;
                return %21;
            };
            """)
    @CodeReflection
    static String defaultCaseNotTheLast(String s) {
        String r = "";
        switch (s) {
            default -> r += "else";
            case "M" -> r += "Mow";
            case "A" -> r += "Aow";
        }
        return r;
    }
}
