import jdk.incubator.code.CodeReflection;

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
    @CodeReflection
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
    @CodeReflection
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
    @CodeReflection
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
    @CodeReflection
    private static String caseConstantMultiLabels(char c) {
        return switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u': yield "vowel";
            default: yield "consonant";
        };
    }

    @IR("""
            func @"caseConstantThrow" (%0 : java.type:"java.lang.Integer")java.type:"java.lang.String" -> {
                %1 : Var<java.type:"java.lang.Integer"> = var %0 @"i";
                %2 : java.type:"java.lang.Integer" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %5 : java.type:"int" = constant @8;
                        %6 : java.type:"java.lang.Integer" = invoke %5 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %7 : java.type:"boolean" = invoke %4 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %7;
                    }
                    ()java.type:"java.lang.String" -> {
                        %8 : java.type:"java.lang.IllegalArgumentException" = new @java.ref:"java.lang.IllegalArgumentException::()";
                        throw %8;
                    }
                    (%9 : java.type:"java.lang.Integer")java.type:"boolean" -> {
                        %10 : java.type:"int" = constant @9;
                        %11 : java.type:"java.lang.Integer" = invoke %10 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        %12 : java.type:"boolean" = invoke %9 %11 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
    @CodeReflection
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
                %3 : java.type:"java.lang.String" = java.switch.expression %2
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
    @CodeReflection
    private static String caseConstantNullLabel(String s) {
        return switch (s) {
            case null -> "null";
            default -> "non null";
        };
    }

    // @CodeReflection
    // compiler code doesn't support case null, default
    // @@@ support such as case and test the switch expression lowering for this case
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
    @CodeReflection
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
                                %7 : java.type:"boolean" = invoke %4 %6 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %7;
                            }
                            ()java.type:"boolean" -> {
                                %8 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::FRI:SwitchExpressionTest2$Day";
                                %9 : java.type:"boolean" = invoke %4 %8 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %9;
                            }
                            ()java.type:"boolean" -> {
                                %10 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::SUN:SwitchExpressionTest2$Day";
                                %11 : java.type:"boolean" = invoke %4 %10 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %15 : java.type:"boolean" = invoke %13 %14 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                                %20 : java.type:"boolean" = invoke %17 %19 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                                yield %20;
                            }
                            ()java.type:"boolean" -> {
                                %21 : java.type:"SwitchExpressionTest2$Day" = field.load @java.ref:"SwitchExpressionTest2$Day::SAT:SwitchExpressionTest2$Day";
                                %22 : java.type:"boolean" = invoke %17 %21 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %26 : java.type:"boolean" = invoke %24 %25 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %29 : java.type:"java.lang.MatchException" = new @java.ref:"java.lang.MatchException::()";
                        throw %29;
                    };
                return %3;
            };
            """)
    @CodeReflection
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
                %3 : Var<java.type:"int"> = var %2 @"eleven";
                %4 : java.type:"int" = var.load %1;
                %5 : java.type:"java.lang.String" = java.switch.expression %4
                    (%6 : java.type:"int")java.type:"boolean" -> {
                        %7 : java.type:"int" = constant @1;
                        %8 : java.type:"int" = constant @15;
                        %9 : java.type:"int" = and %7 %8;
                        %10 : java.type:"boolean" = eq %6 %9;
                        yield %10;
                    }
                    ()java.type:"java.lang.String" -> {
                        %11 : java.type:"java.lang.String" = constant @"1";
                        yield %11;
                    }
                    (%12 : java.type:"int")java.type:"boolean" -> {
                        %13 : java.type:"int" = constant @4;
                        %14 : java.type:"int" = constant @1;
                        %15 : java.type:"int" = ashr %13 %14;
                        %16 : java.type:"boolean" = eq %12 %15;
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"2";
                        yield %17;
                    }
                    (%18 : java.type:"int")java.type:"boolean" -> {
                        %19 : java.type:"long" = constant @3;
                        %20 : java.type:"int" = conv %19;
                        %21 : java.type:"boolean" = eq %18 %20;
                        yield %21;
                    }
                    ()java.type:"java.lang.String" -> {
                        %22 : java.type:"java.lang.String" = constant @"3";
                        yield %22;
                    }
                    (%23 : java.type:"int")java.type:"boolean" -> {
                        %24 : java.type:"int" = constant @2;
                        %25 : java.type:"int" = constant @1;
                        %26 : java.type:"int" = lshl %24 %25;
                        %27 : java.type:"boolean" = eq %23 %26;
                        yield %27;
                    }
                    ()java.type:"java.lang.String" -> {
                        %28 : java.type:"java.lang.String" = constant @"4";
                        yield %28;
                    }
                    (%29 : java.type:"int")java.type:"boolean" -> {
                        %30 : java.type:"int" = constant @10;
                        %31 : java.type:"int" = constant @2;
                        %32 : java.type:"int" = div %30 %31;
                        %33 : java.type:"boolean" = eq %29 %32;
                        yield %33;
                    }
                    ()java.type:"java.lang.String" -> {
                        %34 : java.type:"java.lang.String" = constant @"5";
                        yield %34;
                    }
                    (%35 : java.type:"int")java.type:"boolean" -> {
                        %36 : java.type:"int" = constant @12;
                        %37 : java.type:"int" = constant @6;
                        %38 : java.type:"int" = sub %36 %37;
                        %39 : java.type:"boolean" = eq %35 %38;
                        yield %39;
                    }
                    ()java.type:"java.lang.String" -> {
                        %40 : java.type:"java.lang.String" = constant @"6";
                        yield %40;
                    }
                    (%41 : java.type:"int")java.type:"boolean" -> {
                        %42 : java.type:"int" = constant @3;
                        %43 : java.type:"int" = constant @4;
                        %44 : java.type:"int" = add %42 %43;
                        %45 : java.type:"boolean" = eq %41 %44;
                        yield %45;
                    }
                    ()java.type:"java.lang.String" -> {
                        %46 : java.type:"java.lang.String" = constant @"7";
                        yield %46;
                    }
                    (%47 : java.type:"int")java.type:"boolean" -> {
                        %48 : java.type:"int" = constant @2;
                        %49 : java.type:"int" = constant @2;
                        %50 : java.type:"int" = mul %48 %49;
                        %51 : java.type:"int" = constant @2;
                        %52 : java.type:"int" = mul %50 %51;
                        %53 : java.type:"boolean" = eq %47 %52;
                        yield %53;
                    }
                    ()java.type:"java.lang.String" -> {
                        %54 : java.type:"java.lang.String" = constant @"8";
                        yield %54;
                    }
                    (%55 : java.type:"int")java.type:"boolean" -> {
                        %56 : java.type:"int" = constant @8;
                        %57 : java.type:"int" = constant @1;
                        %58 : java.type:"int" = or %56 %57;
                        %59 : java.type:"boolean" = eq %55 %58;
                        yield %59;
                    }
                    ()java.type:"java.lang.String" -> {
                        %60 : java.type:"java.lang.String" = constant @"9";
                        yield %60;
                    }
                    (%61 : java.type:"int")java.type:"boolean" -> {
                        %62 : java.type:"int" = constant @10;
                        %63 : java.type:"boolean" = eq %61 %62;
                        yield %63;
                    }
                    ()java.type:"java.lang.String" -> {
                        %64 : java.type:"java.lang.String" = constant @"10";
                        yield %64;
                    }
                    (%65 : java.type:"int")java.type:"boolean" -> {
                        %66 : java.type:"int" = var.load %3;
                        %67 : java.type:"boolean" = eq %65 %66;
                        yield %67;
                    }
                    ()java.type:"java.lang.String" -> {
                        %68 : java.type:"java.lang.String" = constant @"11";
                        yield %68;
                    }
                    (%69 : java.type:"int")java.type:"boolean" -> {
                        %70 : java.type:"int" = field.load @java.ref:"SwitchExpressionTest2$Constants::c1:int";
                        %71 : java.type:"boolean" = eq %69 %70;
                        yield %71;
                    }
                    ()java.type:"java.lang.String" -> {
                        %72 : java.type:"int" = field.load @java.ref:"SwitchExpressionTest2$Constants::c1:int";
                        %73 : java.type:"java.lang.String" = invoke %72 @java.ref:"java.lang.String::valueOf(int):java.lang.String";
                        yield %73;
                    }
                    (%74 : java.type:"int")java.type:"boolean" -> {
                        %75 : java.type:"int" = java.cexpression
                            ()java.type:"boolean" -> {
                                %76 : java.type:"int" = constant @1;
                                %77 : java.type:"int" = constant @0;
                                %78 : java.type:"boolean" = gt %76 %77;
                                yield %78;
                            }
                            ()java.type:"int" -> {
                                %79 : java.type:"int" = constant @13;
                                yield %79;
                            }
                            ()java.type:"int" -> {
                                %80 : java.type:"int" = constant @133;
                                yield %80;
                            };
                        %81 : java.type:"boolean" = eq %74 %75;
                        yield %81;
                    }
                    ()java.type:"java.lang.String" -> {
                        %82 : java.type:"java.lang.String" = constant @"13";
                        yield %82;
                    }
                    ()java.type:"boolean" -> {
                        %83 : java.type:"boolean" = constant @true;
                        yield %83;
                    }
                    ()java.type:"java.lang.String" -> {
                        %84 : java.type:"java.lang.String" = constant @"an int";
                        yield %84;
                    };
                return %5;
            };
            """)
    @CodeReflection
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
                %4 : Var<java.type:"short"> = var %3 @"s";
                %5 : java.type:"int" = constant @2;
                %6 : java.type:"byte" = conv %5;
                %7 : Var<java.type:"byte"> = var %6 @"b";
                %8 : java.type:"short" = var.load %1;
                %9 : java.type:"java.lang.String" = java.switch.expression %8
                    (%10 : java.type:"short")java.type:"boolean" -> {
                        %11 : java.type:"short" = var.load %4;
                        %12 : java.type:"boolean" = eq %10 %11;
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @"one";
                        yield %13;
                    }
                    (%14 : java.type:"short")java.type:"boolean" -> {
                        %15 : java.type:"byte" = var.load %7;
                        %16 : java.type:"short" = conv %15;
                        %17 : java.type:"boolean" = eq %14 %16;
                        yield %17;
                    }
                    ()java.type:"java.lang.String" -> {
                        %18 : java.type:"java.lang.String" = constant @"three";
                        yield %18;
                    }
                    (%19 : java.type:"short")java.type:"boolean" -> {
                        %20 : java.type:"int" = constant @3;
                        %21 : java.type:"short" = conv %20;
                        %22 : java.type:"boolean" = eq %19 %21;
                        yield %22;
                    }
                    ()java.type:"java.lang.String" -> {
                        %23 : java.type:"java.lang.String" = constant @"two";
                        yield %23;
                    }
                    ()java.type:"boolean" -> {
                        %24 : java.type:"boolean" = constant @true;
                        yield %24;
                    }
                    ()java.type:"java.lang.String" -> {
                        %25 : java.type:"java.lang.String" = constant @"default";
                        yield %25;
                    };
                return %9;
            };
            """)
    @CodeReflection
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
                %4 : Var<java.type:"byte"> = var %3 @"b";
                %5 : java.type:"java.lang.Byte" = var.load %1;
                %6 : java.type:"java.lang.String" = java.switch.expression %5
                    (%7 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %8 : java.type:"int" = constant @1;
                        %9 : java.type:"byte" = conv %8;
                        %10 : java.type:"java.lang.Byte" = invoke %9 @java.ref:"java.lang.Byte::valueOf(byte):java.lang.Byte";
                        %11 : java.type:"boolean" = invoke %7 %10 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %11;
                    }
                    ()java.type:"java.lang.String" -> {
                        %12 : java.type:"java.lang.String" = constant @"one";
                        yield %12;
                    }
                    (%13 : java.type:"java.lang.Byte")java.type:"boolean" -> {
                        %14 : java.type:"byte" = var.load %4;
                        %15 : java.type:"java.lang.Byte" = invoke %14 @java.ref:"java.lang.Byte::valueOf(byte):java.lang.Byte";
                        %16 : java.type:"boolean" = invoke %13 %15 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %16;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"two";
                        yield %17;
                    }
                    ()java.type:"boolean" -> {
                        %18 : java.type:"boolean" = constant @true;
                        yield %18;
                    }
                    ()java.type:"java.lang.String" -> {
                        %19 : java.type:"java.lang.String" = constant @"default";
                        yield %19;
                    };
                return %6;
            };
            """)
    @CodeReflection
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
                        %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        %7 : java.type:"java.lang.String" = constant @"f";
                        yield %7;
                    }
                    (%8 : java.type:"SwitchExpressionTest2$E")java.type:"boolean" -> {
                        %9 : java.type:"SwitchExpressionTest2$E" = field.load @java.ref:"SwitchExpressionTest2$E::G:SwitchExpressionTest2$E";
                        %10 : java.type:"boolean" = invoke %8 %9 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
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
                        %13 : java.type:"java.lang.MatchException" = new @java.ref:"java.lang.MatchException::()";
                        throw %13;
                    };
                return %3;
            };
            """)
    @CodeReflection
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
                %3 : java.type:"java.lang.Object" = constant @null;
                %4 : Var<java.type:"java.lang.Object"> = var %3 @"o";
                %5 : java.type:"java.lang.String" = java.switch.expression %2
                    (%6 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %7 : java.type:"java.lang.String" = constant @"A";
                        %8 : java.type:"boolean" = invoke %6 %7 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"Alphabet";
                        yield %9;
                    }
                    (%10 : java.type:"java.lang.String")java.type:"boolean" -> {
                        %11 : java.type:"boolean" = pattern.match %10
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" -> {
                                %12 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Object>" = pattern.type @"o";
                                yield %12;
                            }
                            (%13 : java.type:"java.lang.Object")java.type:"void" -> {
                                var.store %4 %13;
                                yield;
                            };
                        yield %11;
                    }
                    ()java.type:"java.lang.String" -> {
                        %14 : java.type:"java.lang.String" = constant @"default";
                        yield %14;
                    };
                return %5;
            };
            """)
    @CodeReflection
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
                %3 : java.type:"SwitchExpressionTest2$B" = constant @null;
                %4 : Var<java.type:"SwitchExpressionTest2$B"> = var %3 @"b";
                %5 : java.type:"SwitchExpressionTest2::C" = constant @null;
                %6 : Var<java.type:"SwitchExpressionTest2::C"> = var %5 @"c";
                %7 : java.type:"java.lang.String" = java.switch.expression %2
                    (%8 : java.type:"SwitchExpressionTest2$A")java.type:"boolean" -> {
                        %9 : java.type:"boolean" = pattern.match %8
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2$B>" -> {
                                %10 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2$B>" = pattern.type @"b";
                                yield %10;
                            }
                            (%11 : java.type:"SwitchExpressionTest2$B")java.type:"void" -> {
                                var.store %4 %11;
                                yield;
                            };
                        yield %9;
                    }
                    ()java.type:"java.lang.String" -> {
                        %12 : java.type:"java.lang.String" = constant @"B";
                        yield %12;
                    }
                    (%13 : java.type:"SwitchExpressionTest2$A")java.type:"boolean" -> {
                        %14 : java.type:"boolean" = pattern.match %13
                            ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2::C>" -> {
                                %15 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<SwitchExpressionTest2::C>" = pattern.type @"c";
                                yield %15;
                            }
                            (%16 : java.type:"SwitchExpressionTest2::C")java.type:"void" -> {
                                var.store %6 %16;
                                yield;
                            };
                        yield %14;
                    }
                    ()java.type:"java.lang.String" -> {
                        %17 : java.type:"java.lang.String" = constant @"C";
                        yield %17;
                    }
                    ()java.type:"boolean" -> {
                        %18 : java.type:"boolean" = constant @true;
                        yield %18;
                    }
                    ()java.type:"java.lang.String" -> {
                        %19 : java.type:"java.lang.MatchException" = new @java.ref:"java.lang.MatchException::()";
                        throw %19;
                    };
                return %7;
            };
            """)
    @CodeReflection
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
                    }
                    ()java.type:"boolean" -> {
                        %12 : java.type:"boolean" = constant @true;
                        yield %12;
                    }
                    ()java.type:"java.lang.String" -> {
                        %13 : java.type:"java.lang.String" = constant @"else";
                        yield %13;
                    };
                return %3;
            };
            """)
    @CodeReflection
    static String defaultNotTheLastLabel(String s) {
        return switch (s) {
            default -> "else";
            case "M" -> "Mow";
            case "A" -> "Aow";
        };
    }
}
