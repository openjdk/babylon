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
            func @"caseConstantRuleExpression" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %7 : java.lang.String = constant @"BAR";
                        yield %7;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"BAZ";
                        yield %11;
                    }
                    (%12 : java.lang.String)boolean -> {
                        %13 : java.lang.String = constant @"BAZ";
                        %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %14;
                    }
                    ()java.lang.String -> {
                        %15 : java.lang.String = constant @"FOO";
                        yield %15;
                    }
                    ()boolean -> {
                        %17 : boolean = constant @"true";
                        yield %17;
                    }
                    ()java.lang.String -> {
                        %16 : java.lang.String = constant @"";
                        yield %16;
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
            func @"caseConstantRuleBlock" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %7 : java.lang.String = constant @"BAR";
                        java.yield %7;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"BAZ";
                        java.yield %11;
                    }
                    (%12 : java.lang.String)boolean -> {
                        %13 : java.lang.String = constant @"BAZ";
                        %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %14;
                    }
                    ()java.lang.String -> {
                        %15 : java.lang.String = constant @"FOO";
                        java.yield %15;
                    }
                    ()boolean -> {
                        %17 : boolean = constant @"true";
                        yield %17;
                    }
                    ()java.lang.String -> {
                        %16 : java.lang.String = constant @"";
                        java.yield %16;
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
            func @"caseConstantStatement" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : java.lang.String)boolean -> {
                        %5 : java.lang.String = constant @"FOO";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %7 : java.lang.String = constant @"BAR";
                        java.yield %7;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"BAZ";
                        java.yield %11;
                    }
                    (%12 : java.lang.String)boolean -> {
                        %13 : java.lang.String = constant @"BAZ";
                        %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %14;
                    }
                    ()java.lang.String -> {
                        %15 : java.lang.String = constant @"FOO";
                        java.yield %15;
                    }
                    ()boolean -> {
                        %17 : boolean = constant @"true";
                        yield %17;
                    }
                    ()java.lang.String -> {
                        %16 : java.lang.String = constant @"";
                        java.yield %16;
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
            func @"caseConstantMultiLabels" (%0 : char)java.lang.String -> {
                %1 : Var<char> = var %0 @"c";
                %2 : char = var.load %1;
                %3 : char = invoke %2 @"java.lang.Character::toLowerCase(char)char";
                %4 : java.lang.String = java.switch.expression %3
                    (%5 : char)boolean -> {
                        %6 : boolean = java.cor
                            ()boolean -> {
                                %7 : char = constant @"a";
                                %8 : boolean = eq %5 %7;
                                yield %8;
                            }
                            ()boolean -> {
                                %9 : char = constant @"e";
                                %10 : boolean = eq %5 %9;
                                yield %10;
                            }
                            ()boolean -> {
                                %11 : char = constant @"i";
                                %12 : boolean = eq %5 %11;
                                yield %12;
                            }
                            ()boolean -> {
                                %13 : char = constant @"o";
                                %14 : boolean = eq %5 %13;
                                yield %14;
                            }
                            ()boolean -> {
                                %15 : char = constant @"u";
                                %16 : boolean = eq %5 %15;
                                yield %16;
                            };
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %17 : java.lang.String = constant @"vowel";
                        java.yield %17;
                    }
                    ()boolean -> {
                        %19 : boolean = constant @"true";
                        yield %19;
                    }
                    ()java.lang.String -> {
                        %18 : java.lang.String = constant @"consonant";
                        java.yield %18;
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
            func @"caseConstantThrow" (%0 : java.lang.Integer)java.lang.String -> {
                %1 : Var<java.lang.Integer> = var %0 @"i";
                %2 : java.lang.Integer = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : java.lang.Integer)boolean -> {
                        %5 : int = constant @"8";
                        %6 : java.lang.Integer = invoke %5 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %7 : boolean = invoke %4 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.IllegalArgumentException = new @"func<java.lang.IllegalArgumentException>";
                        throw %8;
                    }
                    (%9 : java.lang.Integer)boolean -> {
                        %10 : int = constant @"9";
                        %11 : java.lang.Integer = invoke %10 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %12 : boolean = invoke %9 %11 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %12;
                    }
                    ()java.lang.String -> {
                        %13 : java.lang.String = constant @"NINE";
                        yield %13;
                    }
                    ()boolean -> {
                        %15 : boolean = constant @"true";
                        yield %15;
                    }
                    ()java.lang.String -> {
                        %14 : java.lang.String = constant @"An integer";
                        yield %14;
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
            func @"caseConstantNullLabel" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : java.lang.String)boolean -> {
                        %5 : java.lang.Object = constant @null;
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %7 : java.lang.String = constant @"null";
                        yield %7;
                    }
                    ()boolean -> {
                        %9 : boolean = constant @"true";
                        yield %9;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @"non null";
                        yield %8;
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
            func @"caseConstantFallThrough" (%0 : char)java.lang.String -> {
                %1 : Var<char> = var %0 @"c";
                %2 : char = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : char)boolean -> {
                        %5 : char = constant @"A";
                        %6 : boolean = eq %4 %5;
                        yield %6;
                    }
                    ()java.lang.String -> {
                        java.switch.fallthrough;
                    }
                    (%7 : char)boolean -> {
                        %8 : char = constant @"B";
                        %9 : boolean = eq %7 %8;
                        yield %9;
                    }
                    ()java.lang.String -> {
                        %10 : java.lang.String = constant @"A or B";
                        java.yield %10;
                    }
                    ()boolean -> {
                        %12 : boolean = constant @"true";
                        yield %12;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"Neither A nor B";
                        java.yield %11;
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
            func @"caseConstantEnum" (%0 : SwitchExpressionTest2$Day)int -> {
                %1 : Var<SwitchExpressionTest2$Day> = var %0 @"d";
                %2 : SwitchExpressionTest2$Day = var.load %1;
                %3 : int = java.switch.expression %2
                    (%4 : SwitchExpressionTest2$Day)boolean -> {
                        %5 : boolean = java.cor
                            ()boolean -> {
                                %6 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::MON()SwitchExpressionTest2$Day";
                                %7 : boolean = invoke %4 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %7;
                            }
                            ()boolean -> {
                                %8 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::FRI()SwitchExpressionTest2$Day";
                                %9 : boolean = invoke %4 %8 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %9;
                            }
                            ()boolean -> {
                                %10 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::SUN()SwitchExpressionTest2$Day";
                                %11 : boolean = invoke %4 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %11;
                            };
                        yield %5;
                    }
                    ()int -> {
                        %12 : int = constant @"6";
                        yield %12;
                    }
                    (%13 : SwitchExpressionTest2$Day)boolean -> {
                        %14 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::TUE()SwitchExpressionTest2$Day";
                        %15 : boolean = invoke %13 %14 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %15;
                    }
                    ()int -> {
                        %16 : int = constant @"7";
                        yield %16;
                    }
                    (%17 : SwitchExpressionTest2$Day)boolean -> {
                        %18 : boolean = java.cor
                            ()boolean -> {
                                %19 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::THU()SwitchExpressionTest2$Day";
                                %20 : boolean = invoke %17 %19 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %20;
                            }
                            ()boolean -> {
                                %21 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::SAT()SwitchExpressionTest2$Day";
                                %22 : boolean = invoke %17 %21 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %22;
                            };
                        yield %18;
                    }
                    ()int -> {
                        %23 : int = constant @"8";
                        yield %23;
                    }
                    (%24 : SwitchExpressionTest2$Day)boolean -> {
                        %25 : SwitchExpressionTest2$Day = field.load @"SwitchExpressionTest2$Day::WED()SwitchExpressionTest2$Day";
                        %26 : boolean = invoke %24 %25 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %26;
                    }
                    ()int -> {
                        %27 : int = constant @"9";
                        yield %27;
                    }
                    ()boolean -> {
                        %29 : boolean = constant @"true";
                        yield %29;
                    }
                    ()int -> {
                        %28 : java.lang.MatchException = new @"func<java.lang.MatchException>";
                        throw %28;
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
            func @"caseConstantOtherKindsOfExpr" (%0 : int)java.lang.String -> {
                %1 : Var<int> = var %0 @"i";
                %2 : int = constant @"11";
                %3 : Var<int> = var %2 @"eleven";
                %4 : int = var.load %1;
                %5 : java.lang.String = java.switch.expression %4
                    (%6 : int)boolean -> {
                        %7 : int = constant @"1";
                        %8 : int = constant @"15";
                        %9 : int = and %7 %8;
                        %10 : boolean = eq %6 %9;
                        yield %10;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"1";
                        yield %11;
                    }
                    (%12 : int)boolean -> {
                        %13 : int = constant @"4";
                        %14 : int = constant @"1";
                        %15 : int = ashr %13 %14;
                        %16 : boolean = eq %12 %15;
                        yield %16;
                    }
                    ()java.lang.String -> {
                        %17 : java.lang.String = constant @"2";
                        yield %17;
                    }
                    (%18 : int)boolean -> {
                        %19 : long = constant @"3";
                        %20 : int = conv %19;
                        %21 : boolean = eq %18 %20;
                        yield %21;
                    }
                    ()java.lang.String -> {
                        %22 : java.lang.String = constant @"3";
                        yield %22;
                    }
                    (%23 : int)boolean -> {
                        %24 : int = constant @"2";
                        %25 : int = constant @"1";
                        %26 : int = lshl %24 %25;
                        %27 : boolean = eq %23 %26;
                        yield %27;
                    }
                    ()java.lang.String -> {
                        %28 : java.lang.String = constant @"4";
                        yield %28;
                    }
                    (%29 : int)boolean -> {
                        %30 : int = constant @"10";
                        %31 : int = constant @"2";
                        %32 : int = div %30 %31;
                        %33 : boolean = eq %29 %32;
                        yield %33;
                    }
                    ()java.lang.String -> {
                        %34 : java.lang.String = constant @"5";
                        yield %34;
                    }
                    (%35 : int)boolean -> {
                        %36 : int = constant @"12";
                        %37 : int = constant @"6";
                        %38 : int = sub %36 %37;
                        %39 : boolean = eq %35 %38;
                        yield %39;
                    }
                    ()java.lang.String -> {
                        %40 : java.lang.String = constant @"6";
                        yield %40;
                    }
                    (%41 : int)boolean -> {
                        %42 : int = constant @"3";
                        %43 : int = constant @"4";
                        %44 : int = add %42 %43;
                        %45 : boolean = eq %41 %44;
                        yield %45;
                    }
                    ()java.lang.String -> {
                        %46 : java.lang.String = constant @"7";
                        yield %46;
                    }
                    (%47 : int)boolean -> {
                        %48 : int = constant @"2";
                        %49 : int = constant @"2";
                        %50 : int = mul %48 %49;
                        %51 : int = constant @"2";
                        %52 : int = mul %50 %51;
                        %53 : boolean = eq %47 %52;
                        yield %53;
                    }
                    ()java.lang.String -> {
                        %54 : java.lang.String = constant @"8";
                        yield %54;
                    }
                    (%55 : int)boolean -> {
                        %56 : int = constant @"8";
                        %57 : int = constant @"1";
                        %58 : int = or %56 %57;
                        %59 : boolean = eq %55 %58;
                        yield %59;
                    }
                    ()java.lang.String -> {
                        %60 : java.lang.String = constant @"9";
                        yield %60;
                    }
                    (%61 : int)boolean -> {
                        %62 : int = constant @"10";
                        %63 : boolean = eq %61 %62;
                        yield %63;
                    }
                    ()java.lang.String -> {
                        %64 : java.lang.String = constant @"10";
                        yield %64;
                    }
                    (%65 : int)boolean -> {
                        %66 : int = var.load %3;
                        %67 : boolean = eq %65 %66;
                        yield %67;
                    }
                    ()java.lang.String -> {
                        %68 : java.lang.String = constant @"11";
                        yield %68;
                    }
                    (%69 : int)boolean -> {
                        %70 : int = field.load @"SwitchExpressionTest2$Constants::c1()int";
                        %71 : boolean = eq %69 %70;
                        yield %71;
                    }
                    ()java.lang.String -> {
                        %72 : int = field.load @"SwitchExpressionTest2$Constants::c1()int";
                        %73 : java.lang.String = invoke %72 @"java.lang.String::valueOf(int)java.lang.String";
                        yield %73;
                    }
                    (%74 : int)boolean -> {
                        %75 : int = java.cexpression
                            ()boolean -> {
                                %76 : int = constant @"1";
                                %77 : int = constant @"0";
                                %78 : boolean = gt %76 %77;
                                yield %78;
                            }
                            ()int -> {
                                %79 : int = constant @"13";
                                yield %79;
                            }
                            ()int -> {
                                %80 : int = constant @"133";
                                yield %80;
                            };
                        %81 : boolean = eq %74 %75;
                        yield %81;
                    }
                    ()java.lang.String -> {
                        %82 : java.lang.String = constant @"13";
                        yield %82;
                    }
                    ()boolean -> {
                        %84 : boolean = constant @"true";
                        yield %84;
                    }
                    ()java.lang.String -> {
                        %83 : java.lang.String = constant @"an int";
                        yield %83;
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
            func @"caseConstantConv" (%0 : short)java.lang.String -> {
                %1 : Var<short> = var %0 @"a";
                %2 : int = constant @"1";
                %3 : short = conv %2;
                %4 : Var<short> = var %3 @"s";
                %5 : int = constant @"2";
                %6 : byte = conv %5;
                %7 : Var<byte> = var %6 @"b";
                %8 : short = var.load %1;
                %9 : java.lang.String = java.switch.expression %8
                    (%10 : short)boolean -> {
                        %11 : short = var.load %4;
                        %12 : boolean = eq %10 %11;
                        yield %12;
                    }
                    ()java.lang.String -> {
                        %13 : java.lang.String = constant @"one";
                        yield %13;
                    }
                    (%14 : short)boolean -> {
                        %15 : byte = var.load %7;
                        %16 : short = conv %15;
                        %17 : boolean = eq %14 %16;
                        yield %17;
                    }
                    ()java.lang.String -> {
                        %18 : java.lang.String = constant @"three";
                        yield %18;
                    }
                    (%19 : short)boolean -> {
                        %20 : int = constant @"3";
                        %21 : short = conv %20;
                        %22 : boolean = eq %19 %21;
                        yield %22;
                    }
                    ()java.lang.String -> {
                        %23 : java.lang.String = constant @"two";
                        yield %23;
                    }
                    ()boolean -> {
                        %25 : boolean = constant @"true";
                        yield %25;
                    }
                    ()java.lang.String -> {
                        %24 : java.lang.String = constant @"default";
                        yield %24;
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
            func @"caseConstantConv2" (%0 : java.lang.Byte)java.lang.String -> {
                %1 : Var<java.lang.Byte> = var %0 @"a";
                %2 : int = constant @"2";
                %3 : byte = conv %2;
                %4 : Var<byte> = var %3 @"b";
                %5 : java.lang.Byte = var.load %1;
                %6 : java.lang.String = java.switch.expression %5
                    (%7 : java.lang.Byte)boolean -> {
                        %8 : int = constant @"1";
                        %9 : byte = conv %8;
                        %10 : java.lang.Byte = invoke %9 @"java.lang.Byte::valueOf(byte)java.lang.Byte";
                        %11 : boolean = invoke %7 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %11;
                    }
                    ()java.lang.String -> {
                        %12 : java.lang.String = constant @"one";
                        yield %12;
                    }
                    (%13 : java.lang.Byte)boolean -> {
                        %14 : byte = var.load %4;
                        %15 : java.lang.Byte = invoke %14 @"java.lang.Byte::valueOf(byte)java.lang.Byte";
                        %16 : boolean = invoke %13 %15 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %16;
                    }
                    ()java.lang.String -> {
                        %17 : java.lang.String = constant @"two";
                        yield %17;
                    }
                    ()boolean -> {
                        %19 : boolean = constant @"true";
                        yield %19;
                    }
                    ()java.lang.String -> {
                        %18 : java.lang.String = constant @"default";
                        yield %18;
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
            func @"noDefaultLabelEnum" (%0 : SwitchExpressionTest2$E)java.lang.String -> {
                %1 : Var<SwitchExpressionTest2$E> = var %0 @"e";
                %2 : SwitchExpressionTest2$E = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : SwitchExpressionTest2$E)boolean -> {
                        %5 : SwitchExpressionTest2$E = field.load @"SwitchExpressionTest2$E::F()SwitchExpressionTest2$E";
                        %6 : boolean = invoke %4 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()java.lang.String -> {
                        %7 : java.lang.String = constant @"f";
                        yield %7;
                    }
                    (%8 : SwitchExpressionTest2$E)boolean -> {
                        %9 : SwitchExpressionTest2$E = field.load @"SwitchExpressionTest2$E::G()SwitchExpressionTest2$E";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()java.lang.String -> {
                        %11 : java.lang.String = constant @"g";
                        yield %11;
                    }
                    ()boolean -> {
                        %13 : boolean = constant @"true";
                        yield %13;
                    }
                    ()java.lang.String -> {
                        %12 : java.lang.MatchException = new @"func<java.lang.MatchException>";
                        throw %12;
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
            func @"unconditionalPattern" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.Object = constant @null;
                %4 : Var<java.lang.Object> = var %3 @"o";
                %5 : java.lang.String = java.switch.expression %2
                    (%6 : java.lang.String)boolean -> {
                        %7 : java.lang.String = constant @"A";
                        %8 : boolean = invoke %6 %7 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %8;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @"Alphabet";
                        yield %9;
                    }
                    (%10 : java.lang.String)boolean -> {
                        %11 : boolean = pattern.match %10
                            ()jdk.incubator.code.op.ExtendedOp$Pattern$Type<java.lang.Object> -> {
                                %12 : jdk.incubator.code.op.ExtendedOp$Pattern$Type<java.lang.Object> = pattern.type @"o";
                                yield %12;
                            }
                            (%13 : java.lang.Object)void -> {
                                var.store %4 %13;
                                yield;
                            };
                        yield %11;
                    }
                    ()java.lang.String -> {
                        %14 : java.lang.String = constant @"default";
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
            func @"noDefault" (%0 : SwitchExpressionTest2$A)java.lang.String -> {
                %1 : Var<SwitchExpressionTest2$A> = var %0 @"a";
                %2 : SwitchExpressionTest2$A = var.load %1;
                %3 : SwitchExpressionTest2$B = constant @null;
                %4 : Var<SwitchExpressionTest2$B> = var %3 @"b";
                %5 : .<SwitchExpressionTest2, SwitchExpressionTest2$C> = constant @null;
                %6 : Var<.<SwitchExpressionTest2, SwitchExpressionTest2$C>> = var %5 @"c";
                %7 : java.lang.String = java.switch.expression %2
                    (%8 : SwitchExpressionTest2$A)boolean -> {
                        %9 : boolean = pattern.match %8
                            ()jdk.incubator.code.op.ExtendedOp$Pattern$Type<SwitchExpressionTest2$B> -> {
                                %10 : jdk.incubator.code.op.ExtendedOp$Pattern$Type<SwitchExpressionTest2$B> = pattern.type @"b";
                                yield %10;
                            }
                            (%11 : SwitchExpressionTest2$B)void -> {
                                var.store %4 %11;
                                yield;
                            };
                        yield %9;
                    }
                    ()java.lang.String -> {
                        %12 : java.lang.String = constant @"B";
                        yield %12;
                    }
                    (%13 : SwitchExpressionTest2$A)boolean -> {
                        %14 : boolean = pattern.match %13
                            ()jdk.incubator.code.op.ExtendedOp$Pattern$Type<.<SwitchExpressionTest2, SwitchExpressionTest2$C>> -> {
                                %15 : jdk.incubator.code.op.ExtendedOp$Pattern$Type<.<SwitchExpressionTest2, SwitchExpressionTest2$C>> = pattern.type @"c";
                                yield %15;
                            }
                            (%16 : .<SwitchExpressionTest2, SwitchExpressionTest2$C>)void -> {
                                var.store %6 %16;
                                yield;
                            };
                        yield %14;
                    }
                    ()java.lang.String -> {
                        %17 : java.lang.String = constant @"C";
                        yield %17;
                    }
                    ()boolean -> {
                        %19 : boolean = constant @"true";
                        yield %19;
                    }
                    ()java.lang.String -> {
                        %18 : java.lang.MatchException = new @"func<java.lang.MatchException>";
                        throw %18;
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
            func @"defaultNotTheLastLabel" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%5 : java.lang.String)boolean -> {
                        %6 : java.lang.String = constant @"M";
                        %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %7;
                    }
                    ()java.lang.String -> {
                        %8 : java.lang.String = constant @"Mow";
                        yield %8;
                    }
                    (%9 : java.lang.String)boolean -> {
                        %10 : java.lang.String = constant @"A";
                        %11 : boolean = invoke %9 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %11;
                    }
                    ()java.lang.String -> {
                        %12 : java.lang.String = constant @"Aow";
                        yield %12;
                    }
                    ()boolean -> {
                        %13 : boolean = constant @"true";
                        yield %13;
                    }
                    ()java.lang.String -> {
                        %4 : java.lang.String = constant @"else";
                        yield %4;
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
