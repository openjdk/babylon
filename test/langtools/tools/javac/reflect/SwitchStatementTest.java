import java.lang.runtime.CodeReflection;

/*
 * @test
 * @build SwitchStatementTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchStatementTest
 */
public class SwitchStatementTest {

    @IR("""
            func @"caseConstantRuleExpression" (%0 : java.lang.String)void -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                java.switch.statement %2
                    (%3 : java.lang.String)boolean -> {
                        %4 : java.lang.String = constant @"FOO";
                        %5 : boolean = invoke %3 %4 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %5;
                    }
                    ()void -> {
                        %6 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %7 : java.lang.String = constant @"BAR";
                        invoke %6 %7 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()void -> {
                        %11 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %12 : java.lang.String = constant @"BAZ";
                        invoke %11 %12 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%13 : java.lang.String)boolean -> {
                        %14 : java.lang.String = constant @"BAZ";
                        %15 : boolean = invoke %13 %14 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %15;
                    }
                    ()void -> {
                        %16 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %17 : java.lang.String = constant @"FOO";
                        invoke %16 %17 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %18 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %19 : java.lang.String = constant @"else";
                        invoke %18 %19 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    public static void caseConstantRuleExpression(String r) {
        switch (r) {
            case "FOO" -> System.out.println("BAR");
            case "BAR" -> System.out.println("BAZ");
            case "BAZ" -> System.out.println("FOO");
            default -> System.out.println("else");
        }
    }

    @IR("""
            func @"caseConstantRuleBlock" (%0 : java.lang.String)void -> {
                %1 : Var<java.lang.String> = var %0 @"r";
                %2 : java.lang.String = var.load %1;
                java.switch.statement %2
                    (%3 : java.lang.String)boolean -> {
                        %4 : java.lang.String = constant @"FOO";
                        %5 : boolean = invoke %3 %4 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %5;
                    }
                    ()void -> {
                        %6 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %7 : java.lang.String = constant @"BAR";
                        invoke %6 %7 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()void -> {
                        %11 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %12 : java.lang.String = constant @"BAZ";
                        invoke %11 %12 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%13 : java.lang.String)boolean -> {
                        %14 : java.lang.String = constant @"BAZ";
                        %15 : boolean = invoke %13 %14 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %15;
                    }
                    ()void -> {
                        %16 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %17 : java.lang.String = constant @"FOO";
                        invoke %16 %17 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %18 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %19 : java.lang.String = constant @"else";
                        invoke %18 %19 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    public static void caseConstantRuleBlock(String r) {
        switch (r) {
            case "FOO" -> {
                System.out.println("BAR");
            }
            case "BAR" -> {
                System.out.println("BAZ");
            }
            case "BAZ" -> {
                System.out.println("FOO");
            }
            default -> {
                System.out.println("else");
            }
        }
    }

    @IR("""
            func @"caseConstantStatement" (%0 : java.lang.String)void -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                java.switch.statement %2
                    (%3 : java.lang.String)boolean -> {
                        %4 : java.lang.String = constant @"FOO";
                        %5 : boolean = invoke %3 %4 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %5;
                    }
                    ()void -> {
                        %6 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %7 : java.lang.String = constant @"BAR";
                        invoke %6 %7 @"java.io.PrintStream::println(java.lang.String)void";
                        java.break;
                    }
                    (%8 : java.lang.String)boolean -> {
                        %9 : java.lang.String = constant @"BAR";
                        %10 : boolean = invoke %8 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %10;
                    }
                    ()void -> {
                        %11 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %12 : java.lang.String = constant @"BAZ";
                        invoke %11 %12 @"java.io.PrintStream::println(java.lang.String)void";
                        java.break;
                    }
                    (%13 : java.lang.String)boolean -> {
                        %14 : java.lang.String = constant @"BAZ";
                        %15 : boolean = invoke %13 %14 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %15;
                    }
                    ()void -> {
                        %16 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %17 : java.lang.String = constant @"FOO";
                        invoke %16 %17 @"java.io.PrintStream::println(java.lang.String)void";
                        java.break;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %18 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %19 : java.lang.String = constant @"else";
                        invoke %18 %19 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantStatement(String s) {
        switch (s) {
            case "FOO":
                System.out.println("BAR");
                break;
            case "BAR":
                System.out.println("BAZ");
                break;
            case "BAZ":
                System.out.println("FOO");;
                break;
            default:
                System.out.println("else");
        };
    }

    @IR("""
            func @"caseConstantMultiLabels" (%0 : char)void -> {
                %1 : Var<char> = var %0 @"c";
                %2 : char = var.load %1;
                %3 : char = invoke %2 @"java.lang.Character::toLowerCase(char)char";
                java.switch.statement %3
                    (%4 : char)boolean -> {
                        %5 : boolean = java.cor
                            ()boolean -> {
                                %6 : char = constant @"a";
                                %7 : boolean = eq %4 %6;
                                yield %7;
                            }
                            ()boolean -> {
                                %8 : char = constant @"e";
                                %9 : boolean = eq %4 %8;
                                yield %9;
                            }
                            ()boolean -> {
                                %10 : char = constant @"i";
                                %11 : boolean = eq %4 %10;
                                yield %11;
                            }
                            ()boolean -> {
                                %12 : char = constant @"o";
                                %13 : boolean = eq %4 %12;
                                yield %13;
                            }
                            ()boolean -> {
                                %14 : char = constant @"u";
                                %15 : boolean = eq %4 %14;
                                yield %15;
                            };
                        yield %5;
                    }
                    ()void -> {
                        %16 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %17 : java.lang.String = constant @"vowel";
                        invoke %16 %17 @"java.io.PrintStream::println(java.lang.String)void";
                        java.switch.fallthrough;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %18 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %19 : java.lang.String = constant @"else";
                        invoke %18 %19 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantMultiLabels(char c) {
        switch (Character.toLowerCase(c)) {
            case 'a', 'e', 'i', 'o', 'u':
                System.out.println("vowel");
            default:
                System.out.println("else");
        };
    }

    @IR("""
            func @"caseConstantThrow" (%0 : java.lang.Integer)void -> {
                %1 : Var<java.lang.Integer> = var %0 @"i";
                %2 : java.lang.Integer = var.load %1;
                java.switch.statement %2
                    (%3 : java.lang.Integer)boolean -> {
                        %4 : int = constant @"8";
                        %5 : java.lang.Integer = invoke %4 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %6 : boolean = invoke %3 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %6;
                    }
                    ()void -> {
                        %7 : java.lang.IllegalArgumentException = new @"func<java.lang.IllegalArgumentException>";
                        throw %7;
                    }
                    (%8 : java.lang.Integer)boolean -> {
                        %9 : int = constant @"9";
                        %10 : java.lang.Integer = invoke %9 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %11 : boolean = invoke %8 %10 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %11;
                    }
                    ()void -> {
                        %12 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %13 : java.lang.String = constant @"Nine";
                        invoke %12 %13 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %14 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %15 : java.lang.String = constant @"An integer";
                        invoke %14 %15 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantThrow(Integer i) {
        switch (i) {
            case 8 -> throw new IllegalArgumentException();
            case 9 -> System.out.println("Nine");
            default -> System.out.println("An integer");
        };
    }

    @IR("""
            func @"caseConstantNullLabel" (%0 : java.lang.String)void -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = var.load %1;
                java.switch.statement %2
                    (%3 : java.lang.String)boolean -> {
                        %4 : java.lang.Object = constant @null;
                        %5 : boolean = invoke %3 %4 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %5;
                    }
                    ()void -> {
                        %6 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %7 : java.lang.String = constant @"null";
                        invoke %6 %7 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %8 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %9 : java.lang.String = constant @"non null";
                        invoke %8 %9 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantNullLabel(String s) {
        switch (s) {
            case null -> System.out.println("null");
            default -> System.out.println("non null");
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
                    ()void -> {
                        yield;
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
            func @"caseConstantEnum" (%0 : SwitchStatementTest$Day)void -> {
                %1 : Var<SwitchStatementTest$Day> = var %0 @"d";
                %2 : SwitchStatementTest$Day = var.load %1;
                java.switch.statement %2
                    (%3 : SwitchStatementTest$Day)boolean -> {
                        %4 : boolean = java.cor
                            ()boolean -> {
                                %5 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::MON()SwitchStatementTest$Day";
                                %6 : boolean = invoke %3 %5 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %6;
                            }
                            ()boolean -> {
                                %7 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::FRI()SwitchStatementTest$Day";
                                %8 : boolean = invoke %3 %7 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %8;
                            }
                            ()boolean -> {
                                %9 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::SUN()SwitchStatementTest$Day";
                                %10 : boolean = invoke %3 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %10;
                            };
                        yield %4;
                    }
                    ()void -> {
                        %11 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %12 : int = constant @"6";
                        invoke %11 %12 @"java.io.PrintStream::println(int)void";
                        yield;
                    }
                    (%13 : SwitchStatementTest$Day)boolean -> {
                        %14 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::TUE()SwitchStatementTest$Day";
                        %15 : boolean = invoke %13 %14 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %15;
                    }
                    ()void -> {
                        %16 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %17 : int = constant @"7";
                        invoke %16 %17 @"java.io.PrintStream::println(int)void";
                        yield;
                    }
                    (%18 : SwitchStatementTest$Day)boolean -> {
                        %19 : boolean = java.cor
                            ()boolean -> {
                                %20 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::THU()SwitchStatementTest$Day";
                                %21 : boolean = invoke %18 %20 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %21;
                            }
                            ()boolean -> {
                                %22 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::SAT()SwitchStatementTest$Day";
                                %23 : boolean = invoke %18 %22 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %23;
                            };
                        yield %19;
                    }
                    ()void -> {
                        %24 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %25 : int = constant @"8";
                        invoke %24 %25 @"java.io.PrintStream::println(int)void";
                        yield;
                    }
                    (%26 : SwitchStatementTest$Day)boolean -> {
                        %27 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::WED()SwitchStatementTest$Day";
                        %28 : boolean = invoke %26 %27 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %28;
                    }
                    ()void -> {
                        %29 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %30 : int = constant @"9";
                        invoke %29 %30 @"java.io.PrintStream::println(int)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantEnum(Day d) {
        switch (d) {
            case MON, FRI, SUN -> System.out.println(6);
            case TUE -> System.out.println(7);
            case THU, SAT -> System.out.println(8);
            case WED -> System.out.println(9);
        }
    }

    static class Constants {
        static final int c1 = 12;
    }
    @IR("""
            func @"caseConstantOtherKindsOfExpr" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"i";
                %2 : int = constant @"11";
                %3 : Var<int> = var %2 @"eleven";
                %4 : int = var.load %1;
                java.switch.statement %4
                    (%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : int = constant @"15";
                        %8 : int = and %6 %7;
                        %9 : boolean = eq %5 %8;
                        yield %9;
                    }
                    ()void -> {
                        %10 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %11 : java.lang.String = constant @"1";
                        invoke %10 %11 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%12 : int)boolean -> {
                        %13 : int = constant @"4";
                        %14 : int = constant @"1";
                        %15 : int = ashr %13 %14;
                        %16 : boolean = eq %12 %15;
                        yield %16;
                    }
                    ()void -> {
                        %17 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %18 : java.lang.String = constant @"2";
                        invoke %17 %18 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%19 : int)boolean -> {
                        %20 : long = constant @"3";
                        %21 : int = conv %20;
                        %22 : boolean = eq %19 %21;
                        yield %22;
                    }
                    ()void -> {
                        %23 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %24 : java.lang.String = constant @"3";
                        invoke %23 %24 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%25 : int)boolean -> {
                        %26 : int = constant @"2";
                        %27 : int = constant @"1";
                        %28 : int = lshl %26 %27;
                        %29 : boolean = eq %25 %28;
                        yield %29;
                    }
                    ()void -> {
                        %30 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %31 : java.lang.String = constant @"4";
                        invoke %30 %31 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%32 : int)boolean -> {
                        %33 : int = constant @"10";
                        %34 : int = constant @"2";
                        %35 : int = div %33 %34;
                        %36 : boolean = eq %32 %35;
                        yield %36;
                    }
                    ()void -> {
                        %37 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %38 : java.lang.String = constant @"5";
                        invoke %37 %38 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%39 : int)boolean -> {
                        %40 : int = constant @"12";
                        %41 : int = constant @"6";
                        %42 : int = sub %40 %41;
                        %43 : boolean = eq %39 %42;
                        yield %43;
                    }
                    ()void -> {
                        %44 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %45 : java.lang.String = constant @"6";
                        invoke %44 %45 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%46 : int)boolean -> {
                        %47 : int = constant @"3";
                        %48 : int = constant @"4";
                        %49 : int = add %47 %48;
                        %50 : boolean = eq %46 %49;
                        yield %50;
                    }
                    ()void -> {
                        %51 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %52 : java.lang.String = constant @"7";
                        invoke %51 %52 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%53 : int)boolean -> {
                        %54 : int = constant @"2";
                        %55 : int = constant @"2";
                        %56 : int = mul %54 %55;
                        %57 : int = constant @"2";
                        %58 : int = mul %56 %57;
                        %59 : boolean = eq %53 %58;
                        yield %59;
                    }
                    ()void -> {
                        %60 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %61 : java.lang.String = constant @"8";
                        invoke %60 %61 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%62 : int)boolean -> {
                        %63 : int = constant @"8";
                        %64 : int = constant @"1";
                        %65 : int = or %63 %64;
                        %66 : boolean = eq %62 %65;
                        yield %66;
                    }
                    ()void -> {
                        %67 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %68 : java.lang.String = constant @"9";
                        invoke %67 %68 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%69 : int)boolean -> {
                        %70 : int = constant @"10";
                        %71 : boolean = eq %69 %70;
                        yield %71;
                    }
                    ()void -> {
                        %72 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %73 : java.lang.String = constant @"10";
                        invoke %72 %73 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%74 : int)boolean -> {
                        %75 : int = var.load %3;
                        %76 : boolean = eq %74 %75;
                        yield %76;
                    }
                    ()void -> {
                        %77 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %78 : java.lang.String = constant @"11";
                        invoke %77 %78 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    (%79 : int)boolean -> {
                        %80 : int = field.load @"SwitchStatementTest$Constants::c1()int";
                        %81 : boolean = eq %79 %80;
                        yield %81;
                    }
                    ()void -> {
                        %82 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %83 : int = field.load @"SwitchStatementTest$Constants::c1()int";
                        invoke %82 %83 @"java.io.PrintStream::println(int)void";
                        yield;
                    }
                    (%84 : int)boolean -> {
                        %85 : int = java.cexpression
                            ()boolean -> {
                                %86 : int = constant @"1";
                                %87 : int = constant @"0";
                                %88 : boolean = gt %86 %87;
                                yield %88;
                            }
                            ()int -> {
                                %89 : int = constant @"13";
                                yield %89;
                            }
                            ()int -> {
                                %90 : int = constant @"133";
                                yield %90;
                            };
                        %91 : boolean = eq %84 %85;
                        yield %91;
                    }
                    ()void -> {
                        %92 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %93 : java.lang.String = constant @"13";
                        invoke %92 %93 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %94 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %95 : java.lang.String = constant @"an int";
                        invoke %94 %95 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    @CodeReflection
    private static void caseConstantOtherKindsOfExpr(int i) {
        final int eleven = 11;
        switch (i) {
            case 1 & 0xF -> System.out.println("1");
            case 4>>1 -> System.out.println("2");
            case (int) 3L -> System.out.println("3");
            case 2<<1 -> System.out.println("4");
            case 10 / 2 -> System.out.println("5");
            case 12 - 6 -> System.out.println("6");
            case 3 + 4 -> System.out.println("7");
            case 2 * 2 * 2 -> System.out.println("8");
            case 8 | 1 -> System.out.println("9");
            case (10) -> System.out.println("10");
            case eleven -> System.out.println("11");
            case Constants.c1 -> System.out.println(Constants.c1);
            case 1 > 0 ? 13 : 133 -> System.out.println("13");
            default -> System.out.println("an int");
        }
    }

    @IR("""
            func @"caseConstantConv" (%0 : short)java.lang.String -> {
                %1 : Var<short> = var %0 @"a";
                %2 : int = constant @"1";
                %3 : short = conv %2;
                %4 : Var<short> = var %3 @"s";
                %5 : int = constant @"2";
                %6 : byte = conv %5;
                %7 : Var<byte> = var %6 @"b";
                %8 : java.lang.String = constant @"";
                %9 : Var<java.lang.String> = var %8 @"r";
                %10 : short = var.load %1;
                java.switch.statement %10
                    (%11 : short)boolean -> {
                        %12 : short = var.load %4;
                        %13 : boolean = eq %11 %12;
                        yield %13;
                    }
                    ()void -> {
                        %14 : java.lang.String = var.load %9;
                        %15 : java.lang.String = constant @"one";
                        %16 : java.lang.String = add %14 %15;
                        var.store %9 %16;
                        yield;
                    }
                    (%17 : short)boolean -> {
                        %18 : byte = var.load %7;
                        %19 : short = conv %18;
                        %20 : boolean = eq %17 %19;
                        yield %20;
                    }
                    ()void -> {
                        %21 : java.lang.String = var.load %9;
                        %22 : java.lang.String = constant @"two";
                        %23 : java.lang.String = add %21 %22;
                        var.store %9 %23;
                        yield;
                    }
                    (%24 : short)boolean -> {
                        %25 : int = constant @"3";
                        %26 : short = conv %25;
                        %27 : boolean = eq %24 %26;
                        yield %27;
                    }
                    ()void -> {
                        %28 : java.lang.String = var.load %9;
                        %29 : java.lang.String = constant @"three";
                        %30 : java.lang.String = add %28 %29;
                        var.store %9 %30;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %31 : java.lang.String = var.load %9;
                        %32 : java.lang.String = constant @"else";
                        %33 : java.lang.String = add %31 %32;
                        var.store %9 %33;
                        yield;
                    };
                %34 : java.lang.String = var.load %9;
                return %34;
            };
            """)
    @CodeReflection
    static String caseConstantConv(short a) { // @@@ tests should be easy to test with interpreter, e.g. tests returning a string result
        final short s = 1;
        final byte b = 2;
        String r = "";
        switch (a) {
            // @@@ string concat is modeled as: add s1 s2
            case s -> r += "one"; // identity, short -> short
            case b -> r += "two"; // widening primitive conversion, byte -> short
            case 3 -> r += "three"; // narrowing primitive conversion, int -> short
            default -> r += "else";
        }
        return r;
    }

    @IR("""
            func @"caseConstantConv2" (%0 : java.lang.Byte)java.lang.String -> {
                %1 : Var<java.lang.Byte> = var %0 @"a";
                %2 : int = constant @"2";
                %3 : byte = conv %2;
                %4 : Var<byte> = var %3 @"b";
                %5 : java.lang.String = constant @"";
                %6 : Var<java.lang.String> = var %5 @"r";
                %7 : java.lang.Byte = var.load %1;
                java.switch.statement %7
                    (%8 : java.lang.Byte)boolean -> {
                        %9 : int = constant @"1";
                        %10 : byte = conv %9;
                        %11 : java.lang.Byte = invoke %10 @"java.lang.Byte::valueOf(byte)java.lang.Byte";
                        %12 : boolean = invoke %8 %11 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %12;
                    }
                    ()void -> {
                        %13 : java.lang.String = var.load %6;
                        %14 : java.lang.String = constant @"one";
                        %15 : java.lang.String = add %13 %14;
                        var.store %6 %15;
                        yield;
                    }
                    (%16 : java.lang.Byte)boolean -> {
                        %17 : byte = var.load %4;
                        %18 : java.lang.Byte = invoke %17 @"java.lang.Byte::valueOf(byte)java.lang.Byte";
                        %19 : boolean = invoke %16 %18 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %19;
                    }
                    ()void -> {
                        %20 : java.lang.String = var.load %6;
                        %21 : java.lang.String = constant @"two";
                        %22 : java.lang.String = add %20 %21;
                        var.store %6 %22;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %23 : java.lang.String = var.load %6;
                        %24 : java.lang.String = constant @"default";
                        %25 : java.lang.String = add %23 %24;
                        var.store %6 %25;
                        yield;
                    };
                %26 : java.lang.String = var.load %6;
                return %26;
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
}
