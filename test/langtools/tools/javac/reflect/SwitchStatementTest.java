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
}
