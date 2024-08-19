import java.lang.runtime.CodeReflection;
import java.util.Collection;
import java.util.RandomAccess;
import java.util.Stack;

/*
 * @test
 * @build SwitchStatementTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SwitchStatementTest
 */
public class SwitchStatementTest {

    @IR("""
            func @"caseConstantRuleExpression" (%0 : java.lang.String)java.lang.String -> {
                  %1 : Var<java.lang.String> = var %0 @"r";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"s";
                  %4 : java.lang.String = var.load %1;
                  java.switch.statement %4
                      (%5 : java.lang.String)boolean -> {
                          %6 : java.lang.String = constant @"FOO";
                          %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %7;
                      }
                      ()void -> {
                          %8 : java.lang.String = var.load %3;
                          %9 : java.lang.String = constant @"BAR";
                          %10 : java.lang.String = concat %8 %9;
                          var.store %3 %10;
                          yield;
                      }
                      (%11 : java.lang.String)boolean -> {
                          %12 : java.lang.String = constant @"BAR";
                          %13 : boolean = invoke %11 %12 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %13;
                      }
                      ()void -> {
                          %14 : java.lang.String = var.load %3;
                          %15 : java.lang.String = constant @"BAZ";
                          %16 : java.lang.String = concat %14 %15;
                          var.store %3 %16;
                          yield;
                      }
                      (%17 : java.lang.String)boolean -> {
                          %18 : java.lang.String = constant @"BAZ";
                          %19 : boolean = invoke %17 %18 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %19;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"FOO";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"caseConstantRuleBlock" (%0 : java.lang.String)java.lang.String -> {
                  %1 : Var<java.lang.String> = var %0 @"r";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"s";
                  %4 : java.lang.String = var.load %1;
                  java.switch.statement %4
                      (%5 : java.lang.String)boolean -> {
                          %6 : java.lang.String = constant @"FOO";
                          %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %7;
                      }
                      ()void -> {
                          %8 : java.lang.String = var.load %3;
                          %9 : java.lang.String = constant @"BAR";
                          %10 : java.lang.String = concat %8 %9;
                          var.store %3 %10;
                          yield;
                      }
                      (%11 : java.lang.String)boolean -> {
                          %12 : java.lang.String = constant @"BAR";
                          %13 : boolean = invoke %11 %12 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %13;
                      }
                      ()void -> {
                          %14 : java.lang.String = var.load %3;
                          %15 : java.lang.String = constant @"BAZ";
                          %16 : java.lang.String = concat %14 %15;
                          var.store %3 %16;
                          yield;
                      }
                      (%17 : java.lang.String)boolean -> {
                          %18 : java.lang.String = constant @"BAZ";
                          %19 : boolean = invoke %17 %18 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %19;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"FOO";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"caseConstantStatement" (%0 : java.lang.String)java.lang.String -> {
                  %1 : Var<java.lang.String> = var %0 @"s";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : java.lang.String = var.load %1;
                  java.switch.statement %4
                      (%5 : java.lang.String)boolean -> {
                          %6 : java.lang.String = constant @"FOO";
                          %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %7;
                      }
                      ()void -> {
                          %8 : java.lang.String = var.load %3;
                          %9 : java.lang.String = constant @"BAR";
                          %10 : java.lang.String = concat %8 %9;
                          var.store %3 %10;
                          java.break;
                      }
                      (%11 : java.lang.String)boolean -> {
                          %12 : java.lang.String = constant @"BAR";
                          %13 : boolean = invoke %11 %12 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %13;
                      }
                      ()void -> {
                          %14 : java.lang.String = var.load %3;
                          %15 : java.lang.String = constant @"BAZ";
                          %16 : java.lang.String = concat %14 %15;
                          var.store %3 %16;
                          java.break;
                      }
                      (%17 : java.lang.String)boolean -> {
                          %18 : java.lang.String = constant @"BAZ";
                          %19 : boolean = invoke %17 %18 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %19;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"FOO";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          java.break;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"caseConstantMultiLabels" (%0 : char)java.lang.String -> {
                %1 : Var<char> = var %0 @"c";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : char = var.load %1;
                %5 : char = invoke %4 @"java.lang.Character::toLowerCase(char)char";
                java.switch.statement %5
                    (%6 : char)boolean -> {
                        %7 : boolean = java.cor
                            ()boolean -> {
                                %8 : char = constant @"a";
                                %9 : boolean = eq %6 %8;
                                yield %9;
                            }
                            ()boolean -> {
                                %10 : char = constant @"e";
                                %11 : boolean = eq %6 %10;
                                yield %11;
                            }
                            ()boolean -> {
                                %12 : char = constant @"i";
                                %13 : boolean = eq %6 %12;
                                yield %13;
                            }
                            ()boolean -> {
                                %14 : char = constant @"o";
                                %15 : boolean = eq %6 %14;
                                yield %15;
                            }
                            ()boolean -> {
                                %16 : char = constant @"u";
                                %17 : boolean = eq %6 %16;
                                yield %17;
                            };
                        yield %7;
                    }
                    ()void -> {
                        %18 : java.lang.String = var.load %3;
                        %19 : java.lang.String = constant @"vowel";
                        %20 : java.lang.String = concat %18 %19;
                        var.store %3 %20;
                        java.break;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %21 : java.lang.String = var.load %3;
                        %22 : java.lang.String = constant @"consonant";
                        %23 : java.lang.String = concat %21 %22;
                        var.store %3 %23;
                        yield;
                    };
                %24 : java.lang.String = var.load %3;
                return %24;
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
            func @"caseConstantThrow" (%0 : java.lang.Integer)java.lang.String -> {
                %1 : Var<java.lang.Integer> = var %0 @"i";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.Integer = var.load %1;
                java.switch.statement %4
                    (%5 : java.lang.Integer)boolean -> {
                        %6 : int = constant @"8";
                        %7 : java.lang.Integer = invoke %6 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %8 : boolean = invoke %5 %7 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %8;
                    }
                    ()void -> {
                        %9 : java.lang.IllegalArgumentException = new @"func<java.lang.IllegalArgumentException>";
                        throw %9;
                    }
                    (%10 : java.lang.Integer)boolean -> {
                        %11 : int = constant @"9";
                        %12 : java.lang.Integer = invoke %11 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %13 : boolean = invoke %10 %12 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %13;
                    }
                    ()void -> {
                        %14 : java.lang.String = var.load %3;
                        %15 : java.lang.String = constant @"Nine";
                        %16 : java.lang.String = concat %14 %15;
                        var.store %3 %16;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %17 : java.lang.String = var.load %3;
                        %18 : java.lang.String = constant @"An integer";
                        %19 : java.lang.String = concat %17 %18;
                        var.store %3 %19;
                        yield;
                    };
                %20 : java.lang.String = var.load %3;
                return %20;
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
            func @"caseConstantNullLabel" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.String = var.load %1;
                java.switch.statement %4
                    (%5 : java.lang.String)boolean -> {
                        %6 : java.lang.Object = constant @null;
                        %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %7;
                    }
                    ()void -> {
                        %8 : java.lang.String = var.load %3;
                        %9 : java.lang.String = constant @"null";
                        %10 : java.lang.String = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %11 : java.lang.String = var.load %3;
                        %12 : java.lang.String = constant @"non null";
                        %13 : java.lang.String = concat %11 %12;
                        var.store %3 %13;
                        yield;
                    };
                %14 : java.lang.String = var.load %3;
                return %14;
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
            func @"caseConstantFallThrough" (%0 : char)java.lang.String -> {
                  %1 : Var<char> = var %0 @"c";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : char = var.load %1;
                  java.switch.statement %4
                      (%5 : char)boolean -> {
                          %6 : char = constant @"A";
                          %7 : boolean = eq %5 %6;
                          yield %7;
                      }
                      ()void -> {
                          java.switch.fallthrough;
                      }
                      (%8 : char)boolean -> {
                          %9 : char = constant @"B";
                          %10 : boolean = eq %8 %9;
                          yield %10;
                      }
                      ()void -> {
                          %11 : java.lang.String = var.load %3;
                          %12 : java.lang.String = constant @"A or B";
                          %13 : java.lang.String = concat %11 %12;
                          var.store %3 %13;
                          java.break;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %14 : java.lang.String = var.load %3;
                          %15 : java.lang.String = constant @"Neither A nor B";
                          %16 : java.lang.String = concat %14 %15;
                          var.store %3 %16;
                          yield;
                      };
                  %17 : java.lang.String = var.load %3;
                  return %17;
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
            func @"caseConstantEnum" (%0 : SwitchStatementTest$Day)java.lang.String -> {
                %1 : Var<SwitchStatementTest$Day> = var %0 @"d";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : SwitchStatementTest$Day = var.load %1;
                java.switch.statement %4
                    (%5 : SwitchStatementTest$Day)boolean -> {
                        %6 : boolean = java.cor
                            ()boolean -> {
                                %7 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::MON()SwitchStatementTest$Day";
                                %8 : boolean = invoke %5 %7 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %8;
                            }
                            ()boolean -> {
                                %9 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::FRI()SwitchStatementTest$Day";
                                %10 : boolean = invoke %5 %9 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %10;
                            }
                            ()boolean -> {
                                %11 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::SUN()SwitchStatementTest$Day";
                                %12 : boolean = invoke %5 %11 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %12;
                            };
                        yield %6;
                    }
                    ()void -> {
                        %13 : java.lang.String = var.load %3;
                        %14 : int = constant @"6";
                        %15 : java.lang.Integer = invoke %14 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %16 : java.lang.String = concat %13 %15;
                        var.store %3 %16;
                        yield;
                    }
                    (%17 : SwitchStatementTest$Day)boolean -> {
                        %18 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::TUE()SwitchStatementTest$Day";
                        %19 : boolean = invoke %17 %18 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %19;
                    }
                    ()void -> {
                        %20 : java.lang.String = var.load %3;
                        %21 : int = constant @"7";
                        %22 : java.lang.Integer = invoke %21 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %23 : java.lang.String = concat %20 %22;
                        var.store %3 %23;
                        yield;
                    }
                    (%24 : SwitchStatementTest$Day)boolean -> {
                        %25 : boolean = java.cor
                            ()boolean -> {
                                %26 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::THU()SwitchStatementTest$Day";
                                %27 : boolean = invoke %24 %26 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %27;
                            }
                            ()boolean -> {
                                %28 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::SAT()SwitchStatementTest$Day";
                                %29 : boolean = invoke %24 %28 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                                yield %29;
                            };
                        yield %25;
                    }
                    ()void -> {
                        %30 : java.lang.String = var.load %3;
                        %31 : int = constant @"8";
                        %32 : java.lang.Integer = invoke %31 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %33 : java.lang.String = concat %30 %32;
                        var.store %3 %33;
                        yield;
                    }
                    (%34 : SwitchStatementTest$Day)boolean -> {
                        %35 : SwitchStatementTest$Day = field.load @"SwitchStatementTest$Day::WED()SwitchStatementTest$Day";
                        %36 : boolean = invoke %34 %35 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %36;
                    }
                    ()void -> {
                        %37 : java.lang.String = var.load %3;
                        %38 : int = constant @"9";
                        %39 : java.lang.Integer = invoke %38 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %40 : java.lang.String = concat %37 %39;
                        var.store %3 %40;
                        yield;
                    };
                %41 : java.lang.String = var.load %3;
                return %41;
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
            func @"caseConstantOtherKindsOfExpr" (%0 : int)java.lang.String -> {
                %1 : Var<int> = var %0 @"i";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : int = constant @"11";
                %5 : Var<int> = var %4 @"eleven";
                %6 : int = var.load %1;
                java.switch.statement %6
                    (%7 : int)boolean -> {
                        %8 : int = constant @"1";
                        %9 : int = constant @"15";
                        %10 : int = and %8 %9;
                        %11 : boolean = eq %7 %10;
                        yield %11;
                    }
                    ()void -> {
                        %12 : java.lang.String = var.load %3;
                        %13 : int = constant @"1";
                        %14 : java.lang.Integer = invoke %13 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %15 : java.lang.String = concat %12 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : int)boolean -> {
                        %17 : int = constant @"4";
                        %18 : int = constant @"1";
                        %19 : int = ashr %17 %18;
                        %20 : boolean = eq %16 %19;
                        yield %20;
                    }
                    ()void -> {
                        %21 : java.lang.String = var.load %3;
                        %22 : java.lang.String = constant @"2";
                        %23 : java.lang.String = concat %21 %22;
                        var.store %3 %23;
                        yield;
                    }
                    (%24 : int)boolean -> {
                        %25 : long = constant @"3";
                        %26 : int = conv %25;
                        %27 : boolean = eq %24 %26;
                        yield %27;
                    }
                    ()void -> {
                        %28 : java.lang.String = var.load %3;
                        %29 : int = constant @"3";
                        %30 : java.lang.Integer = invoke %29 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %31 : java.lang.String = concat %28 %30;
                        var.store %3 %31;
                        yield;
                    }
                    (%32 : int)boolean -> {
                        %33 : int = constant @"2";
                        %34 : int = constant @"1";
                        %35 : int = lshl %33 %34;
                        %36 : boolean = eq %32 %35;
                        yield %36;
                    }
                    ()void -> {
                        %37 : java.lang.String = var.load %3;
                        %38 : int = constant @"4";
                        %39 : java.lang.Integer = invoke %38 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %40 : java.lang.String = concat %37 %39;
                        var.store %3 %40;
                        yield;
                    }
                    (%41 : int)boolean -> {
                        %42 : int = constant @"10";
                        %43 : int = constant @"2";
                        %44 : int = div %42 %43;
                        %45 : boolean = eq %41 %44;
                        yield %45;
                    }
                    ()void -> {
                        %46 : java.lang.String = var.load %3;
                        %47 : int = constant @"5";
                        %48 : java.lang.Integer = invoke %47 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %49 : java.lang.String = concat %46 %48;
                        var.store %3 %49;
                        yield;
                    }
                    (%50 : int)boolean -> {
                        %51 : int = constant @"12";
                        %52 : int = constant @"6";
                        %53 : int = sub %51 %52;
                        %54 : boolean = eq %50 %53;
                        yield %54;
                    }
                    ()void -> {
                        %55 : java.lang.String = var.load %3;
                        %56 : int = constant @"6";
                        %57 : java.lang.Integer = invoke %56 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %58 : java.lang.String = concat %55 %57;
                        var.store %3 %58;
                        yield;
                    }
                    (%59 : int)boolean -> {
                        %60 : int = constant @"3";
                        %61 : int = constant @"4";
                        %62 : int = add %60 %61;
                        %63 : boolean = eq %59 %62;
                        yield %63;
                    }
                    ()void -> {
                        %64 : java.lang.String = var.load %3;
                        %65 : int = constant @"7";
                        %66 : java.lang.Integer = invoke %65 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %67 : java.lang.String = concat %64 %66;
                        var.store %3 %67;
                        yield;
                    }
                    (%68 : int)boolean -> {
                        %69 : int = constant @"2";
                        %70 : int = constant @"2";
                        %71 : int = mul %69 %70;
                        %72 : int = constant @"2";
                        %73 : int = mul %71 %72;
                        %74 : boolean = eq %68 %73;
                        yield %74;
                    }
                    ()void -> {
                        %75 : java.lang.String = var.load %3;
                        %76 : int = constant @"8";
                        %77 : java.lang.Integer = invoke %76 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %78 : java.lang.String = concat %75 %77;
                        var.store %3 %78;
                        yield;
                    }
                    (%79 : int)boolean -> {
                        %80 : int = constant @"8";
                        %81 : int = constant @"1";
                        %82 : int = or %80 %81;
                        %83 : boolean = eq %79 %82;
                        yield %83;
                    }
                    ()void -> {
                        %84 : java.lang.String = var.load %3;
                        %85 : int = constant @"9";
                        %86 : java.lang.Integer = invoke %85 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %87 : java.lang.String = concat %84 %86;
                        var.store %3 %87;
                        yield;
                    }
                    (%88 : int)boolean -> {
                        %89 : int = constant @"10";
                        %90 : boolean = eq %88 %89;
                        yield %90;
                    }
                    ()void -> {
                        %91 : java.lang.String = var.load %3;
                        %92 : int = constant @"10";
                        %93 : java.lang.Integer = invoke %92 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %94 : java.lang.String = concat %91 %93;
                        var.store %3 %94;
                        yield;
                    }
                    (%95 : int)boolean -> {
                        %96 : int = var.load %5;
                        %97 : boolean = eq %95 %96;
                        yield %97;
                    }
                    ()void -> {
                        %98 : java.lang.String = var.load %3;
                        %99 : int = constant @"11";
                        %100 : java.lang.Integer = invoke %99 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %101 : java.lang.String = concat %98 %100;
                        var.store %3 %101;
                        yield;
                    }
                    (%102 : int)boolean -> {
                        %103 : int = field.load @"SwitchStatementTest$Constants::c1()int";
                        %104 : boolean = eq %102 %103;
                        yield %104;
                    }
                    ()void -> {
                        %105 : java.lang.String = var.load %3;
                        %106 : int = field.load @"SwitchStatementTest$Constants::c1()int";
                        %107 : java.lang.Integer = invoke %106 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %108 : java.lang.String = concat %105 %107;
                        var.store %3 %108;
                        yield;
                    }
                    (%109 : int)boolean -> {
                        %110 : int = java.cexpression
                            ()boolean -> {
                                %111 : int = constant @"1";
                                %112 : int = constant @"0";
                                %113 : boolean = gt %111 %112;
                                yield %113;
                            }
                            ()int -> {
                                %114 : int = constant @"13";
                                yield %114;
                            }
                            ()int -> {
                                %115 : int = constant @"133";
                                yield %115;
                            };
                        %116 : boolean = eq %109 %110;
                        yield %116;
                    }
                    ()void -> {
                        %117 : java.lang.String = var.load %3;
                        %118 : int = constant @"13";
                        %119 : java.lang.Integer = invoke %118 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %120 : java.lang.String = concat %117 %119;
                        var.store %3 %120;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %121 : java.lang.String = var.load %3;
                        %122 : java.lang.String = constant @"an int";
                        %123 : java.lang.String = concat %121 %122;
                        var.store %3 %123;
                        yield;
                    };
                %124 : java.lang.String = var.load %3;
                return %124;
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
                          %16 : java.lang.String = concat %14 %15;
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
                          %23 : java.lang.String = concat %21 %22;
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
                          %30 : java.lang.String = concat %28 %29;
                          var.store %9 %30;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %31 : java.lang.String = var.load %9;
                          %32 : java.lang.String = constant @"else";
                          %33 : java.lang.String = concat %31 %32;
                          var.store %9 %33;
                          yield;
                      };
                  %34 : java.lang.String = var.load %9;
                  return %34;
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
                        %15 : java.lang.String = concat %13 %14;
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
                        %22 : java.lang.String = concat %20 %21;
                        var.store %6 %22;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %23 : java.lang.String = var.load %6;
                        %24 : java.lang.String = constant @"default";
                        %25 : java.lang.String = concat %23 %24;
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

    @IR("""
            func @"nonEnhancedSwStatNoDefault" (%0 : int)java.lang.String -> {
                %1 : Var<int> = var %0 @"a";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : int = var.load %1;
                java.switch.statement %4
                    (%5 : int)boolean -> {
                        %6 : int = constant @"1";
                        %7 : boolean = eq %5 %6;
                        yield %7;
                    }
                    ()void -> {
                        %8 : java.lang.String = var.load %3;
                        %9 : java.lang.String = constant @"1";
                        %10 : java.lang.String = concat %8 %9;
                        var.store %3 %10;
                        yield;
                    }
                    (%11 : int)boolean -> {
                        %12 : int = constant @"2";
                        %13 : boolean = eq %11 %12;
                        yield %13;
                    }
                    ()void -> {
                        %14 : java.lang.String = var.load %3;
                        %15 : int = constant @"2";
                        %16 : java.lang.Integer = invoke %15 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %17 : java.lang.String = concat %14 %16;
                        var.store %3 %17;
                        yield;
                    };
                %18 : java.lang.String = var.load %3;
                return %18;
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
            func @"enhancedSwStatNoDefault1" (%0 : SwitchStatementTest$E)java.lang.String -> {
                  %1 : Var<SwitchStatementTest$E> = var %0 @"e";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : SwitchStatementTest$E = var.load %1;
                  java.switch.statement %4
                      (%5 : SwitchStatementTest$E)boolean -> {
                          %6 : SwitchStatementTest$E = field.load @"SwitchStatementTest$E::A()SwitchStatementTest$E";
                          %7 : boolean = invoke %5 %6 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %7;
                      }
                      ()void -> {
                          %8 : java.lang.String = var.load %3;
                          %9 : SwitchStatementTest$E = field.load @"SwitchStatementTest$E::A()SwitchStatementTest$E";
                          %10 : java.lang.String = cast %9 @"java.lang.String";
                          %11 : java.lang.String = concat %8 %10;
                          var.store %3 %11;
                          yield;
                      }
                      (%12 : SwitchStatementTest$E)boolean -> {
                          %13 : SwitchStatementTest$E = field.load @"SwitchStatementTest$E::B()SwitchStatementTest$E";
                          %14 : boolean = invoke %12 %13 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %14;
                      }
                      ()void -> {
                          %15 : java.lang.String = var.load %3;
                          %16 : SwitchStatementTest$E = field.load @"SwitchStatementTest$E::B()SwitchStatementTest$E";
                          %17 : java.lang.String = cast %16 @"java.lang.String";
                          %18 : java.lang.String = concat %15 %17;
                          var.store %3 %18;
                          yield;
                      }
                      (%19 : SwitchStatementTest$E)boolean -> {
                          %20 : java.lang.Object = constant @null;
                          %21 : boolean = invoke %19 %20 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %21;
                      }
                      ()void -> {
                          %22 : java.lang.String = var.load %3;
                          %23 : java.lang.String = constant @"null";
                          %24 : java.lang.String = concat %22 %23;
                          var.store %3 %24;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %25 : java.lang.MatchException = new @"func<java.lang.MatchException>";
                          throw %25;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"enhancedSwStatNoDefault2" (%0 : SwitchStatementTest$I)java.lang.String -> {
                %1 : Var<SwitchStatementTest$I> = var %0 @"i";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : SwitchStatementTest$I = var.load %1;
                %5 : SwitchStatementTest$K = constant @null;
                %6 : Var<SwitchStatementTest$K> = var %5 @"k";
                %7 : SwitchStatementTest$J = constant @null;
                %8 : Var<SwitchStatementTest$J> = var %7 @"j";
                java.switch.statement %4
                    (%9 : SwitchStatementTest$I)boolean -> {
                        %10 : boolean = pattern.match %9
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<SwitchStatementTest$K> -> {
                                %11 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<SwitchStatementTest$K> = pattern.binding @"k";
                                yield %11;
                            }
                            (%12 : SwitchStatementTest$K)void -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()void -> {
                        %13 : java.lang.String = var.load %3;
                        %14 : java.lang.String = constant @"K";
                        %15 : java.lang.String = concat %13 %14;
                        var.store %3 %15;
                        yield;
                    }
                    (%16 : SwitchStatementTest$I)boolean -> {
                        %17 : boolean = pattern.match %16
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<SwitchStatementTest$J> -> {
                                %18 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<SwitchStatementTest$J> = pattern.binding @"j";
                                yield %18;
                            }
                            (%19 : SwitchStatementTest$J)void -> {
                                var.store %8 %19;
                                yield;
                            };
                        yield %17;
                    }
                    ()void -> {
                        %20 : java.lang.String = var.load %3;
                        %21 : java.lang.String = constant @"J";
                        %22 : java.lang.String = concat %20 %21;
                        var.store %3 %22;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %23 : java.lang.MatchException = new @"func<java.lang.MatchException>";
                        throw %23;
                    };
                %24 : java.lang.String = var.load %3;
                return %24;
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
            func @"enhancedSwStatUnconditionalPattern" (%0 : java.lang.String)java.lang.String -> {
                %1 : Var<java.lang.String> = var %0 @"s";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.String = var.load %1;
                %5 : java.lang.Object = constant @null;
                %6 : Var<java.lang.Object> = var %5 @"o";
                java.switch.statement %4
                    (%7 : java.lang.String)boolean -> {
                        %8 : java.lang.String = constant @"A";
                        %9 : boolean = invoke %7 %8 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                        yield %9;
                    }
                    ()void -> {
                        %10 : java.lang.String = var.load %3;
                        %11 : java.lang.String = constant @"A";
                        %12 : java.lang.String = concat %10 %11;
                        var.store %3 %12;
                        yield;
                    }
                    (%13 : java.lang.String)boolean -> {
                        %14 : boolean = pattern.match %13
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Object> -> {
                                %15 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Object> = pattern.binding @"o";
                                yield %15;
                            }
                            (%16 : java.lang.Object)void -> {
                                var.store %6 %16;
                                yield;
                            };
                        yield %14;
                    }
                    ()void -> {
                        %17 : java.lang.String = var.load %3;
                        %18 : java.lang.String = constant @"obj";
                        %19 : java.lang.String = concat %17 %18;
                        var.store %3 %19;
                        yield;
                    };
                %20 : java.lang.String = var.load %3;
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
            func @"casePatternRuleExpression" (%0 : java.lang.Object)java.lang.String -> {
                  %1 : Var<java.lang.Object> = var %0 @"o";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : java.lang.Object = var.load %1;
                  %5 : java.lang.Integer = constant @null;
                  %6 : Var<java.lang.Integer> = var %5 @"i";
                  %7 : java.lang.String = constant @null;
                  %8 : Var<java.lang.String> = var %7 @"s";
                  java.switch.statement %4
                      (%9 : java.lang.Object)boolean -> {
                          %10 : boolean = pattern.match %9
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> -> {
                                  %11 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> = pattern.binding @"i";
                                  yield %11;
                              }
                              (%12 : java.lang.Integer)void -> {
                                  var.store %6 %12;
                                  yield;
                              };
                          yield %10;
                      }
                      ()void -> {
                          %13 : java.lang.String = var.load %3;
                          %14 : java.lang.String = constant @"integer";
                          %15 : java.lang.String = concat %13 %14;
                          var.store %3 %15;
                          yield;
                      }
                      (%16 : java.lang.Object)boolean -> {
                          %17 : boolean = pattern.match %16
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                  %18 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                  yield %18;
                              }
                              (%19 : java.lang.String)void -> {
                                  var.store %8 %19;
                                  yield;
                              };
                          yield %17;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"string";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"casePatternRuleBlock" (%0 : java.lang.Object)java.lang.String -> {
                  %1 : Var<java.lang.Object> = var %0 @"o";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : java.lang.Object = var.load %1;
                  %5 : java.lang.Integer = constant @null;
                  %6 : Var<java.lang.Integer> = var %5 @"i";
                  %7 : java.lang.String = constant @null;
                  %8 : Var<java.lang.String> = var %7 @"s";
                  java.switch.statement %4
                      (%9 : java.lang.Object)boolean -> {
                          %10 : boolean = pattern.match %9
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> -> {
                                  %11 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> = pattern.binding @"i";
                                  yield %11;
                              }
                              (%12 : java.lang.Integer)void -> {
                                  var.store %6 %12;
                                  yield;
                              };
                          yield %10;
                      }
                      ()void -> {
                          %13 : java.lang.String = var.load %3;
                          %14 : java.lang.String = constant @"integer";
                          %15 : java.lang.String = concat %13 %14;
                          var.store %3 %15;
                          yield;
                      }
                      (%16 : java.lang.Object)boolean -> {
                          %17 : boolean = pattern.match %16
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                  %18 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                  yield %18;
                              }
                              (%19 : java.lang.String)void -> {
                                  var.store %8 %19;
                                  yield;
                              };
                          yield %17;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"string";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"casePatternStatement" (%0 : java.lang.Object)java.lang.String -> {
                  %1 : Var<java.lang.Object> = var %0 @"o";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : java.lang.Object = var.load %1;
                  %5 : java.lang.Integer = constant @null;
                  %6 : Var<java.lang.Integer> = var %5 @"i";
                  %7 : java.lang.String = constant @null;
                  %8 : Var<java.lang.String> = var %7 @"s";
                  java.switch.statement %4
                      (%9 : java.lang.Object)boolean -> {
                          %10 : boolean = pattern.match %9
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> -> {
                                  %11 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> = pattern.binding @"i";
                                  yield %11;
                              }
                              (%12 : java.lang.Integer)void -> {
                                  var.store %6 %12;
                                  yield;
                              };
                          yield %10;
                      }
                      ()void -> {
                          %13 : java.lang.String = var.load %3;
                          %14 : java.lang.String = constant @"integer";
                          %15 : java.lang.String = concat %13 %14;
                          var.store %3 %15;
                          java.break;
                      }
                      (%16 : java.lang.Object)boolean -> {
                          %17 : boolean = pattern.match %16
                              ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                  %18 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                  yield %18;
                              }
                              (%19 : java.lang.String)void -> {
                                  var.store %8 %19;
                                  yield;
                              };
                          yield %17;
                      }
                      ()void -> {
                          %20 : java.lang.String = var.load %3;
                          %21 : java.lang.String = constant @"string";
                          %22 : java.lang.String = concat %20 %21;
                          var.store %3 %22;
                          java.break;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %23 : java.lang.String = var.load %3;
                          %24 : java.lang.String = constant @"else";
                          %25 : java.lang.String = concat %23 %24;
                          var.store %3 %25;
                          yield;
                      };
                  %26 : java.lang.String = var.load %3;
                  return %26;
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
            func @"casePatternThrow" (%0 : java.lang.Object)java.lang.String -> {
                %1 : Var<java.lang.Object> = var %0 @"o";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.Object = var.load %1;
                %5 : java.lang.Number = constant @null;
                %6 : Var<java.lang.Number> = var %5 @"n";
                %7 : java.lang.String = constant @null;
                %8 : Var<java.lang.String> = var %7 @"s";
                java.switch.statement %4
                    (%9 : java.lang.Object)boolean -> {
                        %10 : boolean = pattern.match %9
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> -> {
                                %11 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                yield %11;
                            }
                            (%12 : java.lang.Number)void -> {
                                var.store %6 %12;
                                yield;
                            };
                        yield %10;
                    }
                    ()void -> {
                        %13 : java.lang.IllegalArgumentException = new @"func<java.lang.IllegalArgumentException>";
                        throw %13;
                    }
                    (%14 : java.lang.Object)boolean -> {
                        %15 : boolean = pattern.match %14
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                %16 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                yield %16;
                            }
                            (%17 : java.lang.String)void -> {
                                var.store %8 %17;
                                yield;
                            };
                        yield %15;
                    }
                    ()void -> {
                        %18 : java.lang.String = var.load %3;
                        %19 : java.lang.String = constant @"a string";
                        %20 : java.lang.String = concat %18 %19;
                        var.store %3 %20;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %21 : java.lang.String = var.load %3;
                        %22 : java.lang.Object = var.load %1;
                        %23 : java.lang.Class<+<java.lang.Object>> = invoke %22 @"java.lang.Object::getClass()java.lang.Class";
                        %24 : java.lang.String = invoke %23 @"java.lang.Class::getName()java.lang.String";
                        %25 : java.lang.String = concat %21 %24;
                        var.store %3 %25;
                        yield;
                    };
                %26 : java.lang.String = var.load %3;
                return %26;
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
            func @"casePatternWithCaseConstant" (%0 : java.lang.Integer)java.lang.String -> {
                  %1 : Var<java.lang.Integer> = var %0 @"a";
                  %2 : java.lang.String = constant @"";
                  %3 : Var<java.lang.String> = var %2 @"r";
                  %4 : java.lang.Integer = var.load %1;
                  %5 : java.lang.Integer = constant @null;
                  %6 : Var<java.lang.Integer> = var %5 @"i";
                  %7 : java.lang.Integer = constant @null;
                  %8 : Var<java.lang.Integer> = var %7 @"i";
                  java.switch.statement %4
                      (%9 : java.lang.Integer)boolean -> {
                          %10 : int = constant @"42";
                          %11 : java.lang.Integer = invoke %10 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                          %12 : boolean = invoke %9 %11 @"java.util.Objects::equals(java.lang.Object, java.lang.Object)boolean";
                          yield %12;
                      }
                      ()void -> {
                          %13 : java.lang.String = var.load %3;
                          %14 : java.lang.String = constant @"forty two";
                          %15 : java.lang.String = concat %13 %14;
                          var.store %3 %15;
                          yield;
                      }
                      (%16 : java.lang.Integer)boolean -> {
                          %17 : boolean = java.cand
                              ()boolean -> {
                                  %18 : boolean = pattern.match %16
                                      ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> -> {
                                          %19 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> = pattern.binding @"i";
                                          yield %19;
                                      }
                                      (%20 : java.lang.Integer)void -> {
                                          var.store %6 %20;
                                          yield;
                                      };
                                  yield %18;
                              }
                              ()boolean -> {
                                  %21 : java.lang.Integer = var.load %6;
                                  %22 : int = invoke %21 @"java.lang.Integer::intValue()int";
                                  %23 : int = constant @"0";
                                  %24 : boolean = gt %22 %23;
                                  yield %24;
                              };
                          yield %17;
                      }
                      ()void -> {
                          %25 : java.lang.String = var.load %3;
                          %26 : java.lang.String = constant @"positive int";
                          %27 : java.lang.String = concat %25 %26;
                          var.store %3 %27;
                          yield;
                      }
                      (%28 : java.lang.Integer)boolean -> {
                          %29 : boolean = java.cand
                              ()boolean -> {
                                  %30 : boolean = pattern.match %28
                                      ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> -> {
                                          %31 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Integer> = pattern.binding @"i";
                                          yield %31;
                                      }
                                      (%32 : java.lang.Integer)void -> {
                                          var.store %8 %32;
                                          yield;
                                      };
                                  yield %30;
                              }
                              ()boolean -> {
                                  %33 : java.lang.Integer = var.load %8;
                                  %34 : int = invoke %33 @"java.lang.Integer::intValue()int";
                                  %35 : int = constant @"0";
                                  %36 : boolean = lt %34 %35;
                                  yield %36;
                              };
                          yield %29;
                      }
                      ()void -> {
                          %37 : java.lang.String = var.load %3;
                          %38 : java.lang.String = constant @"negative int";
                          %39 : java.lang.String = concat %37 %38;
                          var.store %3 %39;
                          yield;
                      }
                      ()void -> {
                          yield;
                      }
                      ()void -> {
                          %40 : java.lang.String = var.load %3;
                          %41 : java.lang.String = constant @"zero";
                          %42 : java.lang.String = concat %40 %41;
                          var.store %3 %42;
                          yield;
                      };
                  %43 : java.lang.String = var.load %3;
                  return %43;
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
            func @"caseTypePattern" (%0 : java.lang.Object)java.lang.String -> {
                %1 : Var<java.lang.Object> = var %0 @"o";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.Object = var.load %1;
                %5 : java.lang.String = constant @null;
                %6 : Var<java.lang.String> = var %5 @"";
                %7 : java.util.RandomAccess = constant @null;
                %8 : Var<java.util.RandomAccess> = var %7 @"";
                %9 : int[] = constant @null;
                %10 : Var<int[]> = var %9 @"";
                %11 : java.util.Stack[][] = constant @null;
                %12 : Var<java.util.Stack[][]> = var %11 @"";
                %13 : java.util.Collection[][][] = constant @null;
                %14 : Var<java.util.Collection[][][]> = var %13 @"";
                %15 : java.lang.Number = constant @null;
                %16 : Var<java.lang.Number> = var %15 @"n";
                java.switch.statement %4
                    (%17 : java.lang.Object)boolean -> {
                        %18 : boolean = pattern.match %17
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                %19 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"";
                                yield %19;
                            }
                            (%20 : java.lang.String)void -> {
                                var.store %6 %20;
                                yield;
                            };
                        yield %18;
                    }
                    ()void -> {
                        %21 : java.lang.String = var.load %3;
                        %22 : java.lang.String = constant @"String";
                        %23 : java.lang.String = concat %21 %22;
                        var.store %3 %23;
                        yield;
                    }
                    (%24 : java.lang.Object)boolean -> {
                        %25 : boolean = pattern.match %24
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.RandomAccess> -> {
                                %26 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.RandomAccess> = pattern.binding @"";
                                yield %26;
                            }
                            (%27 : java.util.RandomAccess)void -> {
                                var.store %8 %27;
                                yield;
                            };
                        yield %25;
                    }
                    ()void -> {
                        %28 : java.lang.String = var.load %3;
                        %29 : java.lang.String = constant @"RandomAccess";
                        %30 : java.lang.String = concat %28 %29;
                        var.store %3 %30;
                        yield;
                    }
                    (%31 : java.lang.Object)boolean -> {
                        %32 : boolean = pattern.match %31
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<int[]> -> {
                                %33 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<int[]> = pattern.binding @"";
                                yield %33;
                            }
                            (%34 : int[])void -> {
                                var.store %10 %34;
                                yield;
                            };
                        yield %32;
                    }
                    ()void -> {
                        %35 : java.lang.String = var.load %3;
                        %36 : java.lang.String = constant @"int[]";
                        %37 : java.lang.String = concat %35 %36;
                        var.store %3 %37;
                        yield;
                    }
                    (%38 : java.lang.Object)boolean -> {
                        %39 : boolean = pattern.match %38
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.Stack[][]> -> {
                                %40 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.Stack[][]> = pattern.binding @"";
                                yield %40;
                            }
                            (%41 : java.util.Stack[][])void -> {
                                var.store %12 %41;
                                yield;
                            };
                        yield %39;
                    }
                    ()void -> {
                        %42 : java.lang.String = var.load %3;
                        %43 : java.lang.String = constant @"Stack[][]";
                        %44 : java.lang.String = concat %42 %43;
                        var.store %3 %44;
                        yield;
                    }
                    (%45 : java.lang.Object)boolean -> {
                        %46 : boolean = pattern.match %45
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.Collection[][][]> -> {
                                %47 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.util.Collection[][][]> = pattern.binding @"";
                                yield %47;
                            }
                            (%48 : java.util.Collection[][][])void -> {
                                var.store %14 %48;
                                yield;
                            };
                        yield %46;
                    }
                    ()void -> {
                        %49 : java.lang.String = var.load %3;
                        %50 : java.lang.String = constant @"Collection[][][]";
                        %51 : java.lang.String = concat %49 %50;
                        var.store %3 %51;
                        yield;
                    }
                    (%52 : java.lang.Object)boolean -> {
                        %53 : boolean = pattern.match %52
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> -> {
                                %54 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                yield %54;
                            }
                            (%55 : java.lang.Number)void -> {
                                var.store %16 %55;
                                yield;
                            };
                        yield %53;
                    }
                    ()void -> {
                        %56 : java.lang.String = var.load %3;
                        %57 : java.lang.String = constant @"Number";
                        %58 : java.lang.String = concat %56 %57;
                        var.store %3 %58;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %59 : java.lang.String = var.load %3;
                        %60 : java.lang.String = constant @"something else";
                        %61 : java.lang.String = concat %59 %60;
                        var.store %3 %61;
                        yield;
                    };
                %62 : java.lang.String = var.load %3;
                return %62;
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
            func @"caseRecordPattern" (%0 : java.lang.Object)java.lang.String -> {
                %1 : Var<java.lang.Object> = var %0 @"o";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.Object = var.load %1;
                %5 : java.lang.Number = constant @null;
                %6 : Var<java.lang.Number> = var %5 @"n";
                java.switch.statement %4
                    (%7 : java.lang.Object)boolean -> {
                        %8 : boolean = pattern.match %7
                            ()java.lang.reflect.code.ExtendedOp$Pattern$Record<SwitchStatementTest$R> -> {
                                %9 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                %10 : java.lang.reflect.code.ExtendedOp$Pattern$Record<SwitchStatementTest$R> = pattern.record %9 @"(java.lang.Number n)SwitchStatementTest$R";
                                yield %10;
                            }
                            (%11 : java.lang.Number)void -> {
                                var.store %6 %11;
                                yield;
                            };
                        yield %8;
                    }
                    ()void -> {
                        %12 : java.lang.String = var.load %3;
                        %13 : java.lang.String = constant @"R(_)";
                        %14 : java.lang.String = concat %12 %13;
                        var.store %3 %14;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %15 : java.lang.String = var.load %3;
                        %16 : java.lang.String = constant @"else";
                        %17 : java.lang.String = concat %15 %16;
                        var.store %3 %17;
                        yield;
                    };
                %18 : java.lang.String = var.load %3;
                return %18;
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
            func @"casePatternGuard" (%0 : java.lang.Object)java.lang.String -> {
                %1 : Var<java.lang.Object> = var %0 @"obj";
                %2 : java.lang.String = constant @"";
                %3 : Var<java.lang.String> = var %2 @"r";
                %4 : java.lang.Object = var.load %1;
                %5 : java.lang.String = constant @null;
                %6 : Var<java.lang.String> = var %5 @"s";
                %7 : java.lang.Number = constant @null;
                %8 : Var<java.lang.Number> = var %7 @"n";
                java.switch.statement %4
                    (%9 : java.lang.Object)boolean -> {
                        %10 : boolean = java.cand
                            ()boolean -> {
                                %11 : boolean = pattern.match %9
                                    ()java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> -> {
                                        %12 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.String> = pattern.binding @"s";
                                        yield %12;
                                    }
                                    (%13 : java.lang.String)void -> {
                                        var.store %6 %13;
                                        yield;
                                    };
                                yield %11;
                            }
                            ()boolean -> {
                                %14 : java.lang.String = var.load %6;
                                %15 : int = invoke %14 @"java.lang.String::length()int";
                                %16 : int = constant @"3";
                                %17 : boolean = gt %15 %16;
                                yield %17;
                            };
                        yield %10;
                    }
                    ()void -> {
                        %18 : java.lang.String = var.load %3;
                        %19 : java.lang.String = constant @"str with length > %d";
                        %20 : java.lang.String = var.load %6;
                        %21 : int = invoke %20 @"java.lang.String::length()int";
                        %22 : java.lang.Integer = invoke %21 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                        %23 : java.lang.String = invoke %19 %22 @"java.lang.String::formatted(java.lang.Object[])java.lang.String";
                        %24 : java.lang.String = concat %18 %23;
                        var.store %3 %24;
                        yield;
                    }
                    (%25 : java.lang.Object)boolean -> {
                        %26 : boolean = java.cand
                            ()boolean -> {
                                %27 : boolean = pattern.match %25
                                    ()java.lang.reflect.code.ExtendedOp$Pattern$Record<SwitchStatementTest$R> -> {
                                        %28 : java.lang.reflect.code.ExtendedOp$Pattern$Binding<java.lang.Number> = pattern.binding @"n";
                                        %29 : java.lang.reflect.code.ExtendedOp$Pattern$Record<SwitchStatementTest$R> = pattern.record %28 @"(java.lang.Number n)SwitchStatementTest$R";
                                        yield %29;
                                    }
                                    (%30 : java.lang.Number)void -> {
                                        var.store %8 %30;
                                        yield;
                                    };
                                yield %27;
                            }
                            ()boolean -> {
                                %31 : java.lang.Number = var.load %8;
                                %32 : java.lang.Class<+<java.lang.Object>> = invoke %31 @"java.lang.Object::getClass()java.lang.Class";
                                %33 : java.lang.Class = constant @"java.lang.Double";
                                %34 : boolean = invoke %32 %33 @"java.lang.Object::equals(java.lang.Object)boolean";
                                yield %34;
                            };
                        yield %26;
                    }
                    ()void -> {
                        %35 : java.lang.String = var.load %3;
                        %36 : java.lang.String = constant @"R(Double)";
                        %37 : java.lang.String = concat %35 %36;
                        var.store %3 %37;
                        yield;
                    }
                    ()void -> {
                        yield;
                    }
                    ()void -> {
                        %38 : java.lang.String = var.load %3;
                        %39 : java.lang.String = constant @"else";
                        %40 : java.lang.String = concat %38 %39;
                        var.store %3 %40;
                        yield;
                    };
                %41 : java.lang.String = var.load %3;
                return %41;
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
}
