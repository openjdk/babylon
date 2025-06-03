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

import jdk.incubator.code.CodeReflection;
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with for loops.
 * @modules jdk.incubator.code
 * @build BreakContinueTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BreakContinueTest
 */


public class BreakContinueTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"BreakContinueTest")java.type:"void" -> {
                java.for
                    ()Var<java.type:"int"> -> {
                        %1 : java.type:"int" = constant @0;
                        %2 : Var<java.type:"int"> = var %1 @"i";
                        yield %2;
                    }
                    (%3 : Var<java.type:"int">)java.type:"boolean" -> {
                        %4 : java.type:"int" = var.load %3;
                        %5 : java.type:"int" = constant @10;
                        %6 : java.type:"boolean" = lt %4 %5;
                        yield %6;
                    }
                    (%7 : Var<java.type:"int">)java.type:"void" -> {
                        %8 : java.type:"int" = var.load %7;
                        %9 : java.type:"int" = constant @1;
                        %10 : java.type:"int" = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    (%11 : Var<java.type:"int">)java.type:"void" -> {
                        java.if
                            ()java.type:"boolean" -> {
                                %12 : java.type:"boolean" = constant @true;
                                yield %12;
                            }
                            ()java.type:"void" -> {
                                java.continue;
                            }
                            ()java.type:"void" -> {
                                yield;
                            };
                        java.if
                            ()java.type:"boolean" -> {
                                %13 : java.type:"boolean" = constant @true;
                                yield %13;
                            }
                            ()java.type:"void" -> {
                                java.break;
                            }
                            ()java.type:"void" -> {
                                yield;
                            };
                        java.for
                            ()Var<java.type:"int"> -> {
                                %14 : java.type:"int" = constant @0;
                                %15 : Var<java.type:"int"> = var %14 @"j";
                                yield %15;
                            }
                            (%16 : Var<java.type:"int">)java.type:"boolean" -> {
                                %17 : java.type:"int" = var.load %16;
                                %18 : java.type:"int" = constant @10;
                                %19 : java.type:"boolean" = lt %17 %18;
                                yield %19;
                            }
                            (%20 : Var<java.type:"int">)java.type:"void" -> {
                                %21 : java.type:"int" = var.load %20;
                                %22 : java.type:"int" = constant @1;
                                %23 : java.type:"int" = add %21 %22;
                                var.store %20 %23;
                                yield;
                            }
                            (%24 : Var<java.type:"int">)java.type:"void" -> {
                                java.if
                                    ()java.type:"boolean" -> {
                                        %25 : java.type:"boolean" = constant @true;
                                        yield %25;
                                    }
                                    ()java.type:"void" -> {
                                        java.continue;
                                    }
                                    ()java.type:"void" -> {
                                        yield;
                                    };
                                java.if
                                    ()java.type:"boolean" -> {
                                        %26 : java.type:"boolean" = constant @true;
                                        yield %26;
                                    }
                                    ()java.type:"void" -> {
                                        java.break;
                                    }
                                    ()java.type:"void" -> {
                                        yield;
                                    };
                                java.continue;
                            };
                        java.continue;
                    };
                return;
            };
            """)
    void test1() {
        for (int i = 0; i < 10; i++) {
            if (true) {
                continue;
            }
            if (true) {
                break;
            }
            for (int j = 0; j < 10; j++) {
                if (true) {
                    continue;
                }
                if (true) {
                    break;
                }
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"BreakContinueTest")java.type:"void" -> {
                java.labeled ()java.type:"void" -> {
                    %1 : java.type:"java.lang.String" = constant @"outer";
                    java.for
                        ()Var<java.type:"int"> -> {
                            %2 : java.type:"int" = constant @0;
                            %3 : Var<java.type:"int"> = var %2 @"i";
                            yield %3;
                        }
                        (%4 : Var<java.type:"int">)java.type:"boolean" -> {
                            %5 : java.type:"int" = var.load %4;
                            %6 : java.type:"int" = constant @10;
                            %7 : java.type:"boolean" = lt %5 %6;
                            yield %7;
                        }
                        (%8 : Var<java.type:"int">)java.type:"void" -> {
                            %9 : java.type:"int" = var.load %8;
                            %10 : java.type:"int" = constant @1;
                            %11 : java.type:"int" = add %9 %10;
                            var.store %8 %11;
                            yield;
                        }
                        (%12 : Var<java.type:"int">)java.type:"void" -> {
                            java.if
                                ()java.type:"boolean" -> {
                                    %13 : java.type:"boolean" = constant @true;
                                    yield %13;
                                }
                                ()java.type:"void" -> {
                                    java.continue %1;
                                }
                                ()java.type:"void" -> {
                                    yield;
                                };
                            java.if
                                ()java.type:"boolean" -> {
                                    %14 : java.type:"boolean" = constant @true;
                                    yield %14;
                                }
                                ()java.type:"void" -> {
                                    java.break %1;
                                }
                                ()java.type:"void" -> {
                                    yield;
                                };
                            java.labeled ()java.type:"void" -> {
                                %15 : java.type:"java.lang.String" = constant @"inner";
                                java.for
                                    ()Var<java.type:"int"> -> {
                                        %16 : java.type:"int" = constant @0;
                                        %17 : Var<java.type:"int"> = var %16 @"j";
                                        yield %17;
                                    }
                                    (%18 : Var<java.type:"int">)java.type:"boolean" -> {
                                        %19 : java.type:"int" = var.load %18;
                                        %20 : java.type:"int" = constant @10;
                                        %21 : java.type:"boolean" = lt %19 %20;
                                        yield %21;
                                    }
                                    (%22 : Var<java.type:"int">)java.type:"void" -> {
                                        %23 : java.type:"int" = var.load %22;
                                        %24 : java.type:"int" = constant @1;
                                        %25 : java.type:"int" = add %23 %24;
                                        var.store %22 %25;
                                        yield;
                                    }
                                    (%26 : Var<java.type:"int">)java.type:"void" -> {
                                        java.if
                                            ()java.type:"boolean" -> {
                                                %27 : java.type:"boolean" = constant @true;
                                                yield %27;
                                            }
                                            ()java.type:"void" -> {
                                                java.continue;
                                            }
                                            ()java.type:"void" -> {
                                                yield;
                                            };
                                        java.if
                                            ()java.type:"boolean" -> {
                                                %28 : java.type:"boolean" = constant @true;
                                                yield %28;
                                            }
                                            ()java.type:"void" -> {
                                                java.break;
                                            }
                                            ()java.type:"void" -> {
                                                yield;
                                            };
                                        java.if
                                            ()java.type:"boolean" -> {
                                                %29 : java.type:"boolean" = constant @true;
                                                yield %29;
                                            }
                                            ()java.type:"void" -> {
                                                java.continue %1;
                                            }
                                            ()java.type:"void" -> {
                                                yield;
                                            };
                                        java.if
                                            ()java.type:"boolean" -> {
                                                %30 : java.type:"boolean" = constant @true;
                                                yield %30;
                                            }
                                            ()java.type:"void" -> {
                                                java.break %1;
                                            }
                                            ()java.type:"void" -> {
                                                yield;
                                            };
                                        java.continue;
                                    };
                                yield;
                            };
                            java.continue;
                        };
                    yield;
                };
                return;
            };
            """)
    void test2() {
        outer:
        for (int i = 0; i < 10; i++) {
            if (true) {
                continue outer;
            }
            if (true) {
                break outer;
            }
            inner:
            for (int j = 0; j < 10; j++) {
                if (true) {
                    continue;
                }
                if (true) {
                    break;
                }
                if (true) {
                    continue outer;
                }
                if (true) {
                    break outer;
                }
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"BreakContinueTest")java.type:"void" -> {
                java.labeled ()java.type:"void" -> {
                    %1 : java.type:"java.lang.String" = constant @"b1";
                    java.block ()java.type:"void" -> {
                        java.labeled ()java.type:"void" -> {
                            %2 : java.type:"java.lang.String" = constant @"b2";
                            java.block ()java.type:"void" -> {
                                java.if
                                    ()java.type:"boolean" -> {
                                        %3 : java.type:"boolean" = constant @true;
                                        yield %3;
                                    }
                                    ()java.type:"void" -> {
                                        java.break %1;
                                    }
                                    ()java.type:"void" -> {
                                        yield;
                                    };
                                java.if
                                    ()java.type:"boolean" -> {
                                        %4 : java.type:"boolean" = constant @true;
                                        yield %4;
                                    }
                                    ()java.type:"void" -> {
                                        java.break %2;
                                    }
                                    ()java.type:"void" -> {
                                        yield;
                                    };
                                yield;
                            };
                            yield;
                        };
                        yield;
                    };
                    yield;
                };
                return;
            };
            """)
    void test3() {
        b1:
        {
            b2:
            {
                if (true) {
                    break b1;
                }
                if (true) {
                    break b2;
                }
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"BreakContinueTest")java.type:"void" -> {
                java.labeled ()java.type:"void" -> {
                    %1 : java.type:"java.lang.String" = constant @"b";
                    java.break %1;
                };
                %2 : java.type:"int" = constant @0;
                %3 : Var<java.type:"int"> = var %2 @"i";
                java.labeled ()java.type:"void" -> {
                    %4 : java.type:"java.lang.String" = constant @"b";
                    %5 : java.type:"int" = var.load %3;
                    %6 : java.type:"int" = constant @1;
                    %7 : java.type:"int" = add %5 %6;
                    var.store %3 %7;
                    yield;
                };
                java.labeled ()java.type:"void" -> {
                    %8 : java.type:"java.lang.String" = constant @"a";
                    java.labeled ()java.type:"void" -> {
                        %9 : java.type:"java.lang.String" = constant @"b";
                        java.block ()java.type:"void" -> {
                            yield;
                        };
                        yield;
                    };
                    yield;
                };
                return;
            };
            """)
    void test4() {
        b:
        break b;

        int i = 0;
        b:
        i++;

        a: b: {
        }
    }
}
