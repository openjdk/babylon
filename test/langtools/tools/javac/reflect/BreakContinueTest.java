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
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with for loops.
 * @build BreakContinueTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BreakContinueTest
 */


public class BreakContinueTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : BreakContinueTest)void -> {
                java.for
                    ^init()Var<int> -> {
                        %1 : int = constant @"0";
                        %2 : Var<int> = var %1 @"i";
                        yield %2;
                    }
                    ^cond(%3 : Var<int>)boolean -> {
                        %4 : int = var.load %3;
                        %5 : int = constant @"10";
                        %6 : boolean = lt %4 %5;
                        yield %6;
                    }
                    ^update(%7 : Var<int>)void -> {
                        %8 : int = var.load %7;
                        %9 : int = constant @"1";
                        %10 : int = add %8 %9;
                        var.store %7 %10;
                        yield;
                    }
                    ^body(%11 : Var<int>)void -> {
                        java.if
                            ()boolean -> {
                                %12 : boolean = constant @"true";
                                yield %12;
                            }
                            ^then()void -> {
                                java.continue;
                            }
                            ^else()void -> {
                                yield;
                            };
                        java.if
                            ()boolean -> {
                                %13 : boolean = constant @"true";
                                yield %13;
                            }
                            ^then()void -> {
                                java.break;
                            }
                            ^else()void -> {
                                yield;
                            };
                        java.for
                            ^init()Var<int> -> {
                                %14 : int = constant @"0";
                                %15 : Var<int> = var %14 @"j";
                                yield %15;
                            }
                            ^cond(%16 : Var<int>)boolean -> {
                                %17 : int = var.load %16;
                                %18 : int = constant @"10";
                                %19 : boolean = lt %17 %18;
                                yield %19;
                            }
                            ^update(%20 : Var<int>)void -> {
                                %21 : int = var.load %20;
                                %22 : int = constant @"1";
                                %23 : int = add %21 %22;
                                var.store %20 %23;
                                yield;
                            }
                            ^body(%24 : Var<int>)void -> {
                                java.if
                                    ()boolean -> {
                                        %25 : boolean = constant @"true";
                                        yield %25;
                                    }
                                    ^then()void -> {
                                        java.continue;
                                    }
                                    ^else()void -> {
                                        yield;
                                    };
                                java.if
                                    ()boolean -> {
                                        %26 : boolean = constant @"true";
                                        yield %26;
                                    }
                                    ^then()void -> {
                                        java.break;
                                    }
                                    ^else()void -> {
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
            func @"test2" (%0 : BreakContinueTest)void -> {
                java.labeled ()void -> {
                    %1 : java.lang.String = constant @"outer";
                    java.for
                        ^init()Var<int> -> {
                            %2 : int = constant @"0";
                            %3 : Var<int> = var %2 @"i";
                            yield %3;
                        }
                        ^cond(%4 : Var<int>)boolean -> {
                            %5 : int = var.load %4;
                            %6 : int = constant @"10";
                            %7 : boolean = lt %5 %6;
                            yield %7;
                        }
                        ^update(%8 : Var<int>)void -> {
                            %9 : int = var.load %8;
                            %10 : int = constant @"1";
                            %11 : int = add %9 %10;
                            var.store %8 %11;
                            yield;
                        }
                        ^body(%12 : Var<int>)void -> {
                            java.if
                                ()boolean -> {
                                    %13 : boolean = constant @"true";
                                    yield %13;
                                }
                                ^then()void -> {
                                    java.continue %1;
                                }
                                ^else()void -> {
                                    yield;
                                };
                            java.if
                                ()boolean -> {
                                    %14 : boolean = constant @"true";
                                    yield %14;
                                }
                                ^then()void -> {
                                    java.break %1;
                                }
                                ^else()void -> {
                                    yield;
                                };
                            java.labeled ()void -> {
                                %15 : java.lang.String = constant @"inner";
                                java.for
                                    ^init()Var<int> -> {
                                        %16 : int = constant @"0";
                                        %17 : Var<int> = var %16 @"j";
                                        yield %17;
                                    }
                                    ^cond(%18 : Var<int>)boolean -> {
                                        %19 : int = var.load %18;
                                        %20 : int = constant @"10";
                                        %21 : boolean = lt %19 %20;
                                        yield %21;
                                    }
                                    ^update(%22 : Var<int>)void -> {
                                        %23 : int = var.load %22;
                                        %24 : int = constant @"1";
                                        %25 : int = add %23 %24;
                                        var.store %22 %25;
                                        yield;
                                    }
                                    ^body(%26 : Var<int>)void -> {
                                        java.if
                                            ()boolean -> {
                                                %27 : boolean = constant @"true";
                                                yield %27;
                                            }
                                            ^then()void -> {
                                                java.continue;
                                            }
                                            ^else()void -> {
                                                yield;
                                            };
                                        java.if
                                            ()boolean -> {
                                                %28 : boolean = constant @"true";
                                                yield %28;
                                            }
                                            ^then()void -> {
                                                java.break;
                                            }
                                            ^else()void -> {
                                                yield;
                                            };
                                        java.if
                                            ()boolean -> {
                                                %29 : boolean = constant @"true";
                                                yield %29;
                                            }
                                            ^then()void -> {
                                                java.continue %1;
                                            }
                                            ^else()void -> {
                                                yield;
                                            };
                                        java.if
                                            ()boolean -> {
                                                %30 : boolean = constant @"true";
                                                yield %30;
                                            }
                                            ^then()void -> {
                                                java.break %1;
                                            }
                                            ^else()void -> {
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
            func @"test3" (%0 : BreakContinueTest)void -> {
                java.labeled ()void -> {
                    %1 : java.lang.String = constant @"b1";
                    java.block ()void -> {
                        java.labeled ()void -> {
                            %2 : java.lang.String = constant @"b2";
                            java.block ()void -> {
                                java.if
                                    ()boolean -> {
                                        %3 : boolean = constant @"true";
                                        yield %3;
                                    }
                                    ^then()void -> {
                                        java.break %1;
                                    }
                                    ^else()void -> {
                                        yield;
                                    };
                                java.if
                                    ()boolean -> {
                                        %4 : boolean = constant @"true";
                                        yield %4;
                                    }
                                    ^then()void -> {
                                        java.break %2;
                                    }
                                    ^else()void -> {
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
            func @"test4" (%0 : BreakContinueTest)void -> {
                java.labeled ()void -> {
                    %1 : java.lang.String = constant @"b";
                    java.break %1;
                };
                %2 : int = constant @"0";
                %3 : Var<int> = var %2 @"i";
                java.labeled ()void -> {
                    %4 : java.lang.String = constant @"b";
                    %5 : int = var.load %3;
                    %6 : int = constant @"1";
                    %7 : int = add %5 %6;
                    var.store %3 %7;
                    yield;
                };
                java.labeled ()void -> {
                    %8 : java.lang.String = constant @"a";
                    java.labeled ()void -> {
                        %9 : java.lang.String = constant @"b";
                        java.block ()void -> {
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
