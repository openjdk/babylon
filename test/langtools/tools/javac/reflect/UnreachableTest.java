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

import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.lang.runtime.CodeReflection;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @summary Smoke test for code reflection with unreachable areas.
 * @enablePreview
 * @build UnreachableTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester UnreachableTest
 */

public class UnreachableTest {

    @CodeReflection
    @IR("""
            func @"test1" ()void -> {
                java.block ()void -> {
                    return;
                };
                unreachable;
            };
            """)
    static void test1() {
        {
            return;
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : int)int -> {
                %1 : Var<int> = var %0 @"i";
                java.block ()void -> {
                    %2 : int = var.load %1;
                    return %2;
                };
                unreachable;
            };
            """)
    static int test2(int i) {
        {
            return i;
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : int)int -> {
                %1 : Var<int> = var %0 @"i";
                java.if
                    ()boolean -> {
                        %2 : boolean = constant @"true";
                        yield %2;
                    }
                    ()void -> {
                        %3 : int = var.load %1;
                        return %3;
                    }
                    ()void -> {
                        %4 : int = var.load %1;
                        return %4;
                    };
                unreachable;
            };
            """)
    static int test3(int i) {
        if (true) {
            return i;
        } else {
            return i;
        }
    }


    @CodeReflection
    @IR("""
            func @"test4" ()void -> {
                %0 : java.util.function.IntUnaryOperator = lambda (%1 : int)int -> {
                    %2 : Var<int> = var %1 @"i";
                    java.if
                        ()boolean -> {
                            %3 : boolean = constant @"true";
                            yield %3;
                        }
                        ()void -> {
                            %4 : int = var.load %2;
                            return %4;
                        }
                        ()void -> {
                            %5 : int = var.load %2;
                            return %5;
                        };
                    unreachable;
                };
                %6 : Var<java.util.function.IntUnaryOperator> = var %0 @"f";
                return;
            };
            """)
    static void test4() {
        IntUnaryOperator f = (int i) -> {
            if (true) {
                return i;
            } else {
                return i;
            }
        };
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : int)void -> {
                %1 : Var<int> = var %0 @"n";
                %2 : int = var.load %1;
                %3 : java.lang.String = java.switch.expression %2
                    (%4 : int)boolean -> {
                        %5 : int = constant @"42";
                        %6 : boolean = eq %4 %5;
                        yield %6;
                    }
                    ()java.lang.String -> {
                        java.while
                            ()boolean -> {
                                %7 : boolean = constant @"true";
                                yield %7;
                            }
                            ()void -> {
                                java.continue;
                            };
                        unreachable;
                    }
                    ()boolean -> {
                        %8 : boolean = constant @"true";
                        yield %8;
                    }
                    ()java.lang.String -> {
                        %9 : java.lang.String = constant @"";
                        yield %9;
                    };
                %10 : Var<java.lang.String> = var %3 @"s";
                return;
            };
            """)
    static void test5(int n) {
        String s = switch (n) {
            case 42 -> { while (true); }
            default -> "";
        };
    }

    @IR("""
            func @"f" ()void -> {
                %1 : java.util.function.IntUnaryOperator = lambda (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"i";
                    java.if
                        ()boolean -> {
                            %4 : boolean = constant @"true";
                            yield %4;
                        }
                        ()void -> {
                            %5 : int = var.load %3;
                            return %5;
                        }
                        ()void -> {
                            %6 : int = var.load %3;
                            return %6;
                        };
                    unreachable;
                };
                return;
            };
            """)
    static final Quotable QUOTABLE_TEST = (IntUnaryOperator & Quotable) (int i) -> {
        if (true) {
            return i;
        } else {
            return i;
        }
    };

    @IR("""
            func @"f" ()void -> {
                %1 : func<int, int> = closure (%2 : int)int -> {
                    %3 : Var<int> = var %2 @"i";
                    java.if
                        ()boolean -> {
                            %4 : boolean = constant @"true";
                            yield %4;
                        }
                        ()void -> {
                            %5 : int = var.load %3;
                            return %5;
                        }
                        ()void -> {
                            %6 : int = var.load %3;
                            return %6;
                        };
                    unreachable;
                };
                return;
            };
            """)
    static final Quoted QUOTED_TEST = (int i) -> {
        if (true) {
            return i;
        } else {
            return i;
        }
    };
}
