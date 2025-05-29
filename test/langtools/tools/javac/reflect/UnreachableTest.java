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

import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.CodeReflection;
import java.util.function.IntUnaryOperator;

/*
 * @test
 * @summary Smoke test for code reflection with unreachable areas.
 * @modules jdk.incubator.code
 * @enablePreview
 * @build UnreachableTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester UnreachableTest
 */

public class UnreachableTest {

    @CodeReflection
    @IR("""
            func @"test1" ()java.type:"void" -> {
                java.block ()java.type:"void" -> {
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
            func @"test2" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                java.block ()java.type:"void" -> {
                    %2 : java.type:"int" = var.load %1;
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
            func @"test3" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                java.if
                    ()java.type:"boolean" -> {
                        %2 : java.type:"boolean" = constant @true;
                        yield %2;
                    }
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = var.load %1;
                        return %3;
                    }
                    ()java.type:"void" -> {
                        %4 : java.type:"int" = var.load %1;
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
            func @"test4" ()java.type:"void" -> {
                %0 : java.type:"java.util.function.IntUnaryOperator" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"i";
                    java.if
                        ()java.type:"boolean" -> {
                            %3 : java.type:"boolean" = constant @true;
                            yield %3;
                        }
                        ()java.type:"void" -> {
                            %4 : java.type:"int" = var.load %2;
                            return %4;
                        }
                        ()java.type:"void" -> {
                            %5 : java.type:"int" = var.load %2;
                            return %5;
                        };
                    unreachable;
                };
                %6 : Var<java.type:"java.util.function.IntUnaryOperator"> = var %0 @"f";
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
            func @"test5" (%0 : java.type:"int")java.type:"void" -> {
                %1 : Var<java.type:"int"> = var %0 @"n";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"java.lang.String" = java.switch.expression %2
                    (%4 : java.type:"int")java.type:"boolean" -> {
                        %5 : java.type:"int" = constant @42;
                        %6 : java.type:"boolean" = eq %4 %5;
                        yield %6;
                    }
                    ()java.type:"java.lang.String" -> {
                        java.while
                            ()java.type:"boolean" -> {
                                %7 : java.type:"boolean" = constant @true;
                                yield %7;
                            }
                            ()java.type:"void" -> {
                                java.continue;
                            };
                        unreachable;
                    }
                    ()java.type:"boolean" -> {
                        %8 : java.type:"boolean" = constant @true;
                        yield %8;
                    }
                    ()java.type:"java.lang.String" -> {
                        %9 : java.type:"java.lang.String" = constant @"";
                        yield %9;
                    };
                %10 : Var<java.type:"java.lang.String"> = var %3 @"s";
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
            func @"f" ()java.type:"void" -> {
                %0 : java.type:"java.util.function.IntUnaryOperator" = lambda (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"i";
                    java.if
                        ()java.type:"boolean" -> {
                            %3 : java.type:"boolean" = constant @true;
                            yield %3;
                        }
                        ()java.type:"void" -> {
                            %4 : java.type:"int" = var.load %2;
                            return %4;
                        }
                        ()java.type:"void" -> {
                            %5 : java.type:"int" = var.load %2;
                            return %5;
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
            func @"f" ()java.type:"void" -> {
                %0 : func<java.type:"int", java.type:"int"> = closure (%1 : java.type:"int")java.type:"int" -> {
                    %2 : Var<java.type:"int"> = var %1 @"i";
                    java.if
                        ()java.type:"boolean" -> {
                            %3 : java.type:"boolean" = constant @true;
                            yield %3;
                        }
                        ()java.type:"void" -> {
                            %4 : java.type:"int" = var.load %2;
                            return %4;
                        }
                        ()java.type:"void" -> {
                            %5 : java.type:"int" = var.load %2;
                            return %5;
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
