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
import java.util.function.Consumer;

/*
 * @test
 * @summary Smoke test for code reflection with blocks.
 * @build BlockTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester BlockTest
 */

public class BlockTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : BlockTest)void -> {
                java.block ()void -> {
                    %1 : int = constant @"0";
                    %2 : Var<int> = var %1 @"i";
                    yield;
                };
                java.block ()void -> {
                    %3 : int = constant @"1";
                    %4 : Var<int> = var %3 @"i";
                    java.block ()void -> {
                        %5 : int = constant @"2";
                        %6 : Var<int> = var %5 @"j";
                        yield;
                    };
                    yield;
                };
                return;
            };
            """)
    void test1() {
        {
            int i = 0;
        }

        {
            int i = 1;

            {
                int j = 2;
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : BlockTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                java.if
                    ()boolean -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : boolean = lt %3 %4;
                        yield %5;
                    }
                    ^then()void -> {
                        java.block ()void -> {
                            %6 : int = var.load %2;
                            %7 : int = constant @"1";
                            %8 : int = add %6 %7;
                            var.store %2 %8;
                            yield;
                        };
                        yield;
                    }
                    ^else_if()boolean -> {
                        %9 : int = var.load %2;
                        %10 : int = constant @"2";
                        %11 : boolean = lt %9 %10;
                        yield %11;
                    }
                    ^then()void -> {
                        java.block ()void -> {
                            %12 : int = var.load %2;
                            %13 : int = constant @"2";
                            %14 : int = add %12 %13;
                            var.store %2 %14;
                            yield;
                        };
                        yield;
                    }
                    ^else()void -> {
                        java.block ()void -> {
                            %15 : int = var.load %2;
                            %16 : int = constant @"3";
                            %17 : int = add %15 %16;
                            var.store %2 %17;
                            yield;
                        };
                        yield;
                    };
                return;
            };
            """)
    void test2(int i) {
        if (i < 1) {
            {
                i += 1;
            }
        } else if (i < 2) {
            {
                i += 2;
            }
        } else {
            {
                i += 3;
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : BlockTest)void -> {
                java.for
                    ^init()void -> {
                        yield;
                    }
                    ^cond()boolean -> {
                        %1 : boolean = constant @"true";
                        yield %1;
                    }
                    ^update()void -> {
                        yield;
                    }
                    ^body()void -> {
                        java.block ()void -> {
                            %2 : int = constant @"0";
                            %3 : Var<int> = var %2 @"i";
                            yield;
                        };
                        java.continue;
                    };
                return;
            };
            """)
    void test3() {
        for (;;) {
            {
                int i = 0;
            }
        }
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : BlockTest, %1 : int[])void -> {
                %2 : Var<int[]> = var %1 @"ia";
                java.enhancedFor
                    ^expr()int[] -> {
                        %3 : int[] = var.load %2;
                        yield %3;
                    }
                    ^def(%4 : int)Var<int> -> {
                        %5 : Var<int> = var %4 @"i";
                        yield %5;
                    }
                    ^body(%6 : Var<int>)void -> {
                        java.block ()void -> {
                            %7 : int = var.load %6;
                            %8 : int = constant @"1";
                            %9 : int = add %7 %8;
                            var.store %6 %9;
                            yield;
                        };
                        java.continue;
                    };
                return;
            };
            """)
    void test4(int[] ia) {
        for (int i : ia) {
            {
                i++;
            }
        }
   }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : BlockTest)void -> {
                %1 : java.util.function.Consumer<java.lang.String> = lambda (%2 : java.lang.String)void -> {
                    %3 : Var<java.lang.String> = var %2 @"s";
                    java.block ()void -> {
                        %4 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %5 : java.lang.String = var.load %3;
                        invoke %4 %5 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                    return;
                };
                %6 : Var<java.util.function.Consumer<java.lang.String>> = var %1 @"c";
                return;
            };
            """)
   void test5() {
        Consumer<String> c = s -> {
            {
                System.out.println(s);
            }
        };
    }


    @CodeReflection
    @IR("""
            func @"test6" (%0 : BlockTest)void -> {
                 java.if
                     ()boolean -> {
                         %1 : boolean = constant @"true";
                         yield %1;
                     }
                     ^then()void -> {
                         java.block ()void -> {
                             return;
                         };
                         yield;
                     }
                     ^else()void -> {
                         yield;
                     };
                 java.if
                     ()boolean -> {
                         %2 : boolean = constant @"true";
                         yield %2;
                     }
                     ^then()void -> {
                         java.block ()void -> {
                             %3 : java.lang.RuntimeException = new @"()java.lang.RuntimeException";
                             throw %3;
                         };
                         yield;
                     }
                     ^else()void -> {
                         yield;
                     };
                 return;
             };
             """)
    void test6() {
        if (true) {
            {
                return;
            }
        }

        if (true) {
            {
                throw new RuntimeException();
            }
        }
    }
}
