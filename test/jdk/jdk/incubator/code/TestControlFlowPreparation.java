/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.reflect.Method;
import java.util.Comparator;
import java.util.stream.Stream;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.internal.ControlFlowPreparation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
/*
 * @test
 * @summary Compare CFP shapes with model expectations
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @library lib
 * @run junit TestControlFlowPreparation
 */
public class TestControlFlowPreparation {

    static Stream<Method> testMethods() {
        return Stream.of(TestControlFlowPreparation.class.getDeclaredMethods())
                .filter(m -> m.isAnnotationPresent(Reflect.class))
                .sorted(Comparator.comparingInt(m -> Integer.parseInt(m.getName().substring(4))));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("testMethods")
    void testCFP(Method method) {
        var prepared = ControlFlowPreparation.transform(Op.ofMethod(method).orElseThrow().transform(CodeTransformer.LOWERING_TRANSFORMER));
        Assertions.assertEquals(CodeReflectionTester.canonicalizeModel(method, method.getAnnotation(IR.class).value()),
                                CodeReflectionTester.canonicalizeModel(method, CodeReflectionTester.removeSourceAttribute(prepared)));
    }

    @IR("""
        func @"case1" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @0;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            return %5;
        };
        """)
    @Reflect
    static int case1() {
        if (((false) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case2" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            return %5;
        };
        """)
    @Reflect
    static int case2() {
        if (((true) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case3" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case3() {
        if ((((!false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case4" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @0;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case4() {
        if ((((!true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case5" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @0;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            return %5;
        };
        """)
    @Reflect
    static int case5() {
        if ((((false && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case6" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @0;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            return %5;
        };
        """)
    @Reflect
    static int case6() {
        if ((((false && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case7" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @0;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case7(boolean boolean0) {
        if ((((false && boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case8" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case8() {
        if ((((true && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case9" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case9() {
        if ((((true && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case10" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @0;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case10(boolean boolean0) {
        if ((((boolean0 && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case11" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case11() {
        if ((((false || false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case12" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case12() {
        if ((((false || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case13" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            return %5;
        };
        """)
    @Reflect
    static int case13() {
        if ((((true || false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case14" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            return %5;
        };
        """)
    @Reflect
    static int case14() {
        if ((((true || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case15" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case15(boolean boolean0) {
        if ((((true || boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case16" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_2 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case16(boolean boolean0) {
        if ((((boolean0 || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case17" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case17() {
        if ((((false ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case18" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case18() {
        if ((((false ? false : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case19" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case19() {
        if ((((false ? true : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case20" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case20() {
        if ((((false ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case21" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @0;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case21(boolean boolean0) {
        if ((((false ? boolean0 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case22" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case22(boolean boolean0) {
        if ((((false ? boolean0 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case23" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case23() {
        if ((((true ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case24" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @0;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case24() {
        if ((((true ? false : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case25" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @0;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case25(boolean boolean0) {
        if ((((true ? false : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case26" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case26() {
        if ((((true ? true : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case27" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case27() {
        if ((((true ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case28" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case28(boolean boolean0) {
        if ((((true ? true : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case29" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case29(boolean boolean0) {
        if ((((boolean0 ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case30" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case30(boolean boolean0) {
        if ((((boolean0 ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case31" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            %2 : java.type:"boolean" = not %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @0;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case31() {
        if ((((!(!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case32" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            %2 : java.type:"boolean" = not %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case32() {
        if ((((!(!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case33" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case33(boolean boolean0) {
        if ((((!(boolean0 && false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case34" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_2 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @0;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case34(boolean boolean0) {
        if ((((!(boolean0 || true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case35" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case35() {
        if (((((!false) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case36" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case36() {
        if (((((!false) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case37" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case37(boolean boolean0) {
        if (((((!false) && boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case38" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @0;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case38() {
        if (((((!true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case39" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @0;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            return %6;
        };
        """)
    @Reflect
    static int case39() {
        if (((((!true) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case40" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @0;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_3;

          ^block_3:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case40(boolean boolean0) {
        if (((((!true) && boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case41" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            %3 : java.type:"boolean" = not %2;
            cbranch %3 ^block_1 ^block_2;

          ^block_1:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case41(boolean boolean0) {
        if (((((!boolean0) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case42" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @false;
            %2 : java.type:"boolean" = not %1;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case42() {
        if ((((true && (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case43" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            %2 : java.type:"boolean" = not %1;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case43() {
        if ((((true && (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case44" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case44(boolean boolean0) {
        if ((((boolean0 && (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case45" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case45(boolean boolean0) {
        if ((((boolean0 && (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case46" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @0;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case46(boolean boolean0, boolean boolean1) {
        if (((((false && boolean0) && boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case47" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %3 : java.type:"int" = constant @0;
            branch ^block_4(%3);

          ^block_4(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case47() {
        if (((((true && true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case48" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %3 : java.type:"int" = constant @1;
            branch ^block_4(%3);

          ^block_4(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case48() {
        if (((((true && true) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case49" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_2 ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case49(boolean boolean0) {
        if (((((true && boolean0) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case50" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"int" = constant @0;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case50(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && false) && boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case51" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case51(boolean boolean0) {
        if (((((boolean0 && true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case52" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case52(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case53" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_2 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case53(boolean boolean0) {
        if (((((boolean0 || true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case54" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_2 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case54(boolean boolean0) {
        if (((((boolean0 || true) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case55" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case55(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case56" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_3 ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case56(boolean boolean0) {
        if ((((true && (boolean0 || true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case57" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"int" = constant @0;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case57(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : false) && boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case58" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case58(boolean boolean0) {
        if (((((false ? boolean0 : true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case59" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case59(boolean boolean0) {
        if (((((false ? boolean0 : true) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case60" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case60(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case61" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %2;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case61(boolean boolean0, boolean boolean1) {
        if (((((true ? boolean0 : boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case62" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case62(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : false) && boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case63" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case63(boolean boolean0) {
        if (((((boolean0 ? false : true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case64" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case64(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case65" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case65(boolean boolean0) {
        if (((((boolean0 ? true : false) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case66" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case66(boolean boolean0) {
        if (((((boolean0 ? true : true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case67" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case67(boolean boolean0) {
        if (((((boolean0 ? true : true) && true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case68" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case68(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? true : boolean1) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case69" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case69(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : false) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case70" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_3 ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case70(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : true) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case71" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_1 ^block_2;

          ^block_1:
            %7 : java.type:"boolean" = var.load %4;
            cbranch %7 ^block_3 ^block_4;

          ^block_2:
            %8 : java.type:"boolean" = var.load %5;
            cbranch %8 ^block_3 ^block_4;

          ^block_3:
            %9 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @0;
            branch ^block_5(%10);

          ^block_5(%11 : java.type:"int"):
            %12 : java.type:"int" = constant @1;
            %13 : java.type:"boolean" = eq %11 %12;
            branch ^block_6;

          ^block_6:
            %14 : java.type:"int" = constant @0;
            return %14;
        };
        """)
    @Reflect
    static int case71(boolean boolean0, boolean boolean1, boolean boolean2) {
        if (((((boolean0 ? boolean1 : boolean2) && false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case72" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %7 : java.type:"int" = constant @0;
            branch ^block_2(%7);

          ^block_2(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_3;

          ^block_3:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case72(boolean boolean0, boolean boolean1, boolean boolean2) {
        if ((((false && (boolean0 ? boolean1 : boolean2))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case73" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_2 ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @0;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case73(boolean boolean0) {
        if ((((true && (boolean0 ? false : false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case74" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_2 ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case74(boolean boolean0) {
        if ((((true && (boolean0 ? true : true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case75" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case75(boolean boolean0, boolean boolean1) {
        if ((((boolean0 && (false ? boolean1 : false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case76" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_4;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case76(boolean boolean0, boolean boolean1) {
        if ((((boolean0 && (boolean1 ? false : false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case77" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case77() {
        if (((((!false) || false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case78" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            return %6;
        };
        """)
    @Reflect
    static int case78() {
        if (((((!false) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case79" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_3;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case79(boolean boolean0) {
        if (((((!false) || boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case80" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case80() {
        if (((((!true) || false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case81" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case81() {
        if (((((!true) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case82" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case82(boolean boolean0) {
        if (((((!true) || boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case83" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            %3 : java.type:"boolean" = not %2;
            cbranch %3 ^block_2 ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case83(boolean boolean0) {
        if (((((!boolean0) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case84" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_2 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case84(boolean boolean0) {
        if ((((boolean0 || (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case85" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_4(%3);

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            %5 : java.type:"boolean" = not %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case85(boolean boolean0) {
        if ((((boolean0 || (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case86" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case86(boolean boolean0) {
        if (((((boolean0 && false) || false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case87" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case87(boolean boolean0) {
        if (((((boolean0 && false) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case88" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_3 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case88(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case89" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case89(boolean boolean0, boolean boolean1) {
        if ((((true || (boolean0 && boolean1))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case90" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_3 ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case90(boolean boolean0) {
        if ((((boolean0 || (true && true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case91" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"int" = constant @1;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case91(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || true) || boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case92" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_3 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_3 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case92(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case93" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"int" = constant @1;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case93(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : true) || boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case94" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_3 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case94(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case95" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %2;
            cbranch %5 ^block_3 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case95(boolean boolean0, boolean boolean1) {
        if (((((true ? boolean0 : boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case96" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case96(boolean boolean0) {
        if (((((boolean0 ? false : true) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case97" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_4 ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case97(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case98" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            branch ^block_5(%6);

          ^block_5(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_6;

          ^block_6:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case98(boolean boolean0) {
        if (((((boolean0 ? true : false) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case99" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case99(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? true : true) || boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case100" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_4 ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case100(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? true : boolean1) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case101" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_4 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case101(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : false) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case102" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_4 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case102(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : true) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case103" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_1 ^block_2;

          ^block_1:
            %7 : java.type:"boolean" = var.load %4;
            cbranch %7 ^block_4 ^block_3;

          ^block_2:
            %8 : java.type:"boolean" = var.load %5;
            cbranch %8 ^block_4 ^block_3;

          ^block_3:
            %9 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @1;
            branch ^block_5(%10);

          ^block_5(%11 : java.type:"int"):
            %12 : java.type:"int" = constant @1;
            %13 : java.type:"boolean" = eq %11 %12;
            branch ^block_6;

          ^block_6:
            %14 : java.type:"int" = constant @1;
            return %14;
        };
        """)
    @Reflect
    static int case103(boolean boolean0, boolean boolean1, boolean boolean2) {
        if (((((boolean0 ? boolean1 : boolean2) || true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case104" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %7 : java.type:"int" = constant @1;
            branch ^block_2(%7);

          ^block_2(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_3;

          ^block_3:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case104(boolean boolean0, boolean boolean1, boolean boolean2) {
        if ((((true || (boolean0 ? boolean1 : boolean2))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case105" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_3 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case105(boolean boolean0, boolean boolean1) {
        if ((((boolean0 || (false ? boolean1 : true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case106" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_4 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case106(boolean boolean0, boolean boolean1) {
        if ((((boolean0 || (boolean1 ? true : true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case107" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case107() {
        if (((((!false) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case108" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case108() {
        if (((((!false) ? false : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case109" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case109(boolean boolean0) {
        if (((((!false) ? false : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case110" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case110() {
        if (((((!false) ? true : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case111" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case111() {
        if (((((!false) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case112" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case112(boolean boolean0) {
        if (((((!false) ? true : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case113" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case113(boolean boolean0) {
        if (((((!false) ? boolean0 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case114" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @false;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case114(boolean boolean0) {
        if (((((!false) ? boolean0 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case115" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            %5 : java.type:"boolean" = not %4;
            branch ^block_1;

          ^block_1:
            %6 : java.type:"boolean" = var.load %2;
            cbranch %6 ^block_2 ^block_3;

          ^block_2:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_3:
            %8 : java.type:"int" = constant @0;
            branch ^block_4(%8);

          ^block_4(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            cbranch %11 ^block_5 ^block_6;

          ^block_5:
            %12 : java.type:"int" = constant @1;
            return %12;

          ^block_6:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case115(boolean boolean0, boolean boolean1) {
        if (((((!false) ? boolean0 : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case116" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case116() {
        if (((((!true) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case117" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case117() {
        if (((((!true) ? false : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case118" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case118(boolean boolean0) {
        if (((((!true) ? false : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case119" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @0;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case119() {
        if (((((!true) ? true : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case120" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = not %0;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_3(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case120() {
        if (((((!true) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case121" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = var.load %1;
            cbranch %4 ^block_2 ^block_3;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            cbranch %9 ^block_5 ^block_6;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;

          ^block_6:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case121(boolean boolean0) {
        if (((((!true) ? true : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case122" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case122(boolean boolean0) {
        if (((((!true) ? boolean0 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case123" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            %3 : java.type:"boolean" = not %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case123(boolean boolean0) {
        if (((((!true) ? boolean0 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case124" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            %5 : java.type:"boolean" = not %4;
            branch ^block_1;

          ^block_1:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_2 ^block_3;

          ^block_2:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_3:
            %8 : java.type:"int" = constant @0;
            branch ^block_4(%8);

          ^block_4(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            cbranch %11 ^block_5 ^block_6;

          ^block_5:
            %12 : java.type:"int" = constant @1;
            return %12;

          ^block_6:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case124(boolean boolean0, boolean boolean1) {
        if (((((!true) ? boolean0 : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case125" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            %3 : java.type:"boolean" = not %2;
            cbranch %3 ^block_1 ^block_2;

          ^block_1:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case125(boolean boolean0) {
        if (((((!boolean0) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case126" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            %3 : java.type:"boolean" = not %2;
            cbranch %3 ^block_1 ^block_2;

          ^block_1:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case126(boolean boolean0) {
        if (((((!boolean0) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case127" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case127(boolean boolean0) {
        if ((((true ? (!false) : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case128" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_3(%5);

          ^block_3(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case128(boolean boolean0) {
        if ((((true ? (!true) : boolean0)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case129" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @1;
            branch ^block_5(%5);

          ^block_3:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            branch ^block_5(%7);

          ^block_5(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            cbranch %10 ^block_6 ^block_7;

          ^block_6:
            %11 : java.type:"int" = constant @1;
            return %11;

          ^block_7:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case129(boolean boolean0) {
        if ((((boolean0 ? (!false) : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case130" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = not %3;
            branch ^block_3;

          ^block_2:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case130(boolean boolean0) {
        if ((((boolean0 ? (!false) : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case131" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            %6 : java.type:"boolean" = not %5;
            branch ^block_3;

          ^block_2:
            %7 : java.type:"boolean" = var.load %3;
            cbranch %7 ^block_3 ^block_4;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_4:
            %9 : java.type:"int" = constant @0;
            branch ^block_5(%9);

          ^block_5(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            cbranch %12 ^block_6 ^block_7;

          ^block_6:
            %13 : java.type:"int" = constant @1;
            return %13;

          ^block_7:
            %14 : java.type:"int" = constant @0;
            return %14;
        };
        """)
    @Reflect
    static int case131(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (!false) : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case132" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = not %3;
            branch ^block_3;

          ^block_2:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case132(boolean boolean0) {
        if ((((boolean0 ? (!true) : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case133" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = not %3;
            branch ^block_2;

          ^block_2:
            %5 : java.type:"int" = constant @0;
            branch ^block_5(%5);

          ^block_3:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            branch ^block_5(%7);

          ^block_5(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            cbranch %10 ^block_6 ^block_7;

          ^block_6:
            %11 : java.type:"int" = constant @1;
            return %11;

          ^block_7:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case133(boolean boolean0) {
        if ((((boolean0 ? (!true) : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case134" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            %6 : java.type:"boolean" = not %5;
            branch ^block_4;

          ^block_2:
            %7 : java.type:"boolean" = var.load %3;
            cbranch %7 ^block_3 ^block_4;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_4:
            %9 : java.type:"int" = constant @0;
            branch ^block_5(%9);

          ^block_5(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            cbranch %12 ^block_6 ^block_7;

          ^block_6:
            %13 : java.type:"int" = constant @1;
            return %13;

          ^block_7:
            %14 : java.type:"int" = constant @0;
            return %14;
        };
        """)
    @Reflect
    static int case134(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (!true) : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case135" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @0;
            branch ^block_5(%4);

          ^block_3:
            %5 : java.type:"boolean" = constant @false;
            %6 : java.type:"boolean" = not %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @1;
            branch ^block_5(%7);

          ^block_5(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            cbranch %10 ^block_6 ^block_7;

          ^block_6:
            %11 : java.type:"int" = constant @1;
            return %11;

          ^block_7:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case135(boolean boolean0) {
        if ((((boolean0 ? false : (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case136" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            %5 : java.type:"boolean" = not %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @0;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @0;
            return %10;
        };
        """)
    @Reflect
    static int case136(boolean boolean0) {
        if ((((boolean0 ? false : (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case137" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            %5 : java.type:"boolean" = not %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            branch ^block_4(%6);

          ^block_4(%7 : java.type:"int"):
            %8 : java.type:"int" = constant @1;
            %9 : java.type:"boolean" = eq %7 %8;
            branch ^block_5;

          ^block_5:
            %10 : java.type:"int" = constant @1;
            return %10;
        };
        """)
    @Reflect
    static int case137(boolean boolean0) {
        if ((((boolean0 ? true : (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case138" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_5(%4);

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            %6 : java.type:"boolean" = not %5;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @0;
            branch ^block_5(%7);

          ^block_5(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            cbranch %10 ^block_6 ^block_7;

          ^block_6:
            %11 : java.type:"int" = constant @1;
            return %11;

          ^block_7:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case138(boolean boolean0) {
        if ((((boolean0 ? true : (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case139" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_4 ^block_2;

          ^block_2:
            %6 : java.type:"int" = constant @0;
            branch ^block_5(%6);

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            %8 : java.type:"boolean" = not %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @1;
            branch ^block_5(%9);

          ^block_5(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            cbranch %12 ^block_6 ^block_7;

          ^block_6:
            %13 : java.type:"int" = constant @1;
            return %13;

          ^block_7:
            %14 : java.type:"int" = constant @0;
            return %14;
        };
        """)
    @Reflect
    static int case139(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? boolean1 : (!false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case140" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_4;

          ^block_2:
            %6 : java.type:"int" = constant @1;
            branch ^block_5(%6);

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            %8 : java.type:"boolean" = not %7;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            branch ^block_5(%9);

          ^block_5(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            cbranch %12 ^block_6 ^block_7;

          ^block_6:
            %13 : java.type:"int" = constant @1;
            return %13;

          ^block_7:
            %14 : java.type:"int" = constant @0;
            return %14;
        };
        """)
    @Reflect
    static int case140(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? boolean1 : (!true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case141" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case141(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && false) ? boolean1 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case142" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case142(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && false) ? boolean1 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case143" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case143(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case144" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case144(boolean boolean0, boolean boolean1) {
        if (((((boolean0 && boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case145" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case145(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || true) ? false : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case146" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case146(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || true) ? true : boolean1)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case147" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case147(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case148" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_2 ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case148(boolean boolean0, boolean boolean1) {
        if (((((boolean0 || boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case149" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case149(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : false) ? boolean1 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case150" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case150(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : false) ? boolean1 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case151" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case151(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case152" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case152(boolean boolean0, boolean boolean1) {
        if (((((false ? boolean0 : boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case153" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %2;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case153(boolean boolean0, boolean boolean1) {
        if (((((true ? boolean0 : boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case154" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"boolean" = var.load %2;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case154(boolean boolean0, boolean boolean1) {
        if (((((true ? boolean0 : boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case155" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case155(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : false) ? boolean1 : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case156" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case156(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : false) ? boolean1 : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case157" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = constant @0;
            branch ^block_6(%7);

          ^block_6(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_7;

          ^block_7:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case157(boolean boolean0) {
        if (((((boolean0 ? false : true) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case158" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_3;

          ^block_1:
            %3 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %4 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_3:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = constant @1;
            branch ^block_6(%7);

          ^block_6(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_7;

          ^block_7:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case158(boolean boolean0) {
        if (((((boolean0 ? false : true) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case159" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case159(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case160" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case160(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? false : boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case161" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case161(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? true : boolean1) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case162" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case162(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? true : boolean1) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case163" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case163(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : false) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case164" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case164(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : false) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case165" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_4 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case165(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : true) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case166" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_4 ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case166(boolean boolean0, boolean boolean1) {
        if (((((boolean0 ? boolean1 : true) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case167" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_1 ^block_2;

          ^block_1:
            %7 : java.type:"boolean" = var.load %4;
            cbranch %7 ^block_3 ^block_4;

          ^block_2:
            %8 : java.type:"boolean" = var.load %5;
            cbranch %8 ^block_3 ^block_4;

          ^block_3:
            %9 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_4:
            %10 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            branch ^block_6(%11);

          ^block_6(%12 : java.type:"int"):
            %13 : java.type:"int" = constant @1;
            %14 : java.type:"boolean" = eq %12 %13;
            branch ^block_7;

          ^block_7:
            %15 : java.type:"int" = constant @0;
            return %15;
        };
        """)
    @Reflect
    static int case167(boolean boolean0, boolean boolean1, boolean boolean2) {
        if (((((boolean0 ? boolean1 : boolean2) ? false : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case168" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_1 ^block_2;

          ^block_1:
            %7 : java.type:"boolean" = var.load %4;
            cbranch %7 ^block_3 ^block_4;

          ^block_2:
            %8 : java.type:"boolean" = var.load %5;
            cbranch %8 ^block_3 ^block_4;

          ^block_3:
            %9 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_4:
            %10 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            branch ^block_6(%11);

          ^block_6(%12 : java.type:"int"):
            %13 : java.type:"int" = constant @1;
            %14 : java.type:"boolean" = eq %12 %13;
            branch ^block_7;

          ^block_7:
            %15 : java.type:"int" = constant @1;
            return %15;
        };
        """)
    @Reflect
    static int case168(boolean boolean0, boolean boolean1, boolean boolean2) {
        if (((((boolean0 ? boolean1 : boolean2) ? true : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case169" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %8 : java.type:"int" = constant @0;
            branch ^block_3(%8);

          ^block_3(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_4;

          ^block_4:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case169(boolean boolean0, boolean boolean1, boolean boolean2) {
        if ((((false ? (boolean0 ? boolean1 : boolean2) : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case170" (%0 : java.type:"boolean", %1 : java.type:"boolean", %2 : java.type:"boolean")java.type:"int" -> {
            %3 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %4 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %5 : Var<java.type:"boolean"> = var %2 @"boolean2";
            %6 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %8 : java.type:"int" = constant @1;
            branch ^block_3(%8);

          ^block_3(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_4;

          ^block_4:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case170(boolean boolean0, boolean boolean1, boolean boolean2) {
        if ((((false ? (boolean0 ? boolean1 : boolean2) : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case171" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case171(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (false ? boolean1 : false) : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case172" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_3;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_2;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case172(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (false ? boolean1 : true) : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case173" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_4;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case173(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (boolean1 ? false : false) : false)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case174" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_4;

          ^block_1:
            %5 : java.type:"boolean" = var.load %3;
            cbranch %5 ^block_2 ^block_3;

          ^block_2:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case174(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? (boolean1 ? true : true) : true)) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case175" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @0;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @0;
            return %12;
        };
        """)
    @Reflect
    static int case175(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? false : (false ? boolean1 : false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case176" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case176(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? false : (boolean1 ? false : false))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case177" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_2:
            %6 : java.type:"boolean" = constant @false;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case177(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? true : (false ? boolean1 : true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case178" (%0 : java.type:"boolean", %1 : java.type:"boolean")java.type:"int" -> {
            %2 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %3 : Var<java.type:"boolean"> = var %1 @"boolean1";
            %4 : java.type:"boolean" = var.load %2;
            cbranch %4 ^block_1 ^block_2;

          ^block_1:
            %5 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_2:
            %6 : java.type:"boolean" = var.load %3;
            cbranch %6 ^block_3 ^block_4;

          ^block_3:
            %7 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_4:
            %8 : java.type:"boolean" = constant @true;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            branch ^block_6(%9);

          ^block_6(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_7;

          ^block_7:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case178(boolean boolean0, boolean boolean1) {
        if ((((boolean0 ? true : (boolean1 ? true : true))) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case179" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = pos %0;
            %2 : java.type:"int" = constant @1;
            %3 : java.type:"boolean" = eq %1 %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_3;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case179() {
        if ((((+1) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case180" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @-1;
            %1 : java.type:"int" = constant @0;
            %2 : java.type:"boolean" = lt %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case180() {
        if ((((-1) < 0) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case181" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = compl %0;
            %2 : java.type:"int" = constant @-2;
            %3 : java.type:"boolean" = eq %1 %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_3;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case181() {
        if ((((~1) == -2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case182" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @2;
            %1 : java.type:"int" = constant @3;
            %2 : java.type:"int" = mul %0 %1;
            %3 : java.type:"int" = constant @6;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case182() {
        if ((((2 * 3) == 6) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case183" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @6;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"int" = div %0 %1;
            %3 : java.type:"int" = constant @3;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case183() {
        if ((((6 / 2) == 3) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case184" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @7;
            %1 : java.type:"int" = constant @3;
            %2 : java.type:"int" = mod %0 %1;
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case184() {
        if ((((7 % 3) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case185" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"int" = add %0 %1;
            %3 : java.type:"int" = constant @3;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case185() {
        if ((((1 + 2) == 3) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case186" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @3;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"int" = sub %0 %1;
            %3 : java.type:"int" = constant @2;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case186() {
        if ((((3 - 1) == 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case187" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"int" = lshl %0 %1;
            %3 : java.type:"int" = constant @4;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case187() {
        if ((((1 << 2) == 4) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case188" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @4;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"int" = ashr %0 %1;
            %3 : java.type:"int" = constant @2;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case188() {
        if ((((4 >> 1) == 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case189" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @-1;
            %1 : java.type:"int" = constant @31;
            %2 : java.type:"int" = lshr %0 %1;
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case189() {
        if ((((-1 >>> 31) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case190" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"boolean" = lt %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case190() {
        if (((1 < 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case191" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"boolean" = le %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case191() {
        if (((1 <= 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case192" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @2;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"boolean" = gt %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case192() {
        if (((2 > 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case193" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @2;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"boolean" = ge %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case193() {
        if (((2 >= 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case194" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"boolean" = eq %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case194() {
        if (((1 == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case195" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"boolean" = neq %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case195() {
        if (((1 != 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case196" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @3;
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"int" = and %0 %1;
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case196() {
        if ((((3 & 1) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case197" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @3;
            %2 : java.type:"int" = xor %0 %1;
            %3 : java.type:"int" = constant @2;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case197() {
        if ((((1 ^ 3) == 2) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case198" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"int" = or %0 %1;
            %3 : java.type:"int" = constant @3;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case198() {
        if ((((1 | 2) == 3) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case199" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = constant @false;
            %2 : java.type:"boolean" = and %0 %1;
            %3 : java.type:"boolean" = constant @false;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case199() {
        if ((((true & false) == false) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case200" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = constant @false;
            %2 : java.type:"boolean" = xor %0 %1;
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case200() {
        if ((((true ^ false) == true) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case201" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            %1 : java.type:"boolean" = constant @false;
            %2 : java.type:"boolean" = or %0 %1;
            %3 : java.type:"boolean" = constant @true;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case201() {
        if ((((true | false) == true) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case202" ()java.type:"int" -> {
            %0 : java.type:"double" = constant @1.0d;
            %1 : java.type:"int" = conv %0;
            %2 : java.type:"int" = constant @1;
            %3 : java.type:"boolean" = eq %1 %2;
            branch ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_3;

          ^block_3:
            %8 : java.type:"int" = constant @1;
            return %8;
        };
        """)
    @Reflect
    static int case202() {
        if (((((int) 1.0) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case203" ()java.type:"int" -> {
            %0 : java.type:"java.lang.String" = constant @"a";
            %1 : java.type:"java.lang.String" = constant @"a";
            %2 : java.type:"boolean" = eq %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case203() {
        if (((((String) "a") == "a") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case204" ()java.type:"int" -> {
            %0 : java.type:"java.lang.String" = constant @"a";
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"java.lang.String" = constant @"a1";
            %3 : java.type:"java.lang.String" = constant @"a1";
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_1;

          ^block_1:
            %5 : java.type:"int" = constant @1;
            branch ^block_2(%5);

          ^block_2(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case204() {
        if (((("a" + 1) == "a1") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case205" ()java.type:"int" -> {
            %0 : java.type:"java.lang.String" = constant @"a\\n";
            %1 : java.type:"java.lang.String" = constant @"a\\n";
            %2 : java.type:"boolean" = eq %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case205() {
        if ((("""\u000aa\u000a""" == "a\n") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case206" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @4;
            %1 : java.type:"int" = constant @4;
            %2 : java.type:"boolean" = eq %0 %1;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            return %7;
        };
        """)
    @Reflect
    static int case206() {
        if (((Integer.BYTES == 4) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case207" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"byte" = conv %1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"byte"):
            %4 : java.type:"int" = conv %3;
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"byte" = conv %5;
            %7 : java.type:"int" = conv %6;
            %8 : java.type:"boolean" = eq %4 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            branch ^block_4(%9);

          ^block_4(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_5;

          ^block_5:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case207() {
        if ((((true ? (byte) 1 : (byte) 2) == (byte) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case208" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"byte" = conv %1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"byte"):
            %4 : java.type:"int" = conv %3;
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"byte" = conv %5;
            %7 : java.type:"int" = conv %6;
            %8 : java.type:"boolean" = eq %4 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @0;
            branch ^block_4(%9);

          ^block_4(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_5;

          ^block_5:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case208() {
        if ((((false ? (byte) 1 : (byte) 2) == (byte) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case209" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"byte" = conv %3;
            branch ^block_3(%4);

          ^block_2:
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"byte" = conv %5;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"byte"):
            %8 : java.type:"int" = conv %7;
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"byte" = conv %9;
            %11 : java.type:"int" = conv %10;
            %12 : java.type:"boolean" = eq %8 %11;
            branch ^block_4;

          ^block_4:
            %13 : java.type:"int" = constant @1;
            branch ^block_5(%13);

          ^block_5(%14 : java.type:"int"):
            %15 : java.type:"int" = constant @1;
            %16 : java.type:"boolean" = eq %14 %15;
            branch ^block_6;

          ^block_6:
            %17 : java.type:"int" = constant @1;
            return %17;
        };
        """)
    @Reflect
    static int case209(boolean boolean0) {
        if ((((boolean0 ? (byte) 1 : (byte) 1) == (byte) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case210" (%0 : java.type:"byte")java.type:"int" -> {
            %1 : Var<java.type:"byte"> = var %0 @"byte0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @2;
            %4 : java.type:"byte" = conv %3;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"byte"):
            %6 : java.type:"int" = conv %5;
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"byte" = conv %7;
            %9 : java.type:"int" = conv %8;
            %10 : java.type:"boolean" = eq %6 %9;
            branch ^block_3;

          ^block_3:
            %11 : java.type:"int" = constant @0;
            branch ^block_4(%11);

          ^block_4(%12 : java.type:"int"):
            %13 : java.type:"int" = constant @1;
            %14 : java.type:"boolean" = eq %12 %13;
            branch ^block_5;

          ^block_5:
            %15 : java.type:"int" = constant @0;
            return %15;
        };
        """)
    @Reflect
    static int case210(byte byte0) {
        if ((((false ? byte0 : (byte) 2) == (byte) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case211" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"short" = conv %1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"short"):
            %4 : java.type:"int" = conv %3;
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"short" = conv %5;
            %7 : java.type:"int" = conv %6;
            %8 : java.type:"boolean" = eq %4 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @1;
            branch ^block_4(%9);

          ^block_4(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_5;

          ^block_5:
            %13 : java.type:"int" = constant @1;
            return %13;
        };
        """)
    @Reflect
    static int case211() {
        if ((((true ? (short) 1 : (short) 2) == (short) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case212" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @2;
            %2 : java.type:"short" = conv %1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"short"):
            %4 : java.type:"int" = conv %3;
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"short" = conv %5;
            %7 : java.type:"int" = conv %6;
            %8 : java.type:"boolean" = eq %4 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @0;
            branch ^block_4(%9);

          ^block_4(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_5;

          ^block_5:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case212() {
        if ((((false ? (short) 1 : (short) 2) == (short) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case213" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"short" = conv %3;
            branch ^block_3(%4);

          ^block_2:
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"short" = conv %5;
            branch ^block_3(%6);

          ^block_3(%7 : java.type:"short"):
            %8 : java.type:"int" = conv %7;
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"short" = conv %9;
            %11 : java.type:"int" = conv %10;
            %12 : java.type:"boolean" = eq %8 %11;
            branch ^block_4;

          ^block_4:
            %13 : java.type:"int" = constant @1;
            branch ^block_5(%13);

          ^block_5(%14 : java.type:"int"):
            %15 : java.type:"int" = constant @1;
            %16 : java.type:"boolean" = eq %14 %15;
            branch ^block_6;

          ^block_6:
            %17 : java.type:"int" = constant @1;
            return %17;
        };
        """)
    @Reflect
    static int case213(boolean boolean0) {
        if ((((boolean0 ? (short) 1 : (short) 1) == (short) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case214" (%0 : java.type:"short")java.type:"int" -> {
            %1 : Var<java.type:"short"> = var %0 @"short0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @2;
            %4 : java.type:"short" = conv %3;
            branch ^block_2(%4);

          ^block_2(%5 : java.type:"short"):
            %6 : java.type:"int" = conv %5;
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"short" = conv %7;
            %9 : java.type:"int" = conv %8;
            %10 : java.type:"boolean" = eq %6 %9;
            branch ^block_3;

          ^block_3:
            %11 : java.type:"int" = constant @0;
            branch ^block_4(%11);

          ^block_4(%12 : java.type:"int"):
            %13 : java.type:"int" = constant @1;
            %14 : java.type:"boolean" = eq %12 %13;
            branch ^block_5;

          ^block_5:
            %15 : java.type:"int" = constant @0;
            return %15;
        };
        """)
    @Reflect
    static int case214(short short0) {
        if ((((false ? short0 : (short) 2) == (short) 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case215" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"char" = constant @'a';
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"char"):
            %3 : java.type:"int" = conv %2;
            %4 : java.type:"char" = constant @'a';
            %5 : java.type:"int" = conv %4;
            %6 : java.type:"boolean" = eq %3 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @1;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @1;
            return %11;
        };
        """)
    @Reflect
    static int case215() {
        if ((((true ? 'a' : 'b') == 'a') ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case216" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"char" = constant @'b';
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"char"):
            %3 : java.type:"int" = conv %2;
            %4 : java.type:"char" = constant @'a';
            %5 : java.type:"int" = conv %4;
            %6 : java.type:"boolean" = eq %3 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case216() {
        if ((((false ? 'a' : 'b') == 'a') ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case217" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"char" = constant @'a';
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"char" = constant @'a';
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"char"):
            %6 : java.type:"int" = conv %5;
            %7 : java.type:"char" = constant @'a';
            %8 : java.type:"int" = conv %7;
            %9 : java.type:"boolean" = eq %6 %8;
            branch ^block_4;

          ^block_4:
            %10 : java.type:"int" = constant @1;
            branch ^block_5(%10);

          ^block_5(%11 : java.type:"int"):
            %12 : java.type:"int" = constant @1;
            %13 : java.type:"boolean" = eq %11 %12;
            branch ^block_6;

          ^block_6:
            %14 : java.type:"int" = constant @1;
            return %14;
        };
        """)
    @Reflect
    static int case217(boolean boolean0) {
        if ((((boolean0 ? 'a' : 'a') == 'a') ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case218" (%0 : java.type:"char")java.type:"int" -> {
            %1 : Var<java.type:"char"> = var %0 @"char0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"char" = constant @'b';
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"char"):
            %5 : java.type:"int" = conv %4;
            %6 : java.type:"char" = constant @'a';
            %7 : java.type:"int" = conv %6;
            %8 : java.type:"boolean" = eq %5 %7;
            branch ^block_3;

          ^block_3:
            %9 : java.type:"int" = constant @0;
            branch ^block_4(%9);

          ^block_4(%10 : java.type:"int"):
            %11 : java.type:"int" = constant @1;
            %12 : java.type:"boolean" = eq %10 %11;
            branch ^block_5;

          ^block_5:
            %13 : java.type:"int" = constant @0;
            return %13;
        };
        """)
    @Reflect
    static int case218(char char0) {
        if ((((false ? char0 : 'b') == 'a') ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case219" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case219() {
        if ((((true ? 1 : 2) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case220" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @2;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"int"):
            %3 : java.type:"int" = constant @1;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case220() {
        if ((((false ? 1 : 2) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case221" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"int" = constant @1;
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"int" = constant @1;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"int"):
            %6 : java.type:"int" = constant @1;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case221(boolean boolean0) {
        if ((((boolean0 ? 1 : 1) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case222" (%0 : java.type:"int")java.type:"int" -> {
            %1 : Var<java.type:"int"> = var %0 @"int0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"int" = constant @2;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case222(int int0) {
        if ((((false ? int0 : 2) == 1) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case223" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"long" = constant @1L;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"long"):
            %3 : java.type:"long" = constant @1L;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case223() {
        if ((((true ? 1L : 2L) == 1L) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case224" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"long" = constant @2L;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"long"):
            %3 : java.type:"long" = constant @1L;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case224() {
        if ((((false ? 1L : 2L) == 1L) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case225" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"long" = constant @1L;
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"long" = constant @1L;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"long"):
            %6 : java.type:"long" = constant @1L;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case225(boolean boolean0) {
        if ((((boolean0 ? 1L : 1L) == 1L) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case226" (%0 : java.type:"long")java.type:"int" -> {
            %1 : Var<java.type:"long"> = var %0 @"long0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"long" = constant @2L;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"long"):
            %5 : java.type:"long" = constant @1L;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case226(long long0) {
        if ((((false ? long0 : 2L) == 1L) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case227" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"float" = constant @1.0f;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"float"):
            %3 : java.type:"float" = constant @1.0f;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case227() {
        if ((((true ? 1.0F : 2.0F) == 1.0F) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case228" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"float" = constant @2.0f;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"float"):
            %3 : java.type:"float" = constant @1.0f;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case228() {
        if ((((false ? 1.0F : 2.0F) == 1.0F) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case229" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"float" = constant @1.0f;
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"float" = constant @1.0f;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"float"):
            %6 : java.type:"float" = constant @1.0f;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case229(boolean boolean0) {
        if ((((boolean0 ? 1.0F : 1.0F) == 1.0F) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case230" (%0 : java.type:"float")java.type:"int" -> {
            %1 : Var<java.type:"float"> = var %0 @"float0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"float" = constant @2.0f;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"float"):
            %5 : java.type:"float" = constant @1.0f;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case230(float float0) {
        if ((((false ? float0 : 2.0F) == 1.0F) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case231" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"double" = constant @1.0d;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"double"):
            %3 : java.type:"double" = constant @1.0d;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case231() {
        if ((((true ? 1.0 : 2.0) == 1.0) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case232" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"double" = constant @2.0d;
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"double"):
            %3 : java.type:"double" = constant @1.0d;
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case232() {
        if ((((false ? 1.0 : 2.0) == 1.0) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case233" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"double" = constant @1.0d;
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"double" = constant @1.0d;
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"double"):
            %6 : java.type:"double" = constant @1.0d;
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case233(boolean boolean0) {
        if ((((boolean0 ? 1.0 : 1.0) == 1.0) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case234" (%0 : java.type:"double")java.type:"int" -> {
            %1 : Var<java.type:"double"> = var %0 @"double0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"double" = constant @2.0d;
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"double"):
            %5 : java.type:"double" = constant @1.0d;
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case234(double double0) {
        if ((((false ? double0 : 2.0) == 1.0) ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case235" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"java.lang.String" = constant @"a";
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"java.lang.String"):
            %3 : java.type:"java.lang.String" = constant @"a";
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @1;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @1;
            return %9;
        };
        """)
    @Reflect
    static int case235() {
        if ((((true ? "a" : "b") == "a") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case236" ()java.type:"int" -> {
            %0 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %1 : java.type:"java.lang.String" = constant @"b";
            branch ^block_2(%1);

          ^block_2(%2 : java.type:"java.lang.String"):
            %3 : java.type:"java.lang.String" = constant @"a";
            %4 : java.type:"boolean" = eq %2 %3;
            branch ^block_3;

          ^block_3:
            %5 : java.type:"int" = constant @0;
            branch ^block_4(%5);

          ^block_4(%6 : java.type:"int"):
            %7 : java.type:"int" = constant @1;
            %8 : java.type:"boolean" = eq %6 %7;
            branch ^block_5;

          ^block_5:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case236() {
        if ((((false ? "a" : "b") == "a") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case237" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : java.type:"boolean" = var.load %1;
            cbranch %2 ^block_1 ^block_2;

          ^block_1:
            %3 : java.type:"java.lang.String" = constant @"a";
            branch ^block_3(%3);

          ^block_2:
            %4 : java.type:"java.lang.String" = constant @"a";
            branch ^block_3(%4);

          ^block_3(%5 : java.type:"java.lang.String"):
            %6 : java.type:"java.lang.String" = constant @"a";
            %7 : java.type:"boolean" = eq %5 %6;
            branch ^block_4;

          ^block_4:
            %8 : java.type:"int" = constant @1;
            branch ^block_5(%8);

          ^block_5(%9 : java.type:"int"):
            %10 : java.type:"int" = constant @1;
            %11 : java.type:"boolean" = eq %9 %10;
            branch ^block_6;

          ^block_6:
            %12 : java.type:"int" = constant @1;
            return %12;
        };
        """)
    @Reflect
    static int case237(boolean boolean0) {
        if ((((boolean0 ? "a" : "a") == "a") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case238" (%0 : java.type:"java.lang.String")java.type:"int" -> {
            %1 : Var<java.type:"java.lang.String"> = var %0 @"string0";
            %2 : java.type:"boolean" = constant @false;
            branch ^block_1;

          ^block_1:
            %3 : java.type:"java.lang.String" = constant @"b";
            branch ^block_2(%3);

          ^block_2(%4 : java.type:"java.lang.String"):
            %5 : java.type:"java.lang.String" = constant @"a";
            %6 : java.type:"boolean" = eq %4 %5;
            branch ^block_3;

          ^block_3:
            %7 : java.type:"int" = constant @0;
            branch ^block_4(%7);

          ^block_4(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @1;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_5;

          ^block_5:
            %11 : java.type:"int" = constant @0;
            return %11;
        };
        """)
    @Reflect
    static int case238(String string0) {
        if ((((false ? string0 : "b") == "a") ? 1 : 0) == 1) {
            return 1;
        }
        return 0;
    }

    @IR("""
        func @"case239" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            %1 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            var.store %0 %6;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = var.load %0;
            return %7;
        };
        """)
    @Reflect
    static int case239() {
        int v0; if ((true ? 1 : 2) == 1) { v0 = 1; } return v0;
    }

    @IR("""
        func @"case240" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            %1 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"java.lang.String" = constant @"a";
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"java.lang.String"):
            %4 : java.type:"java.lang.String" = constant @"a";
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"int" = constant @1;
            var.store %0 %6;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = var.load %0;
            return %7;
        };
        """)
    @Reflect
    static int case240() {
        int v0; if ((true ? "a" : "b") == "a") { v0 = 1; } return v0;
    }

    @IR("""
        func @"case241" ()java.type:"int" -> {
            %0 : java.type:"int" = constant @1;
            %1 : java.type:"int" = constant @4;
            %2 : java.type:"int" = add %0 %1;
            %3 : Var<java.type:"int"> = var @"v1";
            %4 : java.type:"int" = constant @5;
            %5 : java.type:"boolean" = eq %2 %4;
            branch ^block_1;

          ^block_1:
            %6 : java.type:"int" = constant @1;
            var.store %3 %6;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = var.load %3;
            return %7;
        };
        """)
    @Reflect
    static int case241() {
        final int v0 = 1 + Integer.BYTES; int v1; if (v0 == 5) { v1 = 1; } return v1;
    }

    @IR("""
        func @"case242" ()java.type:"int" -> {
            %0 : java.type:"java.lang.String" = constant @"a";
            %1 : java.type:"int" = constant @1;
            %2 : java.type:"java.lang.String" = constant @"a1";
            %3 : Var<java.type:"int"> = var @"v1";
            %4 : java.type:"java.lang.String" = constant @"a1";
            %5 : java.type:"boolean" = eq %2 %4;
            branch ^block_1;

          ^block_1:
            %6 : java.type:"int" = constant @1;
            var.store %3 %6;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = var.load %3;
            return %7;
        };
        """)
    @Reflect
    static int case242() {
        final String v0 = "a" + 1; int v1; if (v0 == "a1") { v1 = 1; } return v1;
    }

    @IR("""
        func @"case243" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : Var<java.type:"int"> = var @"v0";
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_1 ^block_3;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            var.store %2 %4;
            %5 : java.type:"int" = constant @0;
            %6 : java.type:"boolean" = gt %4 %5;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = var.load %2;
            return %7;

          ^block_3:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case243(boolean boolean0) {
        int v0; if (boolean0 && ((v0 = 1) > 0)) { return v0; } return 0;
    }

    @IR("""
        func @"case244" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : Var<java.type:"int"> = var @"v0";
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_2 ^block_1;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            var.store %2 %4;
            %5 : java.type:"int" = constant @0;
            %6 : java.type:"boolean" = gt %4 %5;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = constant @0;
            return %7;
        };
        """)
    @Reflect
    static int case244(boolean boolean0) {
        int v0; if (boolean0 || ((v0 = 1) > 0)) { return 0; } return v0;
    }

    @IR("""
        func @"case245" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : Var<java.type:"int"> = var @"v0";
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_1 ^block_3;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            var.store %2 %4;
            %5 : java.type:"int" = constant @0;
            %6 : java.type:"boolean" = gt %4 %5;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = var.load %2;
            return %7;

          ^block_3:
            %8 : java.type:"int" = constant @0;
            return %8;
        };
        """)
    @Reflect
    static int case245(boolean boolean0) {
        int v0; if (!(boolean0 && ((v0 = 1) > 0))) { return 0; } return v0;
    }

    @IR("""
        func @"case246" (%0 : java.type:"boolean")java.type:"int" -> {
            %1 : Var<java.type:"boolean"> = var %0 @"boolean0";
            %2 : Var<java.type:"int"> = var @"v0";
            %3 : java.type:"boolean" = var.load %1;
            cbranch %3 ^block_1 ^block_3;

          ^block_1:
            %4 : java.type:"int" = constant @1;
            var.store %2 %4;
            %5 : java.type:"int" = constant @0;
            %6 : java.type:"boolean" = gt %4 %5;
            branch ^block_2;

          ^block_2:
            %7 : java.type:"int" = var.load %2;
            return %7;

          ^block_3:
            %8 : java.type:"boolean" = constant @false;
            branch ^block_4;

          ^block_4:
            %9 : java.type:"int" = constant @0;
            return %9;
        };
        """)
    @Reflect
    static int case246(boolean boolean0) {
        int v0; if (boolean0 ? ((v0 = 1) > 0) : false) { return v0; } return 0;
    }

    @IR("""
        func @"case247" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            var.store %0 %6;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = var.load %0;
            return %7;
        };
        """)
    @Reflect
    static int case247() {
        int v0; while ((true ? 1 : 2) == 1) { v0 = 1; break; } return v0;
    }

    @IR("""
        func @"case248" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            branch ^block_1;

          ^block_1:
            %1 : java.type:"int" = constant @1;
            var.store %0 %1;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"boolean" = constant @true;
            branch ^block_3;

          ^block_3:
            %3 : java.type:"int" = constant @1;
            branch ^block_4(%3);

          ^block_4(%4 : java.type:"int"):
            %5 : java.type:"int" = constant @1;
            %6 : java.type:"boolean" = neq %4 %5;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = var.load %0;
            return %7;
        };
        """)
    @Reflect
    static int case248() {
        int v0; do { v0 = 1; } while ((true ? 1 : 2) != 1); return v0;
    }

    @IR("""
        func @"case249" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            branch ^block_1;

          ^block_1:
            %1 : java.type:"boolean" = constant @true;
            branch ^block_2;

          ^block_2:
            %2 : java.type:"int" = constant @1;
            branch ^block_3(%2);

          ^block_3(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_4;

          ^block_4:
            %6 : java.type:"int" = constant @1;
            var.store %0 %6;
            branch ^block_5;

          ^block_5:
            %7 : java.type:"int" = var.load %0;
            return %7;
        };
        """)
    @Reflect
    static int case249() {
        int v0; for (; (true ? 1 : 2) == 1;) { v0 = 1; break; } return v0;
    }

    @IR("""
        func @"case250" ()java.type:"int" -> {
            %0 : Var<java.type:"int"> = var @"v0";
            %1 : java.type:"boolean" = constant @true;
            branch ^block_1;

          ^block_1:
            %2 : java.type:"int" = constant @1;
            branch ^block_2(%2);

          ^block_2(%3 : java.type:"int"):
            %4 : java.type:"int" = constant @1;
            %5 : java.type:"boolean" = eq %3 %4;
            branch ^block_3;

          ^block_3:
            %6 : java.type:"boolean" = constant @true;
            branch ^block_4;

          ^block_4:
            %7 : java.type:"int" = constant @3;
            branch ^block_5(%7);

          ^block_5(%8 : java.type:"int"):
            %9 : java.type:"int" = constant @3;
            %10 : java.type:"boolean" = eq %8 %9;
            branch ^block_6;

          ^block_6:
            %11 : java.type:"int" = constant @1;
            var.store %0 %11;
            branch ^block_7;

          ^block_7:
            %12 : java.type:"int" = var.load %0;
            return %12;
        };
        """)
    @Reflect
    static int case250() {
        int v0; if (((true ? 1 : 2) == 1) && ((true ? 3 : 4) == 3)) { v0 = 1; } return v0;
    }
}
