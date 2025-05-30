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

/*
 * @test
 * @summary Smoke test for code reflection with constant values.
 * @modules jdk.incubator.code
 * @build ConstantsTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester ConstantsTest
 */

import jdk.incubator.code.CodeReflection;
import java.util.function.Function;

public class ConstantsTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @"";
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                return;
            };
            """)
    void test1() {
        String s = "";
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @"Hello World";
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                return;
            };
            """)
    void test2() {
        String s = "Hello World";
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                return;
            };
            """)
    void test3() {
        String s = null;
    }

    @IR("""
            func @"test4" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"java.util.function.Function";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test4() {
        Class<?> s = Function.class;
    }

    @IR("""
            func @"test5" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @42;
                %2 : java.type:"byte" = conv %1;
                %3 : Var<java.type:"byte"> = var %2 @"v";
                %4 : java.type:"int" = constant @-42;
                %5 : java.type:"byte" = conv %4;
                var.store %3 %5;
                return;
            };
            """)
    @CodeReflection
    void test5() {
        byte v = 42;
        v = -42;
    }

    @IR("""
            func @"test6" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @42;
                %2 : java.type:"short" = conv %1;
                %3 : Var<java.type:"short"> = var %2 @"v";
                %4 : java.type:"int" = constant @-42;
                %5 : java.type:"short" = conv %4;
                var.store %3 %5;
                return;
            };
            """)
    @CodeReflection
    void test6() {
        short v = 42;
        v = -42;
    }

    @IR("""
            func @"test7" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @42;
                %2 : Var<java.type:"int"> = var %1 @"v";
                %3 : java.type:"int" = constant @-42;
                var.store %2 %3;
                return;
            };
            """)
    @CodeReflection
    void test7() {
        int v = 42;
        v = -42;
    }

    @IR("""
            func @"test8" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"long" = constant @42;
                %2 : Var<java.type:"long"> = var %1 @"v";
                %3 : java.type:"long" = constant @-42;
                var.store %2 %3;
                return;
            };
            """)
    @CodeReflection
    void test8() {
        long v = 42L;
        v = -42L;
    }

    @IR("""
            func @"test9" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"float" = constant @42.0f;
                %2 : Var<java.type:"float"> = var %1 @"v";
                %3 : java.type:"float" = constant @42.0f;
                %4 : java.type:"float" = neg %3;
                var.store %2 %4;
                return;
            };
            """)
    @CodeReflection
    void test9() {
        float v = 42.0f;
        v = -42.0f;
    }

    @IR("""
            func @"test10" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"double" = constant @42.0d;
                %2 : Var<java.type:"double"> = var %1 @"v";
                %3 : java.type:"double" = constant @42.0d;
                %4 : java.type:"double" = neg %3;
                var.store %2 %4;
                return;
            };
            """)
    @CodeReflection
    void test10() {
        double v = 42.0;
        v = -42.0;
    }

    @IR("""
            func @"test11" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"char" = constant @'a';
                %2 : Var<java.type:"char"> = var %1 @"v";
                return;
            };
            """)
    @CodeReflection
    void test11() {
        char v = 'a';
    }

    @IR("""
            func @"test12" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"boolean" = constant @true;
                %2 : Var<java.type:"boolean"> = var %1 @"b";
                %3 : java.type:"boolean" = constant @false;
                var.store %2 %3;
                return;
            };
            """)
    @CodeReflection
    void test12() {
        boolean b = true;
        b = false;
    }

    @IR("""
            func @"test13" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"float";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test13() {
        Class<?> s = float.class;
    }

    @IR("""
            func @"test14" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"java.lang.String[]";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test14() {
        Class<?> s = String[].class;
    }

    @IR("""
            func @"test15" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"java.lang.String[][]";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test15() {
        Class<?> s = String[][].class;
    }

    @IR("""
            func @"test16" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"java.lang.String[][][]";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test16() {
        Class<?> s = String[][][].class;
    }

    @IR("""
            func @"test17" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"boolean[]";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test17() {
        Class<?> s = boolean[].class;
    }

    @IR("""
            func @"test18" (%0 : java.type:"ConstantsTest")java.type:"void" -> {
                %1 : java.type:"java.lang.Class" = constant @java.type:"boolean[][][]";
                %2 : Var<java.type:"java.lang.Class<?>"> = var %1 @"s";
                return;
            };
            """)
    @CodeReflection
    void test18() {
        Class<?> s = boolean[][][].class;
    }
}
