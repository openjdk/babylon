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
import java.util.function.Consumer;
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with lambda expressions.
 * @modules jdk.incubator.code
 * @build LambdaTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester LambdaTest
 */

public class LambdaTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"LambdaTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Consumer<java.lang.String>" = lambda (%2 : java.type:"java.lang.String")java.type:"void" -> {
                    %3 : Var<java.type:"java.lang.String"> = var %2 @"s";
                    %4 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                    %5 : java.type:"java.lang.String" = var.load %3;
                    invoke %4 %5 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                    return;
                };
                %6 : Var<java.type:"java.util.function.Consumer<java.lang.String>"> = var %1 @"c";
                %7 : java.type:"java.util.function.Consumer<java.lang.String>" = var.load %6;
                %8 : java.type:"java.lang.String" = constant @"Hello World";
                invoke %7 %8 @java.ref:"java.util.function.Consumer::accept(java.lang.Object):void";
                return;
            };
            """)
    void test1() {
        Consumer<String> c = s -> {
            System.out.println(s);
        };
        c.accept("Hello World");
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"LambdaTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.String>" = lambda ()java.type:"java.lang.String" -> {
                    %2 : java.type:"java.lang.String" = constant @"Hello World";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<java.lang.String>"> = var %1 @"c";
                %4 : java.type:"java.util.function.Supplier<java.lang.String>" = var.load %3;
                %5 : java.type:"java.lang.String" = invoke %4 @java.ref:"java.util.function.Supplier::get():java.lang.Object";
                %6 : Var<java.type:"java.lang.String"> = var %5 @"s";
                return;
            };
            """)
    void test2() {
        Supplier<String> c = () -> {
            return "Hello World";
        };
        String s = c.get();
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"LambdaTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.String>" = lambda ()java.type:"java.lang.String" -> {
                    %2 : java.type:"java.lang.String" = constant @"Hello World";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<java.lang.String>"> = var %1 @"c";
                return;
            };
            """)
    void test3() {
        Supplier<String> c = () -> "Hello World";
    }

    String s_f;

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"LambdaTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.String>" = lambda ()java.type:"java.lang.String" -> {
                    %2 : java.type:"java.lang.String" = field.load %0 @java.ref:"LambdaTest::s_f:java.lang.String";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<java.lang.String>"> = var %1 @"c";
                return;
            };
            """)
    void test4() {
        Supplier<String> c = () -> {
            return s_f;
        };
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"LambdaTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : Var<java.type:"int"> = var %2 @"j";
                %5 : java.type:"int" = constant @"3";
                %6 : Var<java.type:"int"> = var %5 @"k";
                %7 : java.type:"java.util.function.Supplier<java.lang.Integer>" = lambda ()java.type:"java.lang.Integer" -> {
                    %8 : java.type:"int" = constant @"4";
                    %9 : Var<java.type:"int"> = var %8 @"l";
                    %10 : java.type:"java.util.function.Supplier<java.lang.Integer>" = lambda ()java.type:"java.lang.Integer" -> {
                        %11 : java.type:"int" = var.load %4;
                        %12 : java.type:"int" = var.load %6;
                        %13 : java.type:"int" = add %11 %12;
                        %14 : java.type:"int" = var.load %9;
                        %15 : java.type:"int" = add %13 %14;
                        %16 : Var<java.type:"int"> = var %15 @"r";
                        %17 : java.type:"int" = var.load %16;
                        %18 : java.type:"java.lang.Integer" = invoke %17 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                        return %18;
                    };
                    %19 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %10 @"sInner";
                    %20 : java.type:"int" = var.load %3;
                    %21 : java.type:"java.util.function.Supplier<java.lang.Integer>" = var.load %19;
                    %22 : java.type:"java.lang.Integer" = invoke %21 @java.ref:"java.util.function.Supplier::get():java.lang.Object";
                    %23 : java.type:"int" = invoke %22 @java.ref:"java.lang.Integer::intValue():int";
                    %24 : java.type:"int" = add %20 %23;
                    %25 : Var<java.type:"int"> = var %24 @"r";
                    %26 : java.type:"int" = var.load %25;
                    %27 : java.type:"java.lang.Integer" = invoke %26 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                    return %27;
                };
                %28 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %7 @"sOuter";
                return;
            };
            """)
    void test5(int i, int j) {
        int k = 3;
        Supplier<Integer> sOuter = () -> {
            int l = 4;
            Supplier<Integer> sInner = () -> {
                int r = j + k + l;
                return r;
            };

            int r = i + sInner.get();
            return r;
        };
    }

    int f;

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"LambdaTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.Integer>" = lambda ()java.type:"java.lang.Integer" -> {
                    %2 : java.type:"int" = field.load %0 @java.ref:"LambdaTest::f:int";
                    %3 : java.type:"java.lang.Integer" = invoke %2 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                    return %3;
                };
                %4 : Var<java.type:"java.util.function.Supplier<java.lang.Integer>"> = var %1 @"s";
                return;
            };
            """)
    void test6() {
        Supplier<Integer> s = () -> f;
    }
}
