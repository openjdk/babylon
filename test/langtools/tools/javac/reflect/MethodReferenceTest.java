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
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Supplier;

/*
 * @test
 * @summary Smoke test for code reflection with method reference expressions.
 * @modules jdk.incubator.code
 * @build MethodReferenceTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester MethodReferenceTest
 */

public class MethodReferenceTest {

    static void m_s(String s) {}

    void m(String s) {}

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Consumer<java.lang.String>" = lambda (%2 : java.type:"java.lang.String")java.type:"void" -> {
                    %3 : Var<java.type:"java.lang.String"> = var %2 @"x$0";
                    %4 : java.type:"java.lang.String" = var.load %3;
                    invoke %4 @java.ref:"MethodReferenceTest::m_s(java.lang.String):void";
                    return;
                };
                %5 : Var<java.type:"java.util.function.Consumer<java.lang.String>"> = var %1 @"c";
                return;
            };
            """)
    void test1() {
        Consumer<String> c = MethodReferenceTest::m_s;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.BiConsumer<MethodReferenceTest, java.lang.String>" = lambda (%2 : java.type:"MethodReferenceTest", %3 : java.type:"java.lang.String")java.type:"void" -> {
                    %4 : Var<java.type:"MethodReferenceTest"> = var %2 @"rec$";
                    %5 : Var<java.type:"java.lang.String"> = var %3 @"x$0";
                    %6 : java.type:"MethodReferenceTest" = var.load %4;
                    %7 : java.type:"java.lang.String" = var.load %5;
                    invoke %6 %7 @java.ref:"MethodReferenceTest::m(java.lang.String):void";
                    return;
                };
                %8 : Var<java.type:"java.util.function.BiConsumer<MethodReferenceTest, java.lang.String>"> = var %1 @"bc";
                return;
            };
            """)
    void test2() {
        BiConsumer<MethodReferenceTest, String> bc = MethodReferenceTest::m;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Consumer<java.lang.String>" = lambda (%2 : java.type:"java.lang.String")java.type:"void" -> {
                    %3 : Var<java.type:"java.lang.String"> = var %2 @"x$0";
                    %4 : java.type:"java.lang.String" = var.load %3;
                    invoke %0 %4 @java.ref:"MethodReferenceTest::m(java.lang.String):void";
                    return;
                };
                %5 : Var<java.type:"java.util.function.Consumer<java.lang.String>"> = var %1 @"c";
                return;
            };
            """)
    void test3() {
        Consumer<String> c = this::m;
    }

    class A<T> {
        T m(T t) { return t; }
    }

    <T> A<T> a(T t) { return null; }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @"s";
                %2 : java.type:"MethodReferenceTest::A<java.lang.String>" = invoke %0 %1 @java.ref:"MethodReferenceTest::a(java.lang.Object):MethodReferenceTest::A";
                %3 : Var<java.type:"MethodReferenceTest::A<java.lang.String>"> = var %2 @"rec$";
                %4 : java.type:"java.util.function.Function<java.lang.String, java.lang.String>" = lambda (%5 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                    %6 : Var<java.type:"java.lang.String"> = var %5 @"x$0";
                    %7 : java.type:"MethodReferenceTest::A<java.lang.String>" = var.load %3;
                    %8 : java.type:"java.lang.String" = var.load %6;
                    %9 : java.type:"java.lang.String" = invoke %7 %8 @java.ref:"MethodReferenceTest::A::m(java.lang.Object):java.lang.Object";
                    return %9;
                };
                %10 : Var<java.type:"java.util.function.Function<java.lang.String, java.lang.String>"> = var %4 @"f";
                return;
            };
            """)
    void test4() {
        Function<String, String> f = a("s")::m;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                %2 : Var<java.type:"java.io.PrintStream"> = var %1 @"rec$";
                %3 : java.type:"java.util.function.Consumer<java.lang.String>" = lambda (%4 : java.type:"java.lang.String")java.type:"void" -> {
                    %5 : Var<java.type:"java.lang.String"> = var %4 @"x$0";
                    %6 : java.type:"java.io.PrintStream" = var.load %2;
                    %7 : java.type:"java.lang.String" = var.load %5;
                    invoke %6 %7 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                    return;
                };
                %8 : Var<java.type:"java.util.function.Consumer<java.lang.String>"> = var %3 @"c3";
                return;
            };
            """)
    void test5() {
        Consumer<String> c3 = System.out::println;
    }

    static class X {
        X(int i) {}
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Function<java.lang.Integer, MethodReferenceTest$X>" = lambda (%2 : java.type:"java.lang.Integer")java.type:"MethodReferenceTest$X" -> {
                    %3 : Var<java.type:"java.lang.Integer"> = var %2 @"x$0";
                    %4 : java.type:"java.lang.Integer" = var.load %3;
                    %5 : java.type:"int" = invoke %4 @java.ref:"java.lang.Integer::intValue():int";
                    %6 : java.type:"MethodReferenceTest$X" = new %5 @java.ref:"MethodReferenceTest$X::(int)";
                    return %6;
                };
                %7 : Var<java.type:"java.util.function.Function<java.lang.Integer, MethodReferenceTest$X>"> = var %1 @"xNew";
                return;
            };
            """)
    void test6() {
        Function<Integer, X> xNew = X::new;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<MethodReferenceTest::A<java.lang.String>>" = lambda ()java.type:"MethodReferenceTest::A<java.lang.String>" -> {
                    %2 : java.type:"MethodReferenceTest::A<java.lang.String>" = new %0 @java.ref:"MethodReferenceTest::A::(MethodReferenceTest)";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<MethodReferenceTest::A<java.lang.String>>"> = var %1 @"aNew";
                return;
            };
            """)
    void test7() {
        Supplier<A<String>> aNew = A::new;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"MethodReferenceTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.IntFunction<MethodReferenceTest::A<java.lang.String>[]>" = lambda (%2 : java.type:"int")java.type:"MethodReferenceTest::A<java.lang.String>[]" -> {
                    %3 : Var<java.type:"int"> = var %2 @"x$0";
                    %4 : java.type:"int" = var.load %3;
                    %5 : java.type:"MethodReferenceTest::A[]" = new %4 @java.ref:"MethodReferenceTest::A[]::(int)";
                    return %5;
                };
                %6 : Var<java.type:"java.util.function.IntFunction<MethodReferenceTest::A<java.lang.String>[]>"> = var %1 @"aNewArray";
                return;
            };
            """)
    void test8() {
        IntFunction<A<String>[]> aNewArray = A[]::new;
    }

}
