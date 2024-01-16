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

/*
 * @test
 * @summary Smoke test for code reflection with try statements.
 * @build TryTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TryTest
 */

public class TryTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : TryTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.try
                    ()void -> {
                        %3 : int = constant @"1";
                        var.store %2 %3;
                        yield;
                    }
                    ^catch(%4 : java.lang.Exception)void -> {
                        %5 : Var<java.lang.Exception> = var %4 @"e";
                        %6 : int = constant @"2";
                        var.store %2 %6;
                        yield;
                    }
                    ^finally()void -> {
                        %7 : int = constant @"3";
                        var.store %2 %7;
                        yield;
                    };
                return;
            };
            """)
    void test1() {
        int i = 0;
        try {
            i = 1;
        } catch (Exception e) {
            i = 2;
        } finally {
            i = 3;
        }
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : TryTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.try
                    ()void -> {
                        %3 : int = constant @"1";
                        var.store %2 %3;
                        yield;
                    }
                    ^finally()void -> {
                        %4 : int = constant @"3";
                        var.store %2 %4;
                        yield;
                    };
                return;
            };
            """)
    void test2() {
        int i = 0;
        try {
            i = 1;
        } finally {
            i = 3;
        }
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : TryTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.try
                    ()void -> {
                        %3 : int = constant @"1";
                        var.store %2 %3;
                        yield;
                    }
                    ^catch(%4 : java.lang.Exception)void -> {
                        %5 : Var<java.lang.Exception> = var %4 @"e";
                        %6 : java.lang.Exception = var.load %5;
                        invoke %6 @"java.lang.Exception::printStackTrace()void";
                        yield;
                    };
                return;
            };
            """)
    void test3() {
        int i = 0;
        try {
            i = 1;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    static class A implements AutoCloseable {
        final B b;

        public A() {
            this.b = null;
        }

        @Override
        public void close() throws Exception {

        }
    }

    static class B implements AutoCloseable {
        C c;

        @Override
        public void close() throws Exception {

        }
    }

    static class C implements AutoCloseable {
        @Override
        public void close() throws Exception {

        }
    }

    A a() {
        return null;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : TryTest)void -> {
                java.try
                    ^resources()java.lang.reflect.code.CoreOps$Tuple<Var<TryTest$A>, TryTest$B, Var<TryTest$C>> -> {
                        %1 : TryTest$A = invoke %0 @"TryTest::a()TryTest$A";
                        %2 : Var<TryTest$A> = var %1 @"a";
                        %3 : TryTest$A = var.load %2;
                        %4 : TryTest$B = field.load %3 @"TryTest$A::b()TryTest$B";
                        %5 : TryTest$A = var.load %2;
                        %6 : TryTest$B = field.load %5 @"TryTest$A::b()TryTest$B";
                        %7 : TryTest$C = field.load %6 @"TryTest$B::c()TryTest$C";
                        %8 : Var<TryTest$C> = var %7 @"c";
                        %9 : java.lang.reflect.code.CoreOps$Tuple<Var<TryTest$A>, TryTest$B, Var<TryTest$C>> = tuple %2 %4 %8;
                        yield %9;
                    }
                    (%10 : Var<TryTest$A>, %11 : Var<TryTest$C>)void -> {
                        %12 : TryTest$A = var.load %10;
                        %13 : Var<TryTest$A> = var %12 @"_a";
                        %14 : TryTest$C = var.load %11;
                        %15 : Var<TryTest$C> = var %14 @"_c";
                        yield;
                    }
                    ^catch(%16 : java.lang.Throwable)void -> {
                        %17 : Var<java.lang.Throwable> = var %16 @"t";
                        %18 : java.lang.Throwable = var.load %17;
                        invoke %18 @"java.lang.Throwable::printStackTrace()void";
                        yield;
                    }
                    ^finally()void -> {
                        %19 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                        %20 : java.lang.String = constant @"F";
                        invoke %19 %20 @"java.io.PrintStream::println(java.lang.String)void";
                        yield;
                    };
                return;
            };
            """)
    void test4() throws Exception {
        try (A a = a(); a.b; C c = a.b.c) {
            A _a = a;
            C _c = c;
        } catch (Throwable t) {
            t.printStackTrace();
        } finally {
            System.out.println("F");
        }
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : TryTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.try
                    ()void -> {
                        %3 : int = constant @"1";
                        var.store %2 %3;
                        yield;
                    }
                    ^catch(%4 : java.lang.NullPointerException)void -> {
                        %5 : Var<java.lang.NullPointerException> = var %4 @"e";
                        %6 : int = constant @"2";
                        var.store %2 %6;
                        yield;
                    }
                    ^catch(%7 : java.lang.OutOfMemoryError)void -> {
                        %8 : Var<java.lang.OutOfMemoryError> = var %7 @"e";
                        %9 : int = constant @"3";
                        var.store %2 %9;
                        yield;
                    };
                return;
            };
            """)
    void test5() {
        int i = 0;
        try {
            i = 1;
        } catch (NullPointerException e) {
            i = 2;
        } catch (OutOfMemoryError e) {
            i = 3;
        }
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : TryTest)void -> {
                %1 : int = constant @"0";
                %2 : Var<int> = var %1 @"i";
                java.try
                    ()void -> {
                        return;
                    }
                    ^catch(%3 : java.lang.Exception)void -> {
                        %4 : Var<java.lang.Exception> = var %3 @"e";
                        %5 : java.lang.Exception = var.load %4;
                        throw %5;
                    }
                    ^finally()void -> {
                        return;
                    };
                return;
            };
            """)
    void test6() {
        int i = 0;
        try {
            return;
        } catch (Exception e) {
            throw e;
        } finally {
            return;
        }
     }

}