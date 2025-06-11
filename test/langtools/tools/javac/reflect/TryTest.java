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

/*
 * @test
 * @summary Smoke test for code reflection with try statements.
 * @modules jdk.incubator.code
 * @build TryTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TryTest
 */

public class TryTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"TryTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @0;
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.try
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = constant @1;
                        var.store %2 %3;
                        yield;
                    }
                    (%4 : java.type:"java.lang.Exception")java.type:"void" -> {
                        %5 : Var<java.type:"java.lang.Exception"> = var %4 @"e";
                        %6 : java.type:"int" = constant @2;
                        var.store %2 %6;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %7 : java.type:"int" = constant @3;
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
            func @"test2" (%0 : java.type:"TryTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @0;
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.try
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = constant @1;
                        var.store %2 %3;
                        yield;
                    }
                    ()java.type:"void" -> {
                        %4 : java.type:"int" = constant @3;
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
            func @"test3" (%0 : java.type:"TryTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @0;
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.try
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = constant @1;
                        var.store %2 %3;
                        yield;
                    }
                    (%4 : java.type:"java.lang.Exception")java.type:"void" -> {
                        %5 : Var<java.type:"java.lang.Exception"> = var %4 @"e";
                        %6 : java.type:"java.lang.Exception" = var.load %5;
                        invoke %6 @java.ref:"java.lang.Exception::printStackTrace():void";
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
            func @"test4" (%0 : java.type:"TryTest")java.type:"void" -> {
                java.try
                    ()Tuple<Var<java.type:"TryTest$A">, java.type:"TryTest$B", Var<java.type:"TryTest$C">> -> {
                        %1 : java.type:"TryTest$A" = invoke %0 @java.ref:"TryTest::a():TryTest$A";
                        %2 : Var<java.type:"TryTest$A"> = var %1 @"a";
                        %3 : java.type:"TryTest$A" = var.load %2;
                        %4 : java.type:"TryTest$B" = field.load %3 @java.ref:"TryTest$A::b:TryTest$B";
                        %5 : java.type:"TryTest$A" = var.load %2;
                        %6 : java.type:"TryTest$B" = field.load %5 @java.ref:"TryTest$A::b:TryTest$B";
                        %7 : java.type:"TryTest$C" = field.load %6 @java.ref:"TryTest$B::c:TryTest$C";
                        %8 : Var<java.type:"TryTest$C"> = var %7 @"c";
                        %9 : Tuple<Var<java.type:"TryTest$A">, java.type:"TryTest$B", Var<java.type:"TryTest$C">> = tuple %2 %4 %8;
                        yield %9;
                    }
                    (%10 : Var<java.type:"TryTest$A">, %11 : Var<java.type:"TryTest$C">)java.type:"void" -> {
                        %12 : java.type:"TryTest$A" = var.load %10;
                        %13 : Var<java.type:"TryTest$A"> = var %12 @"_a";
                        %14 : java.type:"TryTest$C" = var.load %11;
                        %15 : Var<java.type:"TryTest$C"> = var %14 @"_c";
                        yield;
                    }
                    (%16 : java.type:"java.lang.Throwable")java.type:"void" -> {
                        %17 : Var<java.type:"java.lang.Throwable"> = var %16 @"t";
                        %18 : java.type:"java.lang.Throwable" = var.load %17;
                        invoke %18 @java.ref:"java.lang.Throwable::printStackTrace():void";
                        yield;
                    }
                    ()java.type:"void" -> {
                        %19 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                        %20 : java.type:"java.lang.String" = constant @"F";
                        invoke %19 %20 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
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
            func @"test5" (%0 : java.type:"TryTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @0;
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.try
                    ()java.type:"void" -> {
                        %3 : java.type:"int" = constant @1;
                        var.store %2 %3;
                        yield;
                    }
                    (%4 : java.type:"java.lang.NullPointerException")java.type:"void" -> {
                        %5 : Var<java.type:"java.lang.NullPointerException"> = var %4 @"e";
                        %6 : java.type:"int" = constant @2;
                        var.store %2 %6;
                        yield;
                    }
                    (%7 : java.type:"java.lang.OutOfMemoryError")java.type:"void" -> {
                        %8 : Var<java.type:"java.lang.OutOfMemoryError"> = var %7 @"e";
                        %9 : java.type:"int" = constant @3;
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
            func @"test6" (%0 : java.type:"TryTest")java.type:"void" -> {
                %1 : java.type:"int" = constant @0;
                %2 : Var<java.type:"int"> = var %1 @"i";
                java.try
                    ()java.type:"void" -> {
                        return;
                    }
                    (%3 : java.type:"java.lang.Exception")java.type:"void" -> {
                        %4 : Var<java.type:"java.lang.Exception"> = var %3 @"e";
                        %5 : java.type:"java.lang.Exception" = var.load %4;
                        throw %5;
                    }
                    ()java.type:"void" -> {
                        return;
                    };
                unreachable;
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
