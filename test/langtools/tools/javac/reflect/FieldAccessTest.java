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

import java.io.PrintStream;
import java.lang.runtime.CodeReflection;
import java.util.Spliterator;
import java.util.Spliterator.OfInt;

import static java.lang.System.out;
import static java.util.Spliterator.OfInt.*;

/*
 * @test
 * @summary Smoke test for code reflection with field access.
 * @build FieldAccessTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester FieldAccessTest
 */

public class FieldAccessTest {
    static int s_f;
    int f;

    @CodeReflection
    @IR("""
            func @"test1" (%0 : FieldAccessTest)void -> {
                %1 : int = constant @"1";
                field.store %1 @"FieldAccessTest::s_f()int";
                %2 : int = constant @"1";
                field.store %0 %2 @"FieldAccessTest::f()int";
                return;
            };
            """)
    void test1() {
        s_f = 1;
        f = 1;
    }

    @CodeReflection
    @IR("""
            func @"test1_1" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : int = field.load @"FieldAccessTest::s_f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %6 @"FieldAccessTest::s_f()int";
                return;
            };
            """)
    void test1_1() {
        f += 1;
        s_f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : FieldAccessTest)void -> {
                %1 : int = constant @"1";
                field.store %0 %1 @"FieldAccessTest::f()int";
                field.store %1 @"FieldAccessTest::s_f()int";
                return;
            };
            """)
    void test2() {
        s_f = f = 1;
    }

    @CodeReflection
    @IR("""
            func @"test2_1" (%0 : FieldAccessTest)void -> {
                %1 : int = constant @"1";
                field.store %0 %1 @"FieldAccessTest::f()int";
                return;
            };
            """)
    void test2_1() {
        this.f = 1;
    }

    @CodeReflection
    @IR("""
            func @"test2_2" (%0 : FieldAccessTest)int -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                return %1;
            };
            """)
    int test2_2() {
        return this.f;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : FieldAccessTest)int -> {
                %1 : int = field.load @"FieldAccessTest::s_f()int";
                %2 : int = field.load %0 @"FieldAccessTest::f()int";
                %3 : int = add %1 %2;
                return %3;
            };
            """)
    int test3() {
        return s_f + f;
    }

    static class A {
        B b;
    }

    static class B {
        C c;
    }

    static class C {
        int f;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : FieldAccessTest, %1 : FieldAccessTest$A)void -> {
                %2 : Var<FieldAccessTest$A> = var %1 @"a";
                %3 : FieldAccessTest$A = var.load %2;
                %4 : FieldAccessTest$B = field.load %3 @"FieldAccessTest$A::b()FieldAccessTest$B";
                %5 : FieldAccessTest$C = field.load %4 @"FieldAccessTest$B::c()FieldAccessTest$C";
                %6 : int = constant @"1";
                field.store %5 %6 @"FieldAccessTest$C::f()int";
                return;
            };
            """)
    void test4(A a) {
        a.b.c.f = 1;
    }

    static class X {
        int f;
        static int s_f;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : FieldAccessTest)int -> {
                %1 : int = field.load @"FieldAccessTest$X::s_f()int";
                return %1;
            };
            """)
    int test5() {
        return X.s_f;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : FieldAccessTest)void -> {
                %1 : int = constant @"1";
                field.store %1 @"FieldAccessTest$X::s_f()int";
                return;
            };
            """)
    void test6() {
        X.s_f = 1;
    }


    @CodeReflection
    @IR("""
            func @"test7" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : int = field.load @"FieldAccessTest::s_f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %6 @"FieldAccessTest::s_f()int";
                return;
            };
            """)
    void test7() {
        f += 1;
        s_f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : int = field.load @"FieldAccessTest::s_f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %6 @"FieldAccessTest::s_f()int";
                return;
            };
            """)
    void test8() {
        this.f += 1;
        this.s_f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load @"FieldAccessTest$X::s_f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %3 @"FieldAccessTest$X::s_f()int";
                return;
            };
            """)
    void test9() {
        X.s_f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                field.store %3 @"FieldAccessTest::s_f()int";
                return;
            };
            """)
    void test10() {
        s_f = f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : FieldAccessTest, %1 : FieldAccessTest$A)void -> {
                %2 : Var<FieldAccessTest$A> = var %1 @"a";
                %3 : FieldAccessTest$A = var.load %2;
                %4 : FieldAccessTest$B = field.load %3 @"FieldAccessTest$A::b()FieldAccessTest$B";
                %5 : FieldAccessTest$C = field.load %4 @"FieldAccessTest$B::c()FieldAccessTest$C";
                %6 : int = field.load %5 @"FieldAccessTest$C::f()int";
                %7 : int = constant @"1";
                %8 : int = add %6 %7;
                field.store %5 %8 @"FieldAccessTest$C::f()int";
                return;
            };
            """)
    void test11(A a) {
        a.b.c.f += 1;
    }

    @CodeReflection
    @IR("""
            func @"test12" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : Var<int> = var %1 @"x";
                %5 : int = field.load %0 @"FieldAccessTest::f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %0 %7 @"FieldAccessTest::f()int";
                %8 : Var<int> = var %5 @"y";
                return;
            };
            """)
    void test12() {
        int x = f++;
        int y = f--;
    }

    @CodeReflection
    @IR("""
            func @"test13" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : Var<int> = var %1 @"x";
                %5 : int = field.load %0 @"FieldAccessTest::f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %0 %7 @"FieldAccessTest::f()int";
                %8 : Var<int> = var %5 @"y";
                return;
            };
            """)
    void test13() {
        int x = this.f++;
        int y = this.f--;
    }

    @CodeReflection
    @IR("""
            func @"test14" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load @"FieldAccessTest::s_f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %3 @"FieldAccessTest::s_f()int";
                %4 : Var<int> = var %1 @"x";
                %5 : int = field.load @"FieldAccessTest::s_f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %7 @"FieldAccessTest::s_f()int";
                %8 : Var<int> = var %5 @"y";
                return;
            };
            """)
    void test14() {
        int x = s_f++;
        int y = s_f--;
    }

    @CodeReflection
    @IR("""
            func @"test15" (%0 : FieldAccessTest, %1 : FieldAccessTest$X)void -> {
                %2 : Var<FieldAccessTest$X> = var %1 @"h";
                %3 : FieldAccessTest$X = var.load %2;
                %4 : int = field.load %3 @"FieldAccessTest$X::f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %3 %6 @"FieldAccessTest$X::f()int";
                %7 : Var<int> = var %4 @"x";
                %8 : FieldAccessTest$X = var.load %2;
                %9 : int = field.load %8 @"FieldAccessTest$X::f()int";
                %10 : int = constant @"1";
                %11 : int = sub %9 %10;
                field.store %8 %11 @"FieldAccessTest$X::f()int";
                %12 : Var<int> = var %9 @"y";
                return;
            };
            """)
    void test15(X h) {
        int x = h.f++;
        int y = h.f--;
    }




    @CodeReflection
    @IR("""
            func @"test16" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : Var<int> = var %3 @"x";
                %5 : int = field.load %0 @"FieldAccessTest::f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %0 %7 @"FieldAccessTest::f()int";
                %8 : Var<int> = var %7 @"y";
                return;
            };
            """)
    void test16() {
        int x = ++f;
        int y = --f;
    }

    @CodeReflection
    @IR("""
            func @"test17" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load %0 @"FieldAccessTest::f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %0 %3 @"FieldAccessTest::f()int";
                %4 : Var<int> = var %3 @"x";
                %5 : int = field.load %0 @"FieldAccessTest::f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %0 %7 @"FieldAccessTest::f()int";
                %8 : Var<int> = var %7 @"y";
                return;
            };
            """)
    void test17() {
        int x = ++this.f;
        int y = --this.f;
    }

    @CodeReflection
    @IR("""
            func @"test18" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load @"FieldAccessTest::s_f()int";
                %2 : int = constant @"1";
                %3 : int = add %1 %2;
                field.store %3 @"FieldAccessTest::s_f()int";
                %4 : Var<int> = var %3 @"x";
                %5 : int = field.load @"FieldAccessTest::s_f()int";
                %6 : int = constant @"1";
                %7 : int = sub %5 %6;
                field.store %7 @"FieldAccessTest::s_f()int";
                %8 : Var<int> = var %7 @"y";
                return;
            };
            """)
    void test18() {
        int x = ++s_f;
        int y = --s_f;
    }

    @CodeReflection
    @IR("""
            func @"test19" (%0 : FieldAccessTest, %1 : FieldAccessTest$X)void -> {
                %2 : Var<FieldAccessTest$X> = var %1 @"h";
                %3 : FieldAccessTest$X = var.load %2;
                %4 : int = field.load %3 @"FieldAccessTest$X::f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %3 %6 @"FieldAccessTest$X::f()int";
                %7 : Var<int> = var %6 @"x";
                %8 : FieldAccessTest$X = var.load %2;
                %9 : int = field.load %8 @"FieldAccessTest$X::f()int";
                %10 : int = constant @"1";
                %11 : int = sub %9 %10;
                field.store %8 %11 @"FieldAccessTest$X::f()int";
                %12 : Var<int> = var %11 @"y";
                return;
            };
            """)
    void test19(X h) {
        int x = ++h.f;
        int y = --h.f;
    }

    static class Y extends X {
        int yf;
        static int s_yf;

        @CodeReflection
        @IR("""
                func @"test" (%0 : FieldAccessTest$Y)void -> {
                    %1 : int = field.load %0 @"FieldAccessTest$Y::f()int";
                    %2 : Var<int> = var %1 @"x";
                    %3 : int = field.load @"FieldAccessTest$Y::s_f()int";
                    var.store %2 %3;
                    return;
                };
                """)
        void test() {
            int x = f;
            x = s_f;
        }

        @CodeReflection
        @IR("""
                func @"test2" (%0 : FieldAccessTest$Y)void -> {
                    %1 : int = constant @"1";
                    field.store %0 %1 @"FieldAccessTest$Y::f()int";
                    %2 : int = constant @"1";
                    field.store %2 @"FieldAccessTest$Y::s_f()int";
                    return;
                };
                """)
        void test2() {
            f = 1;
            s_f = 1;
        }

        @CodeReflection
        @IR("""
                func @"test3" (%0 : FieldAccessTest$Y)void -> {
                    %1 : int = field.load %0 @"FieldAccessTest$Y::f()int";
                    %2 : int = constant @"1";
                    %3 : int = add %1 %2;
                    field.store %0 %3 @"FieldAccessTest$Y::f()int";
                    %4 : int = field.load @"FieldAccessTest$Y::s_f()int";
                    %5 : int = constant @"1";
                    %6 : int = add %4 %5;
                    field.store %6 @"FieldAccessTest$Y::s_f()int";
                    return;
                };
                """)
        void test3() {
            f++;
            s_f++;
        }
    }

    @CodeReflection
    @IR("""
            func @"test20" (%0 : FieldAccessTest, %1 : FieldAccessTest$Y)void -> {
                %2 : Var<FieldAccessTest$Y> = var %1 @"y";
                %3 : FieldAccessTest$Y = var.load %2;
                %4 : int = field.load %3 @"FieldAccessTest$Y::f()int";
                %5 : Var<int> = var %4 @"x";
                %6 : FieldAccessTest$Y = var.load %2;
                %7 : int = field.load %6 @"FieldAccessTest$Y::yf()int";
                var.store %5 %7;
                %8 : FieldAccessTest$Y = var.load %2;
                %9 : int = field.load @"FieldAccessTest$Y::s_yf()int";
                var.store %5 %9;
                %10 : int = field.load @"FieldAccessTest$Y::s_yf()int";
                var.store %5 %10;
                %11 : FieldAccessTest$Y = var.load %2;
                %12 : int = field.load @"FieldAccessTest$Y::s_f()int";
                var.store %5 %12;
                %13 : int = field.load @"FieldAccessTest$Y::s_f()int";
                var.store %5 %13;
                return;
            };
            """)
    void test20(Y y) {
        int x = y.f;
        x = y.yf;
        x = y.s_yf;
        x = Y.s_yf;
        x = y.s_f;
        x = Y.s_f;
    }

    @CodeReflection
    @IR("""
            func @"test21" (%0 : FieldAccessTest, %1 : FieldAccessTest$Y)void -> {
                %2 : Var<FieldAccessTest$Y> = var %1 @"y";
                %3 : FieldAccessTest$Y = var.load %2;
                %4 : int = constant @"1";
                field.store %3 %4 @"FieldAccessTest$Y::f()int";
                %5 : FieldAccessTest$Y = var.load %2;
                %6 : int = constant @"1";
                field.store %5 %6 @"FieldAccessTest$Y::yf()int";
                %7 : FieldAccessTest$Y = var.load %2;
                %8 : int = constant @"1";
                field.store %8 @"FieldAccessTest$Y::s_yf()int";
                %9 : int = constant @"1";
                field.store %9 @"FieldAccessTest$Y::s_yf()int";
                %10 : FieldAccessTest$Y = var.load %2;
                %11 : int = constant @"1";
                field.store %11 @"FieldAccessTest$Y::s_f()int";
                %12 : int = constant @"1";
                field.store %12 @"FieldAccessTest$Y::s_f()int";
                return;
            };
            """)
    void test21(Y y) {
        y.f = 1;
        y.yf = 1;
        y.s_yf = 1;
        Y.s_yf = 1;
        y.s_f = 1;
        Y.s_f = 1;
    }

    @CodeReflection
    @IR("""
          func @"test22" (%0 : FieldAccessTest, %1 : FieldAccessTest$Y)void -> {
                %2 : Var<FieldAccessTest$Y> = var %1 @"y";
                %3 : FieldAccessTest$Y = var.load %2;
                %4 : int = field.load %3 @"FieldAccessTest$Y::f()int";
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                field.store %3 %6 @"FieldAccessTest$Y::f()int";
                %7 : FieldAccessTest$Y = var.load %2;
                %8 : int = field.load %7 @"FieldAccessTest$Y::yf()int";
                %9 : int = constant @"1";
                %10 : int = add %8 %9;
                field.store %7 %10 @"FieldAccessTest$Y::yf()int";
                %11 : FieldAccessTest$Y = var.load %2;
                %12 : int = field.load @"FieldAccessTest$Y::s_yf()int";
                %13 : int = constant @"1";
                %14 : int = add %12 %13;
                field.store %14 @"FieldAccessTest$Y::s_yf()int";
                %15 : int = field.load @"FieldAccessTest$Y::s_yf()int";
                %16 : int = constant @"1";
                %17 : int = add %15 %16;
                field.store %17 @"FieldAccessTest$Y::s_yf()int";
                %18 : FieldAccessTest$Y = var.load %2;
                %19 : int = field.load @"FieldAccessTest$Y::s_f()int";
                %20 : int = constant @"1";
                %21 : int = add %19 %20;
                field.store %21 @"FieldAccessTest$Y::s_f()int";
                %22 : int = field.load @"FieldAccessTest$Y::s_f()int";
                %23 : int = constant @"1";
                %24 : int = add %22 %23;
                field.store %24 @"FieldAccessTest$Y::s_f()int";
                return;
            };
            """)
    void test22(Y y) {
        y.f++;
        y.yf++;
        y.s_yf++;
        Y.s_yf++;
        y.s_f++;
        Y.s_f++;
    }

    // @@@ Should propagate as constant value?
    @CodeReflection
    @IR("""
            func @"test23" (%0 : FieldAccessTest)void -> {
                %1 : int = field.load @"java.util.Spliterator$OfInt::CONCURRENT()int";
                %2 : Var<int> = var %1 @"x";
                %3 : int = field.load @"java.util.Spliterator$OfInt::CONCURRENT()int";
                var.store %2 %3;
                %4 : int = field.load @"java.util.Spliterator$OfInt::CONCURRENT()int";
                var.store %2 %4;
                return;
            };
            """)
    void test23() {
        int x = Spliterator.OfInt.CONCURRENT;
        x = OfInt.CONCURRENT;
        x = CONCURRENT;
    }

    @CodeReflection
    @IR("""
            func @"test24" (%0 : FieldAccessTest)void -> {
                %1 : java.io.PrintStream = field.load @"java.lang.System::out()java.io.PrintStream";
                %2 : Var<java.io.PrintStream> = var %1 @"ps";
                return;
            };
            """)
    void test24() {
        PrintStream ps = out;
    }
}
