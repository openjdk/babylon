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
 * @summary Smoke test for code reflection with intersection type conversions.
 * @build IntersectionTypeTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester IntersectionTypeTest
 */

class IntersectionTypeTest {
    interface A {
        Object f_A = 5;
        void m_A();
    }

    interface B {
        Object f_B = 5;
        void m_B();
    }

    interface C {
        Object f_C = 5;
        void m_C();
    }

    @CodeReflection
    @IR("""
              func @"test1" (%0 : ::X)void -> {
                    %1 : Var<::X> = var %0 @"x";
                    %2 : ::X = var.load %1;
                    invoke %2 @"IntersectionTypeTest$A::m_A()void";
                    %3 : ::X = var.load %1;
                    %4 : IntersectionTypeTest$B = cast %3 @"IntersectionTypeTest$B";
                    invoke %4 @"IntersectionTypeTest$B::m_B()void";
                    %5 : ::X = var.load %1;
                    %6 : IntersectionTypeTest$C = cast %5 @"IntersectionTypeTest$C";
                    invoke %6 @"IntersectionTypeTest$C::m_C()void";
                    return;
              };
            """)
    static <X extends A & B & C> void test1(X x) {
        x.m_A();
        x.m_B();
        x.m_C();
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : ::X)void -> {
                  %1 : Var<::X> = var %0 @"x";
                  %2 : ::X = var.load %1;
                  %3 : java.lang.Object = field.load @"IntersectionTypeTest$A::f_A()java.lang.Object";
                  %4 : Var<java.lang.Object> = var %3 @"oA";
                  %5 : ::X = var.load %1;
                  %6 : java.lang.Object = field.load @"IntersectionTypeTest$B::f_B()java.lang.Object";
                  %7 : Var<java.lang.Object> = var %6 @"oB";
                  %8 : ::X = var.load %1;
                  %9 : java.lang.Object = field.load @"IntersectionTypeTest$C::f_C()java.lang.Object";
                  %10 : Var<java.lang.Object> = var %9 @"oC";
                  return;
            };
            """)
    static <X extends A & B & C> void test2(X x) {
        Object oA = x.f_A;
        Object oB = x.f_B;
        Object oC = x.f_C;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : ::X)void -> {
                  %1 : Var<::X> = var %0 @"x";
                  %2 : ::X = var.load %1;
                  %3 : Var<IntersectionTypeTest$A> = var %2 @"rec$";
                  %4 : java.lang.Runnable = lambda ()void -> {
                      %5 : IntersectionTypeTest$A = var.load %3;
                      invoke %5 @"IntersectionTypeTest$A::m_A()void";
                      return;
                  };
                  %6 : Var<java.lang.Runnable> = var %4 @"rA";
                  %7 : ::X = var.load %1;
                  %8 : IntersectionTypeTest$B = cast %7 @"IntersectionTypeTest$B";
                  %9 : Var<IntersectionTypeTest$B> = var %8 @"rec$";
                  %10 : java.lang.Runnable = lambda ()void -> {
                      %11 : IntersectionTypeTest$B = var.load %9;
                      invoke %11 @"IntersectionTypeTest$B::m_B()void";
                      return;
                  };
                  %12 : Var<java.lang.Runnable> = var %10 @"rB";
                  %13 : ::X = var.load %1;
                  %14 : IntersectionTypeTest$C = cast %13 @"IntersectionTypeTest$C";
                  %15 : Var<IntersectionTypeTest$C> = var %14 @"rec$";
                  %16 : java.lang.Runnable = lambda ()void -> {
                      %17 : IntersectionTypeTest$C = var.load %15;
                      invoke %17 @"IntersectionTypeTest$C::m_C()void";
                      return;
                  };
                  %18 : Var<java.lang.Runnable> = var %16 @"rC";
                  return;
            };
            """)
    static <X extends A & B & C> void test3(X x) {
        Runnable rA = x::m_A;
        Runnable rB = x::m_B;
        Runnable rC = x::m_C;
    }

    static void g_A(A a) { }
    static void g_B(B a) { }
    static void g_C(C a) { }

    @CodeReflection
    @IR("""
              func @"test4" (%0 : ::X)void -> {
                    %1 : Var<::X> = var %0 @"x";
                    %2 : ::X = var.load %1;
                    invoke %2 @"IntersectionTypeTest::g_A(IntersectionTypeTest$A)void";
                    %3 : ::X = var.load %1;
                    %4 : IntersectionTypeTest$B = cast %3 @"IntersectionTypeTest$B";
                    invoke %4 @"IntersectionTypeTest::g_B(IntersectionTypeTest$B)void";
                    %5 : ::X = var.load %1;
                    %6 : IntersectionTypeTest$C = cast %5 @"IntersectionTypeTest$C";
                    invoke %6 @"IntersectionTypeTest::g_C(IntersectionTypeTest$C)void";
                    return;
              };
            """)
    static <X extends A & B & C> void test4(X x) {
        g_A(x);
        g_B(x);
        g_C(x);
    }

    static <X extends A & B & C> X makeIntersection(X x1, X x2) {
        return null;
    }

    class E1 implements A, B, C {
        @Override
        public void m_A() { }
        @Override
        public void m_B() { }
        @Override
        public void m_C() { }
    }

    class E2 implements A, B, C {
        @Override
        public void m_A() { }
        @Override
        public void m_B() { }
        @Override
        public void m_C() { }
    }

    @CodeReflection
    @IR("""
              func @"test5" (%0 : IntersectionTypeTest$E1, %1 : IntersectionTypeTest$E2)void -> {
                    %2 : Var<IntersectionTypeTest$E1> = var %0 @"e1";
                    %3 : Var<IntersectionTypeTest$E2> = var %1 @"e2";
                    %4 : IntersectionTypeTest$E1 = var.load %2;
                    %5 : IntersectionTypeTest$E2 = var.load %3;
                    %6 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = invoke %4 %5 @"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A)IntersectionTypeTest$A";
                    %7 : Var<&<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C>> = var %6 @"x";
                    %8 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                    invoke %8 @"IntersectionTypeTest$A::m_A()void";
                    %9 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                    %10 : IntersectionTypeTest$B = cast %9 @"IntersectionTypeTest$B";
                    invoke %10 @"IntersectionTypeTest$B::m_B()void";
                    %11 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                    %12 : IntersectionTypeTest$C = cast %11 @"IntersectionTypeTest$C";
                    invoke %12 @"IntersectionTypeTest$C::m_C()void";
                    return;
              };
            """)
    static void test5(E1 e1, E2 e2) {
        var x = makeIntersection(e1, e2);
        x.m_A();
        x.m_B();
        x.m_C();
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : IntersectionTypeTest$E1, %1 : IntersectionTypeTest$E2)void -> {
                  %2 : Var<IntersectionTypeTest$E1> = var %0 @"e1";
                  %3 : Var<IntersectionTypeTest$E2> = var %1 @"e2";
                  %4 : IntersectionTypeTest$E1 = var.load %2;
                  %5 : IntersectionTypeTest$E2 = var.load %3;
                  %6 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = invoke %4 %5 @"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A)IntersectionTypeTest$A";
                  %7 : Var<&<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C>> = var %6 @"x";
                  %8 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %9 : java.lang.Object = field.load @"IntersectionTypeTest$A::f_A()java.lang.Object";
                  %10 : Var<java.lang.Object> = var %9 @"oA";
                  %11 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %12 : java.lang.Object = field.load @"IntersectionTypeTest$B::f_B()java.lang.Object";
                  %13 : Var<java.lang.Object> = var %12 @"oB";
                  %14 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %15 : java.lang.Object = field.load @"IntersectionTypeTest$C::f_C()java.lang.Object";
                  %16 : Var<java.lang.Object> = var %15 @"oC";
                  return;
            };
            """)
    static void test6(E1 e1, E2 e2) {
        var x = makeIntersection(e1, e2);
        Object oA = x.f_A;
        Object oB = x.f_B;
        Object oC = x.f_C;
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : IntersectionTypeTest$E1, %1 : IntersectionTypeTest$E2)void -> {
                  %2 : Var<IntersectionTypeTest$E1> = var %0 @"e1";
                  %3 : Var<IntersectionTypeTest$E2> = var %1 @"e2";
                  %4 : IntersectionTypeTest$E1 = var.load %2;
                  %5 : IntersectionTypeTest$E2 = var.load %3;
                  %6 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = invoke %4 %5 @"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A)IntersectionTypeTest$A";
                  %7 : Var<&<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C>> = var %6 @"x";
                  %8 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %9 : Var<IntersectionTypeTest$A> = var %8 @"rec$";
                  %10 : java.lang.Runnable = lambda ()void -> {
                      %11 : IntersectionTypeTest$A = var.load %9;
                      invoke %11 @"IntersectionTypeTest$A::m_A()void";
                      return;
                  };
                  %12 : Var<java.lang.Runnable> = var %10 @"rA";
                  %13 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %14 : IntersectionTypeTest$B = cast %13 @"IntersectionTypeTest$B";
                  %15 : Var<IntersectionTypeTest$B> = var %14 @"rec$";
                  %16 : java.lang.Runnable = lambda ()void -> {
                      %17 : IntersectionTypeTest$B = var.load %15;
                      invoke %17 @"IntersectionTypeTest$B::m_B()void";
                      return;
                  };
                  %18 : Var<java.lang.Runnable> = var %16 @"rB";
                  %19 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %20 : IntersectionTypeTest$C = cast %19 @"IntersectionTypeTest$C";
                  %21 : Var<IntersectionTypeTest$C> = var %20 @"rec$";
                  %22 : java.lang.Runnable = lambda ()void -> {
                      %23 : IntersectionTypeTest$C = var.load %21;
                      invoke %23 @"IntersectionTypeTest$C::m_C()void";
                      return;
                  };
                  %24 : Var<java.lang.Runnable> = var %22 @"rC";
                  return;
            };
            """)
    static void test7(E1 e1, E2 e2) {
        var x = makeIntersection(e1, e2);
        Runnable rA = x::m_A;
        Runnable rB = x::m_B;
        Runnable rC = x::m_C;
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : IntersectionTypeTest$E1, %1 : IntersectionTypeTest$E2)void -> {
                  %2 : Var<IntersectionTypeTest$E1> = var %0 @"e1";
                  %3 : Var<IntersectionTypeTest$E2> = var %1 @"e2";
                  %4 : IntersectionTypeTest$E1 = var.load %2;
                  %5 : IntersectionTypeTest$E2 = var.load %3;
                  %6 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = invoke %4 %5 @"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A)IntersectionTypeTest$A";
                  %7 : Var<&<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C>> = var %6 @"x";
                  %8 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  invoke %8 @"IntersectionTypeTest::g_A(IntersectionTypeTest$A)void";
                  %9 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %10 : IntersectionTypeTest$B = cast %9 @"IntersectionTypeTest$B";
                  invoke %10 @"IntersectionTypeTest::g_B(IntersectionTypeTest$B)void";
                  %11 : &<java.lang.Object, IntersectionTypeTest$A, IntersectionTypeTest$B, IntersectionTypeTest$C> = var.load %7;
                  %12 : IntersectionTypeTest$C = cast %11 @"IntersectionTypeTest$C";
                  invoke %12 @"IntersectionTypeTest::g_C(IntersectionTypeTest$C)void";
                  return;
            };
            """)
    static void test8(E1 e1, E2 e2) {
        var x = makeIntersection(e1, e2);
        g_A(x);
        g_B(x);
        g_C(x);
    }
}
