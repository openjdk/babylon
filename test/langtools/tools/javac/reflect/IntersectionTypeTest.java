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
 * @summary Smoke test for code reflection with intersection type conversions.
 * @modules jdk.incubator.code
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
            func @"test1" (%0 : java.type:"&IntersectionTypeTest::test1(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)")java.type:"void" -> {
                %1 : Var<java.type:"&IntersectionTypeTest::test1(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>"> = var %0 @"x";
                %2 : java.type:"&IntersectionTypeTest::test1(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                invoke %2 @java.ref:"IntersectionTypeTest$A::m_A():void";
                %3 : java.type:"&IntersectionTypeTest::test1(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                %4 : java.type:"IntersectionTypeTest$B" = cast %3 @java.type:"IntersectionTypeTest$B";
                invoke %4 @java.ref:"IntersectionTypeTest$B::m_B():void";
                %5 : java.type:"&IntersectionTypeTest::test1(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                %6 : java.type:"IntersectionTypeTest$C" = cast %5 @java.type:"IntersectionTypeTest$C";
                invoke %6 @java.ref:"IntersectionTypeTest$C::m_C():void";
                return;
            };
            """)
    static <X extends A & B & C> void test1(X x) {
        x.m_A();
        x.m_B();
        x.m_C();
    }

    // #X<&m<IntersectionTypeTest, test2, func<void, IntersectionTypeTest$A>>, IntersectionTypeTest$A>
    // #X<&m<IntersectionTypeTest, test2, func<void, IntersectionTypeTest$A>, IntersectionTypeTest$A>
    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"&IntersectionTypeTest::test2(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)")java.type:"void" -> {
                %1 : Var<java.type:"&IntersectionTypeTest::test2(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)"> = var %0 @"x";
                %2 : java.type:"IntersectionTypeTest::test2(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %3 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$A::f_A:java.lang.Object";
                %4 : Var<java.type:"java.lang.Object"> = var %3 @"oA";
                %5 : java.type:"&IntersectionTypeTest::test2(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %6 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$B::f_B:java.lang.Object";
                %7 : Var<java.type:"java.lang.Object"> = var %6 @"oB";
                %8 : java.type:"&IntersectionTypeTest::test2(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %9 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$C::f_C:java.lang.Object";
                %10 : Var<java.type:"java.lang.Object"> = var %9 @"oC";
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
            func @"test3" (%0 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)")java.type:"void" -> {
                %1 : Var<java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)"> = var %0 @"x";
                %2 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %3 : Var<java.type:"IntersectionTypeTest$A"> = var %2 @"rec$";
                %4 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %5 : java.type:"IntersectionTypeTest$A" = var.load %3;
                    invoke %5 @java.ref:"IntersectionTypeTest$A::m_A():void";
                    return;
                };
                %6 : Var<java.type:"java.lang.Runnable"> = var %4 @"rA";
                %7 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %8 : java.type:"IntersectionTypeTest$B" = cast %7 @java.type:"IntersectionTypeTest$B";
                %9 : Var<java.type:"IntersectionTypeTest$B"> = var %8 @"rec$";
                %10 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %11 : java.type:"IntersectionTypeTest$B" = var.load %9;
                    invoke %11 @java.ref:"IntersectionTypeTest$B::m_B():void";
                    return;
                };
                %12 : Var<java.type:"java.lang.Runnable"> = var %10 @"rB";
                %13 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %14 : java.type:"IntersectionTypeTest$C" = cast %13 @java.type:"IntersectionTypeTest$C";
                %15 : Var<java.type:"IntersectionTypeTest$C"> = var %14 @"rec$";
                %16 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %17 : java.type:"IntersectionTypeTest$C" = var.load %15;
                    invoke %17 @java.ref:"IntersectionTypeTest$C::m_C():void";
                    return;
                };
                %18 : Var<java.type:"java.lang.Runnable"> = var %16 @"rC";
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
            func @"test4" (%0 : java.type:"&IntersectionTypeTest::test4(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)")java.type:"void" -> {
                %1 : Var<java.type:"&IntersectionTypeTest::test4(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)"> = var %0 @"x";
                %2 : java.type:"&IntersectionTypeTest::test4(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                invoke %2 @java.ref:"IntersectionTypeTest::g_A(IntersectionTypeTest$A):void";
                %3 : java.type:"&IntersectionTypeTest::test4(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %4 : java.type:"IntersectionTypeTest$B" = cast %3 @java.type:"IntersectionTypeTest$B";
                invoke %4 @java.ref:"IntersectionTypeTest::g_B(IntersectionTypeTest$B):void";
                %5 : java.type:"&IntersectionTypeTest::test4(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>)" = var.load %1;
                %6 : java.type:"IntersectionTypeTest$C" = cast %5 @java.type:"IntersectionTypeTest$C";
                invoke %6 @java.ref:"IntersectionTypeTest::g_C(IntersectionTypeTest$C):void";
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

    static class E1 implements A, B, C {
        @Override
        public void m_A() { }
        @Override
        public void m_B() { }
        @Override
        public void m_C() { }
    }

    static class E2 implements A, B, C {
        @Override
        public void m_A() { }
        @Override
        public void m_B() { }
        @Override
        public void m_C() { }
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"IntersectionTypeTest$E1", %1 : java.type:"IntersectionTypeTest$E2")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$E1"> = var %0 @"e1";
                %3 : Var<java.type:"IntersectionTypeTest$E2"> = var %1 @"e2";
                %4 : java.type:"IntersectionTypeTest$E1" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$E2" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$A" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A):IntersectionTypeTest$A";
                %7 : Var<java.type:"IntersectionTypeTest$A"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$A" = var.load %7;
                invoke %8 @java.ref:"IntersectionTypeTest$A::m_A():void";
                %9 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %10 : java.type:"IntersectionTypeTest$B" = cast %9 @java.type:"IntersectionTypeTest$B";
                invoke %10 @java.ref:"IntersectionTypeTest$B::m_B():void";
                %11 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %12 : java.type:"IntersectionTypeTest$C" = cast %11 @java.type:"IntersectionTypeTest$C";
                invoke %12 @java.ref:"IntersectionTypeTest$C::m_C():void";
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
            func @"test6" (%0 : java.type:"IntersectionTypeTest$E1", %1 : java.type:"IntersectionTypeTest$E2")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$E1"> = var %0 @"e1";
                %3 : Var<java.type:"IntersectionTypeTest$E2"> = var %1 @"e2";
                %4 : java.type:"IntersectionTypeTest$E1" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$E2" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$A" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A):IntersectionTypeTest$A";
                %7 : Var<java.type:"IntersectionTypeTest$A"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %9 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$A::f_A:java.lang.Object";
                %10 : Var<java.type:"java.lang.Object"> = var %9 @"oA";
                %11 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %12 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$B::f_B:java.lang.Object";
                %13 : Var<java.type:"java.lang.Object"> = var %12 @"oB";
                %14 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %15 : java.type:"java.lang.Object" = field.load @java.ref:"IntersectionTypeTest$C::f_C:java.lang.Object";
                %16 : Var<java.type:"java.lang.Object"> = var %15 @"oC";
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
            func @"test7" (%0 : java.type:"IntersectionTypeTest$E1", %1 : java.type:"IntersectionTypeTest$E2")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$E1"> = var %0 @"e1";
                %3 : Var<java.type:"IntersectionTypeTest$E2"> = var %1 @"e2";
                %4 : java.type:"IntersectionTypeTest$E1" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$E2" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$A" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A):IntersectionTypeTest$A";
                %7 : Var<java.type:"IntersectionTypeTest$A"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %9 : Var<java.type:"IntersectionTypeTest$A"> = var %8 @"rec$";
                %10 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %11 : java.type:"IntersectionTypeTest$A" = var.load %9;
                    invoke %11 @java.ref:"IntersectionTypeTest$A::m_A():void";
                    return;
                };
                %12 : Var<java.type:"java.lang.Runnable"> = var %10 @"rA";
                %13 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %14 : java.type:"IntersectionTypeTest$B" = cast %13 @java.type:"IntersectionTypeTest$B";
                %15 : Var<java.type:"IntersectionTypeTest$B"> = var %14 @"rec$";
                %16 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %17 : java.type:"IntersectionTypeTest$B" = var.load %15;
                    invoke %17 @java.ref:"IntersectionTypeTest$B::m_B():void";
                    return;
                };
                %18 : Var<java.type:"java.lang.Runnable"> = var %16 @"rB";
                %19 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %20 : java.type:"IntersectionTypeTest$C" = cast %19 @java.type:"IntersectionTypeTest$C";
                %21 : Var<java.type:"IntersectionTypeTest$C"> = var %20 @"rec$";
                %22 : java.type:"java.lang.Runnable" = lambda ()java.type:"void" -> {
                    %23 : java.type:"IntersectionTypeTest$C" = var.load %21;
                    invoke %23 @java.ref:"IntersectionTypeTest$C::m_C():void";
                    return;
                };
                %24 : Var<java.type:"java.lang.Runnable"> = var %22 @"rC";
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
            func @"test8" (%0 : java.type:"IntersectionTypeTest$E1", %1 : java.type:"IntersectionTypeTest$E2")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$E1"> = var %0 @"e1";
                %3 : Var<java.type:"IntersectionTypeTest$E2"> = var %1 @"e2";
                %4 : java.type:"IntersectionTypeTest$E1" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$E2" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$A" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A):IntersectionTypeTest$A";
                %7 : Var<java.type:"IntersectionTypeTest$A"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$A" = var.load %7;
                invoke %8 @java.ref:"IntersectionTypeTest::g_A(IntersectionTypeTest$A):void";
                %9 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %10 : java.type:"IntersectionTypeTest$B" = cast %9 @java.type:"IntersectionTypeTest$B";
                invoke %10 @java.ref:"IntersectionTypeTest::g_B(IntersectionTypeTest$B):void";
                %11 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %12 : java.type:"IntersectionTypeTest$C" = cast %11 @java.type:"IntersectionTypeTest$C";
                invoke %12 @java.ref:"IntersectionTypeTest::g_C(IntersectionTypeTest$C):void";
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
