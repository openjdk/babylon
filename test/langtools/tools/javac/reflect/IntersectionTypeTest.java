/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;

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

    @Reflect
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
    @Reflect
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

    @Reflect
    @IR("""
            func @"test3" (%0 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>")java.type:"void" -> {
                %1 : Var<java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>"> = var %0 @"x";
                %2 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                %3 : java.type:"java.lang.Object" = invoke %2 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %4 : Var<java.type:"IntersectionTypeTest$A"> = var %2 @"rec$";
                %5 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %6 : java.type:"IntersectionTypeTest$A" = var.load %4;
                    invoke %6 @java.ref:"IntersectionTypeTest$A::m_A():void";
                    return;
                };
                %7 : Var<java.type:"java.lang.Runnable"> = var %5 @"rA";
                %8 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                %9 : java.type:"java.lang.Object" = invoke %8 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %10 : java.type:"IntersectionTypeTest$B" = cast %8 @java.type:"IntersectionTypeTest$B";
                %11 : Var<java.type:"IntersectionTypeTest$B"> = var %10 @"rec$";
                %12 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %13 : java.type:"IntersectionTypeTest$B" = var.load %11;
                    invoke %13 @java.ref:"IntersectionTypeTest$B::m_B():void";
                    return;
                };
                %14 : Var<java.type:"java.lang.Runnable"> = var %12 @"rB";
                %15 : java.type:"&IntersectionTypeTest::test3(IntersectionTypeTest$A):void::<X extends IntersectionTypeTest$A>" = var.load %1;
                %16 : java.type:"java.lang.Object" = invoke %15 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %17 : java.type:"IntersectionTypeTest$C" = cast %15 @java.type:"IntersectionTypeTest$C";
                %18 : Var<java.type:"IntersectionTypeTest$C"> = var %17 @"rec$";
                %19 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %20 : java.type:"IntersectionTypeTest$C" = var.load %18;
                    invoke %20 @java.ref:"IntersectionTypeTest$C::m_C():void";
                    return;
                };
                %21 : Var<java.type:"java.lang.Runnable"> = var %19 @"rC";
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

    @Reflect
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

    @Reflect
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

    @Reflect
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

    @Reflect
    @IR("""
            func @"test7" (%0 : java.type:"IntersectionTypeTest$E1", %1 : java.type:"IntersectionTypeTest$E2")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$E1"> = var %0 @"e1";
                %3 : Var<java.type:"IntersectionTypeTest$E2"> = var %1 @"e2";
                %4 : java.type:"IntersectionTypeTest$E1" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$E2" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$A" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$A, IntersectionTypeTest$A):IntersectionTypeTest$A";
                %7 : Var<java.type:"IntersectionTypeTest$A"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %9 : java.type:"java.lang.Object" = invoke %8 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %10 : Var<java.type:"IntersectionTypeTest$A"> = var %8 @"rec$";
                %11 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %12 : java.type:"IntersectionTypeTest$A" = var.load %10;
                    invoke %12 @java.ref:"IntersectionTypeTest$A::m_A():void";
                    return;
                };
                %13 : Var<java.type:"java.lang.Runnable"> = var %11 @"rA";
                %14 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %15 : java.type:"java.lang.Object" = invoke %14 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %16 : java.type:"IntersectionTypeTest$B" = cast %14 @java.type:"IntersectionTypeTest$B";
                %17 : Var<java.type:"IntersectionTypeTest$B"> = var %16 @"rec$";
                %18 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %19 : java.type:"IntersectionTypeTest$B" = var.load %17;
                    invoke %19 @java.ref:"IntersectionTypeTest$B::m_B():void";
                    return;
                };
                %20 : Var<java.type:"java.lang.Runnable"> = var %18 @"rB";
                %21 : java.type:"IntersectionTypeTest$A" = var.load %7;
                %22 : java.type:"java.lang.Object" = invoke %21 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %23 : java.type:"IntersectionTypeTest$C" = cast %21 @java.type:"IntersectionTypeTest$C";
                %24 : Var<java.type:"IntersectionTypeTest$C"> = var %23 @"rec$";
                %25 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %26 : java.type:"IntersectionTypeTest$C" = var.load %24;
                    invoke %26 @java.ref:"IntersectionTypeTest$C::m_C():void";
                    return;
                };
                %27 : Var<java.type:"java.lang.Runnable"> = var %25 @"rC";
                return;
            };
            """)
    static void test7(E1 e1, E2 e2) {
        var x = makeIntersection(e1, e2);
        Runnable rA = x::m_A;
        Runnable rB = x::m_B;
        Runnable rC = x::m_C;
    }

    @Reflect
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

    static class F {
        void m_F() {
        }
    }

    static class G extends F {
    }

    static <X extends G & A> X makeIntersection(G g, A a) {
        return null;
    }

    @Reflect
    @IR("""
            func @"test9" (%0 : java.type:"IntersectionTypeTest$G", %1 : java.type:"IntersectionTypeTest$A")java.type:"void" -> {
                %2 : Var<java.type:"IntersectionTypeTest$G"> = var %0 @"g";
                %3 : Var<java.type:"IntersectionTypeTest$A"> = var %1 @"a";
                %4 : java.type:"IntersectionTypeTest$G" = var.load %2;
                %5 : java.type:"IntersectionTypeTest$A" = var.load %3;
                %6 : java.type:"IntersectionTypeTest$G" = invoke %4 %5 @java.ref:"IntersectionTypeTest::makeIntersection(IntersectionTypeTest$G, IntersectionTypeTest$A):IntersectionTypeTest$G";
                %7 : Var<java.type:"IntersectionTypeTest$G"> = var %6 @"x";
                %8 : java.type:"IntersectionTypeTest$G" = var.load %7;
                %9 : java.type:"java.lang.Object" = invoke %8 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %10 : Var<java.type:"IntersectionTypeTest$G"> = var %8 @"rec$";
                %11 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %12 : java.type:"IntersectionTypeTest$G" = var.load %10;
                    invoke %12 @java.ref:"IntersectionTypeTest$G::m_F():void";
                    return;
                };
                %13 : Var<java.type:"java.lang.Runnable"> = var %11 @"rG";
                %14 : java.type:"IntersectionTypeTest$G" = var.load %7;
                %15 : java.type:"java.lang.Object" = invoke %14 @java.ref:"java.util.Objects::requireNonNull(java.lang.Object):java.lang.Object";
                %16 : java.type:"IntersectionTypeTest$A" = cast %14 @java.type:"IntersectionTypeTest$A";
                %17 : Var<java.type:"IntersectionTypeTest$A"> = var %16 @"rec$";
                %18 : java.type:"java.lang.Runnable" = lambda @lambda.isReflectable=true ()java.type:"void" -> {
                    %19 : java.type:"IntersectionTypeTest$A" = var.load %17;
                    invoke %19 @java.ref:"IntersectionTypeTest$A::m_A():void";
                    return;
                };
                %20 : Var<java.type:"java.lang.Runnable"> = var %18 @"rA";
                return;
            };
            """)
    static void test9(G g, A a) {
        var x = makeIntersection(g, a);
        Runnable rG = x::m_F;
        Runnable rA = x::m_A;
    }
}
