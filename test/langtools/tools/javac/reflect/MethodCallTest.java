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
import java.util.ArrayList;
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with method calls.
 * @modules jdk.incubator.code
 * @build MethodCallTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester MethodCallTest
 */

public class MethodCallTest {

    void m() {
    }

    int m_int() {
        return 0;
    }

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"MethodCallTest")java.type:"void" -> {
                invoke %0 @java.ref:"MethodCallTest::m():void";
                return;
            };
            """)
    void test1() {
        m();
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"MethodCallTest")java.type:"void" -> {
                invoke %0 @java.ref:"MethodCallTest::m():void";
                return;
            };
            """)
    void test2() {
        this.m();
    }

    @CodeReflection
    @IR("""
            func @"test2_1" (%0 : java.type:"MethodCallTest")java.type:"void" -> {
                invoke %0 @java.ref:"MethodCallTest::m():void";
                return;
            };
            """)
    void test2_1() {
        MethodCallTest.this.m();
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : java.type:"MethodCallTest")java.type:"int" -> {
                %1 : java.type:"int" = invoke %0 @java.ref:"MethodCallTest::m_int():int";
                return %1;
            };
            """)
    int test3() {
        return m_int();
    }


    static void ms() {
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : java.type:"MethodCallTest")java.type:"void" -> {
                invoke @java.ref:"MethodCallTest::ms():void";
                return;
            };
            """)
    void test4() {
        ms();
    }

    @CodeReflection
    @IR("""
            func @"test4_1" (%0 : java.type:"MethodCallTest")java.type:"void" -> {
                invoke @java.ref:"MethodCallTest::ms():void";
                return;
            };
            """)
    void test4_1() {
        MethodCallTest.ms();
    }

    @CodeReflection
    @IR("""
            func @"test4_2" (%0 : java.type:"MethodCallTest")java.type:"java.util.List<java.lang.String>" -> {
                %1 : java.type:"java.util.List<java.lang.String>" = invoke @java.ref:"java.util.List::of():java.util.List";
                return %1;
            };
            """)
    List<String> test4_2() {
        return List.of();
    }

    String m(int i, String s, List<Number> l) {
        return s;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : java.type:"MethodCallTest", %1 : java.type:"java.util.List<java.lang.Number>")java.type:"void" -> {
                %2 : Var<java.type:"java.util.List<java.lang.Number>"> = var %1 @"l";
                %3 : java.type:"int" = constant @1;
                %4 : java.type:"java.lang.String" = constant @"1";
                %5 : java.type:"java.util.List<java.lang.Number>" = var.load %2;
                %6 : java.type:"java.lang.String" = invoke %0 %3 %4 %5 @java.ref:"MethodCallTest::m(int, java.lang.String, java.util.List):java.lang.String";
                %7 : Var<java.type:"java.lang.String"> = var %6 @"s";
                return;
            };
            """)
    void test5(List<Number> l) {
        String s = m(1, "1", l);
    }


    static class A {
        B b;

        B m() {
            return null;
        }
    }

    static class B {
        C m() {
            return null;
        }
    }

    static class C {
        int m() {
            return 0;
        }
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : java.type:"MethodCallTest", %1 : java.type:"MethodCallTest$A")java.type:"void" -> {
                %2 : Var<java.type:"MethodCallTest$A"> = var %1 @"a";
                %3 : java.type:"MethodCallTest$A" = var.load %2;
                %4 : java.type:"MethodCallTest$B" = invoke %3 @java.ref:"MethodCallTest$A::m():MethodCallTest$B";
                %5 : java.type:"MethodCallTest$C" = invoke %4 @java.ref:"MethodCallTest$B::m():MethodCallTest$C";
                %6 : java.type:"int" = invoke %5 @java.ref:"MethodCallTest$C::m():int";
                return;
            };
            """)
    void test6(A a) {
        a.m().m().m();
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : java.type:"MethodCallTest", %1 : java.type:"MethodCallTest$A")java.type:"void" -> {
                %2 : Var<java.type:"MethodCallTest$A"> = var %1 @"a";
                %3 : java.type:"MethodCallTest$A" = var.load %2;
                %4 : java.type:"MethodCallTest$B" = field.load %3 @java.ref:"MethodCallTest$A::b:MethodCallTest$B";
                %5 : java.type:"MethodCallTest$C" = invoke %4 @java.ref:"MethodCallTest$B::m():MethodCallTest$C";
                return;
            };
            """)
    void test7(A a) {
        a.b.m();
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.type:"MethodCallTest", %1 : java.type:"java.lang.String")java.type:"void" -> {
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.io.PrintStream" = field.load @java.ref:"java.lang.System::out:java.io.PrintStream";
                %4 : java.type:"java.lang.String" = var.load %2;
                invoke %3 %4 @java.ref:"java.io.PrintStream::println(java.lang.String):void";
                return;
            };
            """)
    void test8(String s) {
        System.out.println(s);
    }

    static class X {
        int x;
        void x() {}

        static void sx() {}
    }

    static class Y extends X {
        void y() {}
        static void sy() {}

        @CodeReflection
        @IR("""
                func @"test" (%0 : java.type:"MethodCallTest$Y")java.type:"void" -> {
                    invoke %0 @java.ref:"MethodCallTest$Y::x():void";
                    invoke %0 @java.ref:"MethodCallTest$Y::y():void";
                    invoke @java.ref:"MethodCallTest$Y::sx():void";
                    invoke @java.ref:"MethodCallTest$Y::sy():void";
                    invoke @java.ref:"MethodCallTest$Y::sx():void";
                    invoke @java.ref:"MethodCallTest$Y::sy():void";
                    return;
                };
                """)
        void test() {
            x();
            y();

            sx();
            sy();

            Y.sx();
            Y.sy();
        }
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.type:"MethodCallTest$Y")java.type:"void" -> {
                %1 : Var<java.type:"MethodCallTest$Y"> = var %0 @"y";
                %2 : java.type:"MethodCallTest$Y" = var.load %1;
                invoke %2 @java.ref:"MethodCallTest$Y::x():void";
                %3 : java.type:"MethodCallTest$Y" = var.load %1;
                invoke %3 @java.ref:"MethodCallTest$Y::y():void";
                %4 : java.type:"MethodCallTest$Y" = var.load %1;
                invoke @java.ref:"MethodCallTest$Y::sx():void";
                %5 : java.type:"MethodCallTest$Y" = var.load %1;
                invoke @java.ref:"MethodCallTest$Y::sy():void";
                invoke @java.ref:"MethodCallTest$Y::sx():void";
                invoke @java.ref:"MethodCallTest$Y::sy():void";
                return;
            };
            """)
    static void test9(Y y) {
        y.x();
        y.y();

        y.sx();
        y.sy();

        Y.sx();
        Y.sy();
    }

    @CodeReflection
    @IR("""
            func @"test10" (%0 : java.type:"java.util.ArrayList<java.lang.String>")java.type:"void" -> {
                %1 : Var<java.type:"java.util.ArrayList<java.lang.String>"> = var %0 @"al";
                %2 : java.type:"java.util.ArrayList<java.lang.String>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"java.lang.String" = invoke %2 %3 @java.ref:"java.util.ArrayList::get(int):java.lang.Object";
                %5 : Var<java.type:"java.lang.String"> = var %4 @"s";
                %6 : java.type:"java.util.ArrayList<java.lang.String>" = var.load %1;
                %7 : Var<java.type:"java.util.List<java.lang.String>"> = var %6 @"l";
                %8 : java.type:"java.util.List<java.lang.String>" = var.load %7;
                %9 : java.type:"int" = constant @0;
                %10 : java.type:"java.lang.String" = invoke %8 %9 @java.ref:"java.util.List::get(int):java.lang.Object";
                var.store %5 %10;
                return;
            };
            """)
    static void test10(ArrayList<String> al) {
        String s = al.get(0);
        List<String> l = al;
        s = l.get(0);
    }
}
