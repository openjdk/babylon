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
import java.util.List;

/*
 * @test
 * @summary Smoke test for non-denotable types in IR type descriptors
 * @build DenotableTypesTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester DenotableTypesTest
 */

public class DenotableTypesTest {
    static <X extends Number & Runnable> X m1(X x) { return null; }
    @CodeReflection
    @IR("""
            func @"test1" ()void -> {
                  %0 : &<java.lang.Number, java.lang.Runnable> = constant @null;
                  %1 : &<java.lang.Number, java.lang.Runnable> = invoke %0 @"DenotableTypesTest::m1(java.lang.Number)java.lang.Number";
                  return;
            };
            """)
    static void test1() {
        m1(null);
    }

    @CodeReflection
    @IR("""
            func @"test2" ()void -> {
                  %0 : int = constant @"1";
                  %1 : java.lang.Integer = invoke %0 @"java.lang.Integer::valueOf(int)java.lang.Integer";
                  %2 : double = constant @"3.0";
                  %3 : java.lang.Double = invoke %2 @"java.lang.Double::valueOf(double)java.lang.Double";
                  %4 : java.util.List<&<java.lang.Number, java.lang.Comparable<+<&<java.lang.Number, java.lang.Comparable<+<java.lang.Object>>, java.lang.constant.Constable, java.lang.constant.ConstantDesc>>>, java.lang.constant.Constable, java.lang.constant.ConstantDesc>> = invoke %1 %3 @"java.util.List::of(java.lang.Object, java.lang.Object)java.util.List";
                  return;
            };
            """)
    static void test2() {
        List.of(1, 3d); // infinite type! (List<Object & Serializable & Comparable<...>>)
    }

    static <X extends Throwable> X m2(X x) throws X { return null; }

    @CodeReflection
    @IR("""
            func @"test3" ()void -> {
                %0 : java.lang.RuntimeException = constant @null;
                %1 : java.lang.RuntimeException = invoke %0 @"DenotableTypesTest::m2(java.lang.Throwable)java.lang.Throwable";
                return;
            };
            """)
    static void test3() { // @@@ cast?
        m2(null);
    }

    interface A { }
    interface B { }
    static class C implements A, B { }
    static class D implements A, B { }

    static <Z> Z pick(Z z1, Z z2) { return null; }

    @CodeReflection
    @IR("""
            func @"test4" ()void -> {
                  %0 : java.lang.Object = constant @null;
                  %1 : DenotableTypesTest$C = cast %0 @"DenotableTypesTest$C";
                  %2 : java.lang.Object = constant @null;
                  %3 : DenotableTypesTest$D = cast %2 @"DenotableTypesTest$D";
                  %4 : &<java.lang.Object, DenotableTypesTest$A, DenotableTypesTest$B> = invoke %1 %3 @"DenotableTypesTest::pick(java.lang.Object, java.lang.Object)java.lang.Object";
                  return;
            };
            """)
    static void test4() { // @@@ cast?
        pick((C)null, (D)null);
    }

    @CodeReflection
    @IR("""
            func @"test5" ()void -> {
                  %0 : java.util.List<+<java.lang.Number>> = constant @null;
                  %1 : Var<java.util.List<+<java.lang.Number>>> = var %0 @"l";
                  %2 : java.util.List<+<java.lang.Number>> = var.load %1;
                  %3 : int = constant @"0";
                  %4 : java.lang.Number = invoke %2 %3 @"java.util.List::get(int)java.lang.Object";
                  return;
            };
            """)
    static void test5() { // @@@ cast?
        List<? extends Number> l = null;
        l.get(0);
    }

    @CodeReflection
    @IR("""
            func @"test6" ()void -> {
                  %0 : java.util.List<-<java.lang.Number>> = constant @null;
                  %1 : Var<java.util.List<-<java.lang.Number>>> = var %0 @"l";
                  %2 : java.util.List<-<java.lang.Number>> = var.load %1;
                  %3 : int = constant @"0";
                  %4 : java.lang.Object = invoke %2 %3 @"java.util.List::get(int)java.lang.Object";
                  return;
            };
            """)
    static void test6() {
        List<? super Number> l = null;
        l.get(0);
    }

    static void consume(Runnable r) { }

    @CodeReflection
    @IR("""
            func @"test7" ()void -> {
                  %0 : ::X = constant @null;
                  %1 : Var<::X> = var %0 @"x";
                  %2 : ::X = var.load %1;
                  %3 : java.lang.Runnable = cast %2 @"java.lang.Runnable";
                  invoke %3 @"DenotableTypesTest::consume(java.lang.Runnable)void";
                  return;
            };
            """)
    static <X extends Object & Runnable> void test7() {
        X x = null;
        consume(x);
    }

    interface Adder<X> {
        void add(Adder<X> adder);
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : java.util.List<+<DenotableTypesTest$Adder<java.lang.Integer>>>)void -> {
                  %1 : Var<java.util.List<+<DenotableTypesTest$Adder<java.lang.Integer>>>> = var %0 @"list";
                  %2 : java.util.List<+<DenotableTypesTest$Adder<java.lang.Integer>>> = var.load %1;
                  %3 : int = constant @"0";
                  %4 : DenotableTypesTest$Adder<java.lang.Integer> = invoke %2 %3 @"java.util.List::get(int)java.lang.Object";
                  %5 : java.util.List<+<DenotableTypesTest$Adder<java.lang.Integer>>> = var.load %1;
                  %6 : int = constant @"1";
                  %7 : DenotableTypesTest$Adder<java.lang.Integer> = invoke %5 %6 @"java.util.List::get(int)java.lang.Object";
                  invoke %4 %7 @"DenotableTypesTest$Adder::add(DenotableTypesTest$Adder)void";
                  return;
            };
            """)
    static void test8(List<? extends Adder<Integer>> list) {
        list.get(0).add(list.get(1));
    }

    static class Box<X> {
        X x;
    }

    @CodeReflection
    @IR("""
            func @"test9" (%0 : java.util.List<+<DenotableTypesTest$Box<java.lang.Integer>>>)void -> {
                  %1 : Var<java.util.List<+<DenotableTypesTest$Box<java.lang.Integer>>>> = var %0 @"list";
                  %2 : java.util.List<+<DenotableTypesTest$Box<java.lang.Integer>>> = var.load %1;
                  %3 : int = constant @"0";
                  %4 : DenotableTypesTest$Box<java.lang.Integer> = invoke %2 %3 @"java.util.List::get(int)java.lang.Object";
                  %5 : java.lang.Integer = field.load %4 @"DenotableTypesTest$Box<::X>::x()::X";
                  %6 : Var<java.lang.Integer> = var %5 @"i";
                  return;
            };
            """)
    static void test9(List<? extends Box<Integer>> list) {
        Integer i = list.get(0).x;
    }

    static class XA extends Exception {
        public void m() { }
    }

    static class XB extends Exception {
        public void m() { }
    }

    static void g() throws XA, XB { }

    @CodeReflection
    @IR("""
            func @"test10" ()void -> {
                  java.try
                      ()void -> {
                          invoke @"DenotableTypesTest::g()void";
                          yield;
                      }
                      (%0 : |<DenotableTypesTest$XA, DenotableTypesTest$XB>)void -> {
                          %1 : Var<|<DenotableTypesTest$XA, DenotableTypesTest$XB>> = var %0 @"x";
                          %2 : |<DenotableTypesTest$XA, DenotableTypesTest$XB> = var.load %1;
                          %3 : java.lang.Throwable = invoke %2 @"java.lang.Exception::getCause()java.lang.Throwable";
                          yield;
                      };
                  return;
            };
            """)
    static void test10() {
        try {
            g();
        } catch (XA | XB x) {
            x.getCause();
        }
    }
}
