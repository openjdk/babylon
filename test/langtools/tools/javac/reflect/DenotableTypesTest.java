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

import jdk.incubator.code.Reflect;
import java.util.List;

/*
 * @test
 * @summary Smoke test for non-denotable types in IR type descriptors
 * @modules jdk.incubator.code
 * @build DenotableTypesTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester DenotableTypesTest
 */

public class DenotableTypesTest {
    static <X extends Number & Runnable> X m1(X x) { return null; }
    @Reflect
    @IR("""
            func @"test1" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Number" = constant @null;
                %1 : java.type:"java.lang.Number" = invoke %0 @java.ref:"DenotableTypesTest::m1(java.lang.Number):java.lang.Number";
                return;
            };
            """)
    static void test1() {
        m1(null);
    }

    @Reflect
    @IR("""
            func @"test2" ()java.type:"void" -> {
                %0 : java.type:"int" = constant @1;
                %1 : java.type:"java.lang.Integer" = invoke %0 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %2 : java.type:"double" = constant @3.0d;
                %3 : java.type:"java.lang.Double" = invoke %2 @java.ref:"java.lang.Double::valueOf(double):java.lang.Double";
                %4 : java.type:"java.util.List<? extends java.lang.Number>" = invoke %1 %3 @java.ref:"java.util.List::of(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test2() {
        List.of(1, 3d); // infinite type! (List<Object & Serializable & Comparable<...>>)
    }

    static <X extends Throwable> X m2(X x) throws X { return null; }

    @Reflect
    @IR("""
            func @"test3" ()java.type:"void" -> {
                %0 : java.type:"java.lang.RuntimeException" = constant @null;
                %1 : java.type:"java.lang.RuntimeException" = invoke %0 @java.ref:"DenotableTypesTest::m2(java.lang.Throwable):java.lang.Throwable";
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

    @Reflect
    @IR("""
            func @"test4" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"DenotableTypesTest$A" = invoke %1 %3 @java.ref:"DenotableTypesTest::pick(java.lang.Object, java.lang.Object):java.lang.Object";
                return;
            };
            """)
    static void test4() { // @@@ cast?
        pick((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test5" ()java.type:"void" -> {
                %0 : java.type:"java.util.List<? extends java.lang.Number>" = constant @null;
                %1 : Var<java.type:"java.util.List<? extends java.lang.Number>"> = var %0 @"l";
                %2 : java.type:"java.util.List<? extends java.lang.Number>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"java.lang.Number" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                return;
            };
            """)
    static void test5() { // @@@ cast?
        List<? extends Number> l = null;
        l.get(0);
    }

    @Reflect
    @IR("""
            func @"test6" ()java.type:"void" -> {
                %0 : java.type:"java.util.List<? super java.lang.Number>" = constant @null;
                %1 : Var<java.type:"java.util.List<? super java.lang.Number>"> = var %0 @"l";
                %2 : java.type:"java.util.List<? super java.lang.Number>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"java.lang.Object" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                return;
            };
            """)
    static void test6() {
        List<? super Number> l = null;
        l.get(0);
    }

    static void consume(Runnable r) { }

    @Reflect
    @IR("""
            func @"test7" ()java.type:"void" -> {
                %0 : java.type:"&DenotableTypesTest::test7():void::<X>" = constant @null;
                %1 : Var<java.type:"&DenotableTypesTest::test7():void::<X>"> = var %0 @"x";
                %2 : java.type:"&DenotableTypesTest::test7():void::<X>" = var.load %1;
                %3 : java.type:"java.lang.Runnable" = cast %2 @java.type:"java.lang.Runnable";
                invoke %3 @java.ref:"DenotableTypesTest::consume(java.lang.Runnable):void";
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

    @Reflect
    @IR("""
            func @"test8" (%0 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>")java.type:"void" -> {
                %1 : Var<java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>"> = var %0 @"list";
                %2 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                %5 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>" = var.load %1;
                %6 : java.type:"int" = constant @1;
                %7 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = invoke %5 %6 @java.ref:"java.util.List::get(int):java.lang.Object";
                invoke %4 %7 @java.ref:"DenotableTypesTest$Adder::add(DenotableTypesTest$Adder):void";
                return;
            };
            """)
    static void test8(List<? extends Adder<Integer>> list) {
        list.get(0).add(list.get(1));
    }

    static class Box<X> {
        X x;
    }

    @Reflect
    @IR("""
            func @"test9" (%0 : java.type:"java.util.List<? extends DenotableTypesTest$Box<java.lang.Integer>>")java.type:"void" -> {
                %1 : Var<java.type:"java.util.List<? extends DenotableTypesTest$Box<java.lang.Integer>>"> = var %0 @"list";
                %2 : java.type:"java.util.List<? extends DenotableTypesTest$Box<java.lang.Integer>>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"DenotableTypesTest$Box<java.lang.Integer>" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                %5 : java.type:"java.lang.Integer" = field.load %4 @java.ref:"DenotableTypesTest$Box::x:java.lang.Object";
                %6 : Var<java.type:"java.lang.Integer"> = var %5 @"i";
                return;
            };
            """)
    static void test9(List<? extends Box<Integer>> list) {
        Integer i = list.get(0).x;
    }

    interface E {
        void m();
    }

    static class XA extends Exception implements E {
        public void m() { }
    }

    static class XB extends Exception implements E {
        public void m() { }
    }

    static void g() throws XA, XB { }

    @Reflect
    @IR("""
            func @"test10" ()java.type:"void" -> {
                java.try
                    ()java.type:"void" -> {
                        invoke @java.ref:"DenotableTypesTest::g():void";
                        yield;
                    }
                    (%0 : java.type:"java.lang.Exception")java.type:"void" -> {
                        %1 : Var<java.type:"java.lang.Exception"> = var %0 @"x";
                        %2 : java.type:"java.lang.Exception" = var.load %1;
                        %3 : java.type:"DenotableTypesTest$E" = cast %2 @java.type:"DenotableTypesTest$E";
                        invoke %3 @java.ref:"DenotableTypesTest$E::m():void";
                        yield;
                    };
                return;
            };
            """)
    static void test10() {
        try {
            g();
        } catch (XA | XB x) {
            x.m();
        }
    }

    static <Z> List<Z> pickInv(Z z1, Z z2) { return null; }
    static <Z> List<? extends Z> pickExt(Z z1, Z z2) { return null; }
    static <Z> List<? super Z> pickSup(Z z1, Z z2) { return null; }

    // test intersections

    @Reflect
    @IR("""
            func @"test11" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickInv(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test11() {
        pickInv((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test12" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickExt(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test12() {
        pickExt((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test13" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<?>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickSup(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test13() {
        pickSup((C)null, (D)null);
    }

    static <Z> List<Z[]> pickInvArr(Z z1, Z z2) { return null; }
    static <Z> List<? extends Z[]> pickExtArr(Z z1, Z z2) { return null; }
    static <Z> List<? super Z[]> pickSupArr(Z z1, Z z2) { return null; }

    // test arrays of intersections

    @Reflect
    @IR("""
            func @"test14" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A[]>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickInvArr(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test14() {
        pickInvArr((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test15" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A[]>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickExtArr(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test15() {
        pickExtArr((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test16" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<?>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickSupArr(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test16() {
        pickSupArr((C)null, (D)null);
    }

    interface F<X> { }
    interface G<X> { }
    static class H<X> implements F<X>, G<X> { }
    static class I<X> implements F<X>, G<X> { }

    static <Z> H<Z> pickH(Z z1, Z z2) { return null; }
    static <Z> I<Z> pickI(Z z1, Z z2) { return null; }

    // test intersections of intersections

    @Reflect
    @IR("""
            func @"test17" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickH(java.lang.Object, java.lang.Object):DenotableTypesTest$H";
                %5 : Var<java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>"> = var %4 @"fst";
                %6 : java.type:"java.lang.Object" = constant @null;
                %7 : java.type:"DenotableTypesTest$C" = cast %6 @java.type:"DenotableTypesTest$C";
                %8 : java.type:"java.lang.Object" = constant @null;
                %9 : java.type:"DenotableTypesTest$D" = cast %8 @java.type:"DenotableTypesTest$D";
                %10 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = invoke %7 %9 @java.ref:"DenotableTypesTest::pickI(java.lang.Object, java.lang.Object):DenotableTypesTest$I";
                %11 : Var<java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>"> = var %10 @"snd";
                %12 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = var.load %5;
                %13 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = var.load %11;
                %14 : java.type:"java.util.List<? extends DenotableTypesTest$F<? extends DenotableTypesTest$A>>" = invoke %12 %13 @java.ref:"DenotableTypesTest::pickInv(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test17() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        pickInv(fst, snd);
    }

    @Reflect
    @IR("""
            func @"test18" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickH(java.lang.Object, java.lang.Object):DenotableTypesTest$H";
                %5 : Var<java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>"> = var %4 @"fst";
                %6 : java.type:"java.lang.Object" = constant @null;
                %7 : java.type:"DenotableTypesTest$C" = cast %6 @java.type:"DenotableTypesTest$C";
                %8 : java.type:"java.lang.Object" = constant @null;
                %9 : java.type:"DenotableTypesTest$D" = cast %8 @java.type:"DenotableTypesTest$D";
                %10 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = invoke %7 %9 @java.ref:"DenotableTypesTest::pickI(java.lang.Object, java.lang.Object):DenotableTypesTest$I";
                %11 : Var<java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>"> = var %10 @"snd";
                %12 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = var.load %5;
                %13 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = var.load %11;
                %14 : java.type:"java.util.List<? extends DenotableTypesTest$F<? extends DenotableTypesTest$A>>" = invoke %12 %13 @java.ref:"DenotableTypesTest::pickExt(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test18() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        pickExt(fst, snd);
    }

    @Reflect
    @IR("""
            func @"test19" ()java.type:"void" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickH(java.lang.Object, java.lang.Object):DenotableTypesTest$H";
                %5 : Var<java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>"> = var %4 @"fst";
                %6 : java.type:"java.lang.Object" = constant @null;
                %7 : java.type:"DenotableTypesTest$C" = cast %6 @java.type:"DenotableTypesTest$C";
                %8 : java.type:"java.lang.Object" = constant @null;
                %9 : java.type:"DenotableTypesTest$D" = cast %8 @java.type:"DenotableTypesTest$D";
                %10 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = invoke %7 %9 @java.ref:"DenotableTypesTest::pickI(java.lang.Object, java.lang.Object):DenotableTypesTest$I";
                %11 : Var<java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>"> = var %10 @"snd";
                %12 : java.type:"DenotableTypesTest$H<? extends DenotableTypesTest$A>" = var.load %5;
                %13 : java.type:"DenotableTypesTest$I<? extends DenotableTypesTest$A>" = var.load %11;
                %14 : java.type:"java.util.List<?>" = invoke %12 %13 @java.ref:"DenotableTypesTest::pickSup(java.lang.Object, java.lang.Object):java.util.List";
                return;
            };
            """)
    static void test19() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        pickSup(fst, snd);
    }
}
