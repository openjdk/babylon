/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
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
            func @"test1" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Number" = constant @null;
                %1 : java.type:"java.lang.Number" = invoke %0 @java.ref:"DenotableTypesTest::m1(java.lang.Number):java.lang.Number";
                return %1;
            };
            """)
    static Object test1() {
        return m1(null);
    }

    @Reflect
    @IR("""
            func @"test2" (%0 : java.type:"DenotableTypesTest")java.type:"java.lang.Object" -> {
                %1 : java.type:"int" = constant @1;
                %2 : java.type:"java.lang.Integer" = invoke %1 @java.ref:"java.lang.Integer::valueOf(int):java.lang.Integer";
                %3 : java.type:"double" = constant @3.0d;
                %4 : java.type:"java.lang.Double" = invoke %3 @java.ref:"java.lang.Double::valueOf(double):java.lang.Double";
                %5 : java.type:"java.util.List<? extends java.lang.Number>" = invoke %2 %4 @java.ref:"java.util.List::of(java.lang.Object, java.lang.Object):java.util.List";
                return %5;
            };
            """)
    Object test2() {
        return List.of(1, 3d); // infinite type! (List<Object & Serializable & Comparable<...>>)
    }

    static <X extends Throwable> X m2(X x) throws X { return null; }

    @Reflect
    @IR("""
            func @"test3" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.RuntimeException" = constant @null;
                %1 : java.type:"java.lang.RuntimeException" = invoke %0 @java.ref:"DenotableTypesTest::m2(java.lang.Throwable):java.lang.Throwable";
                return %1;
            };
            """)
    static Object test3() { // @@@ cast?
        return m2(null);
    }

    interface A { }
    interface B { }
    static class C implements A, B { }
    static class D implements A, B { }

    static <Z> Z pick(Z z1, Z z2) { return null; }

    @Reflect
    @IR("""
            func @"test4" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"DenotableTypesTest$A" = invoke %1 %3 @java.ref:"DenotableTypesTest::pick(java.lang.Object, java.lang.Object):java.lang.Object";
                return %4;
            };
            """)
    static Object test4() { // @@@ cast?
        return pick((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test5" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.util.List<? extends java.lang.Number>" = constant @null;
                %1 : Var<java.type:"java.util.List<? extends java.lang.Number>"> = var %0 @"l";
                %2 : java.type:"java.util.List<? extends java.lang.Number>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"java.lang.Number" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                return %4;
            };
            """)
    static Object test5() { // @@@ cast?
        List<? extends Number> l = null;
        return l.get(0);
    }

    @Reflect
    @IR("""
            func @"test6" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.util.List<? super java.lang.Number>" = constant @null;
                %1 : Var<java.type:"java.util.List<? super java.lang.Number>"> = var %0 @"l";
                %2 : java.type:"java.util.List<? super java.lang.Number>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"java.lang.Object" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                return %4;
            };
            """)
    static Object test6() {
        List<? super Number> l = null;
        return l.get(0);
    }

    static Object consume(Runnable r) { return null; }

    @Reflect
    @IR("""
            func @"test7" ()java.type:"void" -> {
                %0 : java.type:"&DenotableTypesTest::test7():void::<X>" = constant @null;
                %1 : Var<java.type:"&DenotableTypesTest::test7():void::<X>"> = var %0 @"x";
                %2 : java.type:"&DenotableTypesTest::test7():void::<X>" = var.load %1;
                %3 : java.type:"java.lang.Runnable" = cast %2 @java.type:"java.lang.Runnable";
                %4 : java.type:"java.lang.Object" = invoke %3 @java.ref:"DenotableTypesTest::consume(java.lang.Runnable):java.lang.Object";
                return;
            };
            """)
    static <X extends Object & Runnable> void test7() {
        // @@@: FIXME, we can't update this because we have a JavaType grammar ambiguity
        X x = null;
        consume(x);
    }

    interface Adder<X> {
        Object add(Adder<X> adder);
    }

    @Reflect
    @IR("""
            func @"test8" (%0 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>"> = var %0 @"list";
                %2 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = invoke %2 %3 @java.ref:"java.util.List::get(int):java.lang.Object";
                %5 : Var<java.type:"DenotableTypesTest$Adder<java.lang.Integer>"> = var %4 @"x";
                %6 : java.type:"java.util.List<? extends DenotableTypesTest$Adder<java.lang.Integer>>" = var.load %1;
                %7 : java.type:"int" = constant @1;
                %8 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = invoke %6 %7 @java.ref:"java.util.List::get(int):java.lang.Object";
                %9 : Var<java.type:"DenotableTypesTest$Adder<java.lang.Integer>"> = var %8 @"y";
                %10 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = var.load %5;
                %11 : java.type:"DenotableTypesTest$Adder<java.lang.Integer>" = var.load %9;
                %12 : java.type:"java.lang.Object" = invoke %10 %11 @java.ref:"DenotableTypesTest$Adder::add(DenotableTypesTest$Adder):java.lang.Object";
                return %12;
            };
            """)
    static Object test8(List<? extends Adder<Integer>> list) {
        var x = list.get(0);
        var y = list.get(1);
        return x.add(y);
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
        Object m();
    }

    static class XA extends Exception implements E {
        public Object m() { return null; }
    }

    static class XB extends Exception implements E {
        public Object m() { return null; }
    }

    static Object g() throws XA, XB { return null; }

    @Reflect
    @IR("""
            func @"test10" ()java.type:"java.lang.Object" -> {
                java.try @Tuple<Tuple<java.type:"DenotableTypesTest$XA", java.type:"DenotableTypesTest$XB">>
                    ()java.type:"void" -> {
                        %0 : java.type:"java.lang.Object" = invoke @java.ref:"DenotableTypesTest::g():java.lang.Object";
                        return %0;
                    }
                    (%1 : java.type:"java.lang.Exception")java.type:"void" -> {
                        %2 : Var<java.type:"java.lang.Exception"> = var %1 @"x";
                        %3 : java.type:"java.lang.Exception" = var.load %2;
                        %4 : java.type:"DenotableTypesTest$E" = cast %3 @java.type:"DenotableTypesTest$E";
                        %5 : java.type:"java.lang.Object" = invoke %4 @java.ref:"DenotableTypesTest$E::m():java.lang.Object";
                        return %5;
                    };
                unreachable;
            };
            """)
    static Object test10() {
        try {
            return g();
        } catch (XA | XB x) {
            return x.m();
        }
    }

    static <Z> List<Z> pickInv(Z z1, Z z2) { return null; }
    static <Z> List<? extends Z> pickExt(Z z1, Z z2) { return null; }
    static <Z> List<? super Z> pickSup(Z z1, Z z2) { return null; }

    // test intersections

    @Reflect
    @IR("""
            func @"test11" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickInv(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test11() {
        return pickInv((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test12" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickExt(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test12() {
        return pickExt((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test13" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<?>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickSup(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test13() {
        return pickSup((C)null, (D)null);
    }

    static <Z> List<Z[]> pickInvArr(Z z1, Z z2) { return null; }
    static <Z> List<? extends Z[]> pickExtArr(Z z1, Z z2) { return null; }
    static <Z> List<? super Z[]> pickSupArr(Z z1, Z z2) { return null; }

    // test arrays of intersections

    @Reflect
    @IR("""
            func @"test14" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A[]>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickInvArr(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test14() {
        return pickInvArr((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test15" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<? extends DenotableTypesTest$A[]>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickExtArr(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test15() {
        return pickExtArr((C)null, (D)null);
    }

    @Reflect
    @IR("""
            func @"test16" ()java.type:"java.lang.Object" -> {
                %0 : java.type:"java.lang.Object" = constant @null;
                %1 : java.type:"DenotableTypesTest$C" = cast %0 @java.type:"DenotableTypesTest$C";
                %2 : java.type:"java.lang.Object" = constant @null;
                %3 : java.type:"DenotableTypesTest$D" = cast %2 @java.type:"DenotableTypesTest$D";
                %4 : java.type:"java.util.List<?>" = invoke %1 %3 @java.ref:"DenotableTypesTest::pickSupArr(java.lang.Object, java.lang.Object):java.util.List";
                return %4;
            };
            """)
    static Object test16() {
        return pickSupArr((C)null, (D)null);
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
            func @"test17" ()java.type:"java.lang.Object" -> {
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
                return %14;
            };
            """)
    static Object test17() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        return pickInv(fst, snd);
    }

    @Reflect
    @IR("""
            func @"test18" ()java.type:"java.lang.Object" -> {
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
                return %14;
            };
            """)
    static Object test18() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        return pickExt(fst, snd);
    }

    @Reflect
    @IR("""
            func @"test19" ()java.type:"java.lang.Object" -> {
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
                return %14;
            };
            """)
    static Object test19() {
        var fst = pickH((C)null, (D)null);
        var snd = pickI((C)null, (D)null);
        return pickSup(fst, snd);
    }

    @Reflect
    @IR("""
            func @"test20" (%0 : java.type:"DenotableTypesTest$Box<? extends java.lang.Number>")java.type:"java.lang.Number" -> {
                %1 : Var<java.type:"DenotableTypesTest$Box<? extends java.lang.Number>"> = var %0 @"box";
                %2 : java.type:"DenotableTypesTest$Box<? extends java.lang.Number>" = var.load %1;
                %3 : java.type:"java.lang.Number" = field.load %2 @java.ref:"DenotableTypesTest$Box::x:java.lang.Object";
                return %3;
            };
            """)
    static Number test20(Box<? extends Number> box) {
        return box.x;
    }

    @Reflect
    @IR("""
            func @"test21" (%0 : java.type:"DenotableTypesTest$Box<? super java.lang.Number>")java.type:"java.lang.Object" -> {
                %1 : Var<java.type:"DenotableTypesTest$Box<? super java.lang.Number>"> = var %0 @"box";
                %2 : java.type:"DenotableTypesTest$Box<? super java.lang.Number>" = var.load %1;
                %3 : java.type:"java.lang.Object" = field.load %2 @java.ref:"DenotableTypesTest$Box::x:java.lang.Object";
                return %3;
            };
            """)
    static Object test21(Box<? super Number> box) {
        return box.x;
    }
}
