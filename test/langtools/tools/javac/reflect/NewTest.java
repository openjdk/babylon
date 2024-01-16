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
import java.math.BigDecimal;
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with new expressions.
 * @build NewTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester NewTest
 */

public class NewTest {

    @CodeReflection
    @IR("""
            func @"test0" (%0 : NewTest)void -> {
                %1 : java.lang.String = constant @"1";
                %2 : java.math.BigDecimal = new %1 @"(java.lang.String)java.math.BigDecimal";
                %3 : Var<java.math.BigDecimal> = var %2 @"a";
                return;
            };
            """)
    void test0() {
        BigDecimal a = new BigDecimal("1");
    }

    static class A {
        A() {}

        A(int i, int j) {}
    }

    @CodeReflection
    @IR("""
            func @"test1" (%0 : NewTest)void -> {
                %1 : NewTest$A = new @"()NewTest$A";
                %2 : Var<NewTest$A> = var %1 @"a";
                return;
            };
            """)
    void test1() {
        A a = new A();
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : NewTest)void -> {
                %1 : int = constant @"1";
                %2 : int = constant @"2";
                %3 : NewTest$A = new %1 %2 @"(int, int)NewTest$A";
                %4 : Var<NewTest$A> = var %3 @"a";
                return;
            };
            """)
    void test2() {
        A a = new A(1, 2);
    }

    class B {
        B() {}

        B(int i, int j) {}

        class C {
        }
    }

    B f;

    B b() { return f; }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : NewTest)void -> {
                %1 : NewTest$B = new %0 @"()NewTest$B";
                %2 : Var<NewTest$B> = var %1 @"b";
                return;
            };
            """)
    void test3() {
        B b = new B();
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : NewTest)void -> {
                %1 : int = constant @"1";
                %2 : int = constant @"2";
                %3 : NewTest$B = new %0 %1 %2 @"(int, int)NewTest$B";
                %4 : Var<NewTest$B> = var %3 @"b";
                return;
            };
            """)
    void test4() {
        B b = new B(1, 2);
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : NewTest)void -> {
                %1 : NewTest$B = new %0 @"()NewTest$B";
                %2 : Var<NewTest$B> = var %1 @"b";
                return;
            };
            """)
    void test5() {
        B b = this.new B();
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : NewTest)void -> {
                %1 : NewTest$B = field.load %0 @"NewTest::f()NewTest$B";
                %2 : NewTest$B$C = new %1 @"()NewTest$B$C";
                %3 : Var<NewTest$B$C> = var %2 @"c";
                return;
            };
            """)
    void test6() {
        B.C c = f.new C();
    }

    @CodeReflection
    @IR("""
            func @"test7" (%0 : NewTest)void -> {
                %1 : NewTest$B = invoke %0 @"NewTest::b()NewTest$B";
                %2 : NewTest$B$C = new %1 @"()NewTest$B$C";
                %3 : Var<NewTest$B$C> = var %2 @"c";
                return;
            };
            """)
    void test7() {
        B.C c = b().new C();
    }

    static class AG<T> {
        AG(List<T> l) {}
    }

    @CodeReflection
    @IR("""
            func @"test8" (%0 : NewTest, %1 : java.util.List<java.lang.String>)void -> {
                %2 : Var<java.util.List<java.lang.String>> = var %1 @"l";
                %3 : java.util.List<java.lang.String> = var.load %2;
                %4 : NewTest$AG<java.lang.String> = new %3 @"(java.util.List)NewTest$AG";
                %5 : Var<NewTest$AG<java.lang.String>> = var %4 @"a";
                return;
            };
            """)
    void test8(List<String> l) {
        AG<String> a = new AG<>(l);
    }

    class BG<T> {
        BG(List<T> l) {}

        class CG<U> {
            CG(List<U> l) {}
        }
    }

    // @@@ This produces incorrect type descriptors for generic inner classes
    // the type argument for type BG is not preserved
//    @CodeReflection
    @IR("""
            func @"test9" (%0 : NewTest, %1 : java.util.List<java.lang.String>, %2 : java.util.List<java.lang.Number>)void -> {
                %3 : Var<java.util.List<java.lang.String>> = var %1 @"l1";
                %4 : Var<java.util.List<java.lang.Number>> = var %2 @"l2";
                %5 : java.util.List<java.lang.String> = var.load %3;
                %6 : NewTest$BG<java.lang.String> = new %0 %5 @"(java.util.List)NewTest$BG";
                %7 : java.util.List<java.lang.Number> = var.load %4;
                %8 : NewTest$BG$CG<java.lang.Number> = new %6 %7 @"(java.util.List)NewTest$BG$CG";
                %9 : Var<NewTest$BG$CG<java.lang.Number>> = var %8 @"numberCG";
                return;
            };
            """)
    void test9(List<String> l1, List<Number> l2) {
        BG<String>.CG<Number> numberCG = new BG<String>(l1).new CG<Number>(l2);
    }


    @CodeReflection
    @IR("""
            func @"test10" (%0 : NewTest)void -> {
                %1 : int = constant @"10";
                %2 : int[] = new %1 @"(int)int[]";
                %3 : Var<int[]> = var %2 @"i";
                return;
            };
            """)
    void test10() {
        int[] i = new int[10];
    }

    @CodeReflection
    @IR("""
            func @"test11" (%0 : NewTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : int = var.load %2;
                %5 : int = constant @"1";
                %6 : int = add %4 %5;
                %7 : int = var.load %2;
                %8 : int = constant @"2";
                %9 : int = add %7 %8;
                %10 : java.lang.String[][][] = new %3 %6 %9 @"(int, int, int)java.lang.String[][][]";
                %11 : Var<java.lang.String[][][]> = var %10 @"s";
                return;
            };
            """)
    void test11(int i) {
        String[][][] s = new String[i][i + 1][i + 2];
    }
}
