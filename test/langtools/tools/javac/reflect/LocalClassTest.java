/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

/*
 * @test
 * @summary Smoke test for code reflection with local class creation expressions.
 * @build LocalClassTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester LocalClassTest
 */

import java.lang.runtime.CodeReflection;

public class LocalClassTest {

    final static String CONST_STRING = "Hello!";
    String nonConstString = "Hello!";

    @CodeReflection
    @IR("""
            func @"testLocalNoCapture" (%0 : LocalClassTest)void -> {
                %1 : .<LocalClassTest, LocalClassTest$1Foo> = new %0 @"func<.<LocalClassTest, LocalClassTest$1Foo>, LocalClassTest>";
                invoke %1 @".<LocalClassTest, LocalClassTest$1Foo>::m()void";
                return;
            };
            """)
    void testLocalNoCapture() {
        class Foo {
            void m() { }
        }
        new Foo().m();
    }

    @CodeReflection
    @IR("""
            func @"testAnonNoCapture" (%0 : LocalClassTest)void -> {
                %1 : .<LocalClassTest, LocalClassTest$1> = new %0 @"func<.<LocalClassTest, LocalClassTest$1>, LocalClassTest>";
                invoke %1 @".<LocalClassTest, LocalClassTest$1>::m()void";
                return;
            };
            """)
    void testAnonNoCapture() {
        new Object() {
            void m() { }
        }.m();
    }

    @CodeReflection
    @IR("""
            func @"testLocalCaptureParam" (%0 : LocalClassTest, %1 : java.lang.String)java.lang.String -> {
                %2 : Var<java.lang.String> = var %1 @"s";
                %3 : java.lang.String = var.load %2;
                %4 : .<LocalClassTest, LocalClassTest$2Foo> = new %0 %3 @"func<.<LocalClassTest, LocalClassTest$2Foo>, LocalClassTest, java.lang.String>";
                %5 : java.lang.String = invoke %4 @".<LocalClassTest, LocalClassTest$2Foo>::m()java.lang.String";
                return %5;
            };
            """)
    String testLocalCaptureParam(String s) {
        class Foo {
            String m() { return s; }
        }
        return new Foo().m();
    }

    @CodeReflection
    @IR("""
            func @"testAnonCaptureParam" (%0 : LocalClassTest, %1 : java.lang.String)java.lang.String -> {
                %2 : Var<java.lang.String> = var %1 @"s";
                %3 : java.lang.String = var.load %2;
                %4 : .<LocalClassTest, LocalClassTest$2> = new %0 %3 @"func<.<LocalClassTest, LocalClassTest$2>, LocalClassTest, java.lang.String>";
                %5 : java.lang.String = invoke %4 @".<LocalClassTest, LocalClassTest$2>::m()java.lang.String";
                return %5;
            };
            """)
    String testAnonCaptureParam(String s) {
        return new Object() {
            String m() { return s; }
        }.m();
    }

    @CodeReflection
    @IR("""
            func @"testLocalCaptureParamAndField" (%0 : LocalClassTest, %1 : java.lang.String)java.lang.String -> {
                %2 : Var<java.lang.String> = var %1 @"s";
                %3 : java.lang.String = constant @"Hello!";
                %4 : Var<java.lang.String> = var %3 @"localConst";
                %5 : java.lang.String = var.load %2;
                %6 : .<LocalClassTest, LocalClassTest$3Foo> = new %0 %5 @"func<.<LocalClassTest, LocalClassTest$3Foo>, LocalClassTest, java.lang.String>";
                %7 : java.lang.String = invoke %6 @".<LocalClassTest, LocalClassTest$3Foo>::m()java.lang.String";
                return %7;
            };
            """)
    String testLocalCaptureParamAndField(String s) {
        final String localConst = "Hello!";
        class Foo {
            String m() { return localConst + s + nonConstString + CONST_STRING; }
        }
        return new Foo().m();
    }

    @CodeReflection
    @IR("""
            func @"testAnonCaptureParamAndField" (%0 : LocalClassTest, %1 : java.lang.String)java.lang.String -> {
                %2 : Var<java.lang.String> = var %1 @"s";
                %3 : java.lang.String = constant @"Hello!";
                %4 : Var<java.lang.String> = var %3 @"localConst";
                %5 : java.lang.String = var.load %2;
                %6 : .<LocalClassTest, LocalClassTest$3> = new %0 %5 @"func<.<LocalClassTest, LocalClassTest$3>, LocalClassTest, java.lang.String>";
                %7 : java.lang.String = invoke %6 @".<LocalClassTest, LocalClassTest$3>::m()java.lang.String";
                return %7;
            };
            """)
    String testAnonCaptureParamAndField(String s) {
        final String localConst = "Hello!";
        return new Object() {
            String m() { return localConst + s + nonConstString + CONST_STRING; }
        }.m();
    }

    @CodeReflection
    @IR("""
            func @"testLocalDependency" (%0 : LocalClassTest, %1 : int, %2 : int)void -> {
                %3 : Var<int> = var %1 @"s";
                %4 : Var<int> = var %2 @"i";
                %5 : int = var.load %3;
                %6 : int = var.load %4;
                %7 : .<LocalClassTest, LocalClassTest$1Bar> = new %0 %5 %6 @"func<.<LocalClassTest, LocalClassTest$1Bar>, LocalClassTest, int, int>";
                return;
            };
            """)
    void testLocalDependency(int s, int i) {
        class Foo {
            int i() { return i; }
        }
        class Bar {
            int s() { return s; }
            Foo foo() { return new Foo(); }
        }
        new Bar();
    }

    @CodeReflection
    @IR("""
            func @"testAnonDependency" (%0 : LocalClassTest, %1 : int, %2 : int)void -> {
                %3 : Var<int> = var %1 @"s";
                %4 : Var<int> = var %2 @"i";
                %5 : int = var.load %3;
                %6 : int = var.load %4;
                %7 : .<LocalClassTest, LocalClassTest$4> = new %0 %5 %6 @"func<.<LocalClassTest, LocalClassTest$4>, LocalClassTest, int, int>";
                return;
            };
            """)
    void testAnonDependency(int s, int i) {
        class Foo {
            int i() { return i; }
        }
        new Object() {
            int s() { return s; }
            Foo foo() { return new Foo(); }
        };
    }
}
