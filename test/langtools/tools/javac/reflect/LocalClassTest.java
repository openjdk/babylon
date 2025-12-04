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
 * @modules jdk.incubator.code
 * @build LocalClassTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester LocalClassTest
 */

import jdk.incubator.code.Reflect;

import java.util.function.Supplier;

public class LocalClassTest {

    final static String CONST_STRING = "Hello!";
    String nonConstString = "Hello!";

    @Reflect
    @IR("""
            func @"testLocalNoCapture" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"LocalClassTest::$1Foo" = new %0 @java.ref:"LocalClassTest::$1Foo::(LocalClassTest)";
                invoke %1 @java.ref:"LocalClassTest::$1Foo::m():void";
                return;
            };
            """)
    void testLocalNoCapture() {
        class Foo {
            void m() { }
        }
        new Foo().m();
    }

    @Reflect
    @IR("""
            func @"testAnonNoCapture" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"LocalClassTest::$1" = new %0 @java.ref:"LocalClassTest::$1::(LocalClassTest)";
                invoke %1 @java.ref:"LocalClassTest::$1::m():void";
                return;
            };
            """)
    void testAnonNoCapture() {
        new Object() {
            void m() { }
        }.m();
    }

    @Reflect
    @IR("""
            func @"testLocalCaptureParam" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.lang.String" = var.load %2;
                %4 : java.type:"LocalClassTest::$2Foo" = new %0 %3 @java.ref:"LocalClassTest::$2Foo::(LocalClassTest, java.lang.String)";
                %5 : java.type:"java.lang.String" = invoke %4 @java.ref:"LocalClassTest::$2Foo::m():java.lang.String";
                return %5;
            };
            """)
    String testLocalCaptureParam(String s) {
        class Foo {
            String m() { return s; }
        }
        return new Foo().m();
    }

    @Reflect
    @IR("""
            func @"testAnonCaptureParam" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.lang.String" = var.load %2;
                %4 : java.type:"LocalClassTest::$2" = new %0 %3 @java.ref:"LocalClassTest::$2::(LocalClassTest, java.lang.String)";
                %5 : java.type:"java.lang.String" = invoke %4 @java.ref:"LocalClassTest::$2::m():java.lang.String";
                return %5;
            };
            """)
    String testAnonCaptureParam(String s) {
        return new Object() {
            String m() { return s; }
        }.m();
    }

    @Reflect
    @IR("""
            func @"testLocalCaptureParamAndField" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.lang.String" = constant @"Hello!";
                %4 : Var<java.type:"java.lang.String"> = var %3 @"localConst";
                %5 : java.type:"java.lang.String" = var.load %2;
                %6 : java.type:"LocalClassTest::$3Foo" = new %0 %5 @java.ref:"LocalClassTest::$3Foo::(LocalClassTest, java.lang.String)";
                %7 : java.type:"java.lang.String" = invoke %6 @java.ref:"LocalClassTest::$3Foo::m():java.lang.String";
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

    @Reflect
    @IR("""
            func @"testAnonCaptureParamAndField" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.lang.String" = constant @"Hello!";
                %4 : Var<java.type:"java.lang.String"> = var %3 @"localConst";
                %5 : java.type:"java.lang.String" = var.load %2;
                %6 : java.type:"LocalClassTest::$3" = new %0 %5 @java.ref:"LocalClassTest::$3::(LocalClassTest, java.lang.String)";
                %7 : java.type:"java.lang.String" = invoke %6 @java.ref:"LocalClassTest::$3::m():java.lang.String";
                return %7;
            };
            """)
    String testAnonCaptureParamAndField(String s) {
        final String localConst = "Hello!";
        return new Object() {
            String m() { return localConst + s + nonConstString + CONST_STRING; }
        }.m();
    }

    @Reflect
    @IR("""
            func @"testLocalDependency" (%0 : java.type:"LocalClassTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"s";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"LocalClassTest::$1Bar" = new %0 %5 %6 @java.ref:"LocalClassTest::$1Bar::(LocalClassTest, int, int)";
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

    @Reflect
    @IR("""
            func @"testAnonDependency" (%0 : java.type:"LocalClassTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"void" -> {
                %3 : Var<java.type:"int"> = var %1 @"s";
                %4 : Var<java.type:"int"> = var %2 @"i";
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"int" = var.load %4;
                %7 : java.type:"LocalClassTest::$4" = new %0 %5 %6 @java.ref:"LocalClassTest::$4::(LocalClassTest, int, int)";
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

    class Inner { }

    @Reflect
    @IR("""
            func @"testImplicitInner" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<LocalClassTest::Inner>" = lambda @lambda.isQuotable=true ()java.type:"LocalClassTest::Inner" -> {
                    %2 : java.type:"LocalClassTest::Inner" = new %0 @java.ref:"LocalClassTest::Inner::(LocalClassTest)";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<LocalClassTest::Inner>"> = var %1 @"aNew";
                return;
            };
            """)
    void testImplicitInner() {
        Supplier<Inner> aNew = (@Reflect Supplier<Inner>) () -> new Inner();
    }

    @Reflect
    @IR("""
            func @"testExplicitInner" (%0 : java.type:"LocalClassTest", %1 : java.type:"LocalClassTest")java.type:"void" -> {
                %2 : Var<java.type:"LocalClassTest"> = var %1 @"test";
                %3 : java.type:"java.util.function.Supplier<LocalClassTest::Inner>" = lambda @lambda.isQuotable=true ()java.type:"LocalClassTest::Inner" -> {
                    %4 : java.type:"LocalClassTest" = var.load %2;
                    %5 : java.type:"LocalClassTest::Inner" = new %4 @java.ref:"LocalClassTest::Inner::(LocalClassTest)";
                    return %5;
                };
                %6 : Var<java.type:"java.util.function.Supplier<LocalClassTest::Inner>"> = var %3 @"aNew";
                return;
            };
            """)
    void testExplicitInner(LocalClassTest test) {
        Supplier<Inner> aNew = (@Reflect Supplier<Inner>) () -> test.new Inner();
    }

    @Reflect
    @IR("""
            func @"testLocalInMethod" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<LocalClassTest::$1L>" = lambda @lambda.isQuotable=true ()java.type:"LocalClassTest::$1L" -> {
                    %2 : java.type:"LocalClassTest::$1L" = new %0 @java.ref:"LocalClassTest::$1L::(LocalClassTest)";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<LocalClassTest::$1L>"> = var %1 @"aNew";
                return;
            };
            """)
    void testLocalInMethod() {
        class L { }
        Supplier<L> aNew = (@Reflect Supplier<L>) () -> new L();
    }

    @Reflect
    @IR("""
            func @"testLocalInLambda" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"java.util.function.Supplier<java.lang.Object>" = lambda @lambda.isQuotable=true ()java.type:"java.lang.Object" -> {
                    %2 : java.type:"LocalClassTest::$2L" = new %0 @java.ref:"LocalClassTest::$2L::(LocalClassTest)";
                    return %2;
                };
                %3 : Var<java.type:"java.util.function.Supplier<java.lang.Object>"> = var %1 @"aNew";
                return;
            };
            """)
    void testLocalInLambda() {
        Supplier<Object> aNew = (@Reflect Supplier<Object>) () -> {
            class L { }
            return new L();
        };
    }

    @Reflect
    @IR("""
            func @"testLocalInMethodWithCaptures" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @"Foo";
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.util.function.Supplier<LocalClassTest::$3L>" = lambda @lambda.isQuotable=true ()java.type:"LocalClassTest::$3L" -> {
                    %4 : java.type:"java.lang.String" = var.load %2;
                    %5 : java.type:"LocalClassTest::$3L" = new %0 %4 @java.ref:"LocalClassTest::$3L::(LocalClassTest, java.lang.String)";
                    return %5;
                };
                %6 : Var<java.type:"java.util.function.Supplier<LocalClassTest::$3L>"> = var %3 @"aNew";
                return;
            };
            """)
    void testLocalInMethodWithCaptures() {
        String s = "Foo";
        class L {
            String s() {
                return s;
            }
        }
        Supplier<L> aNew = (@Reflect Supplier<L>) () -> new L();
    }

    @Reflect
    @IR("""
            func @"testLocalInLambdaWithCaptures" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @"Foo";
                %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                %3 : java.type:"java.util.function.Supplier<java.lang.Object>" = lambda @lambda.isQuotable=true ()java.type:"java.lang.Object" -> {
                    %4 : java.type:"java.lang.String" = var.load %2;
                    %5 : java.type:"LocalClassTest::$4L" = new %0 %4 @java.ref:"LocalClassTest::$4L::(LocalClassTest, java.lang.String)";
                    return %5;
                };
                %6 : Var<java.type:"java.util.function.Supplier<java.lang.Object>"> = var %3 @"aNew";
                return;
            };
            """)
    void testLocalInLambdaWithCaptures() {
        String s = "Foo";
        Supplier<Object> aNew = (@Reflect Supplier<Object>) () -> {
            class L {
                String s() {
                    return s;
                }
            }
            return new L();
        };
    }
}
