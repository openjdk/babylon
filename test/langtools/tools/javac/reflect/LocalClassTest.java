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

import jdk.incubator.code.CodeReflection;

public class LocalClassTest {

    final static String CONST_STRING = "Hello!";
    String nonConstString = "Hello!";

    @CodeReflection
    @IR("""
            func @"testLocalNoCapture" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                  class.dec ()java.type:"void" -> {
                      %1 : Var<java.type:"LocalClassTest"> = var;
                      func @"<init>" (%3 : java.type:"LocalClassTest")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %1 %3;
                          return;
                      };
                      func @"m" (%2 : java.type:"LocalClassTest::$1Foo")java.type:"void" -> {
                          return;
                      };
                      yield;
                  };
                  %4 : java.type:"LocalClassTest::$1Foo" = new %0 @java.ref:"LocalClassTest::$1Foo::(LocalClassTest)";
                  invoke %4 @java.ref:"LocalClassTest::$1Foo::m():void";
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
            func @"testAnonNoCapture" (%0 : java.type:"LocalClassTest")java.type:"void" -> {
                  class.dec ()java.type:"void" -> {
                      %1 : Var<java.type:"LocalClassTest"> = var;
                      func @"<init>" (%2 : java.type:"LocalClassTest")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %1 %2;
                          return;
                      };
                      func @"m" (%3 : java.type:"LocalClassTest::$1")java.type:"void" -> {
                          return;
                      };
                      yield;
                  };
                  %4 : java.type:"LocalClassTest::$1" = new %0 @java.ref:"LocalClassTest::$1::(LocalClassTest)";
                  invoke %4 @java.ref:"LocalClassTest::$1::m():void";
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
            func @"testLocalCaptureParam" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                  class.dec ()java.type:"void" -> {
                      %3 : Var<java.type:"LocalClassTest"> = var;
                      %4 : Var<java.type:"java.lang.String"> = var;
                      func @"<init>" (%5 : java.type:"LocalClassTest", %6 : java.type:"java.lang.String")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %3 %5;
                          var.store %4 %6;
                          return;
                      };
                      func @"m" (%7 : java.type:"LocalClassTest::$2Foo")java.type:"java.lang.String" -> {
                          %8 : java.type:"java.lang.String" = var.load %4;
                          return %8;
                      };
                      yield;
                  };
                  %9 : java.type:"java.lang.String" = var.load %2;
                  %10 : java.type:"LocalClassTest::$2Foo" = new %0 %9 @java.ref:"LocalClassTest::$2Foo::(LocalClassTest, java.lang.String)";
                  %11 : java.type:"java.lang.String" = invoke %10 @java.ref:"LocalClassTest::$2Foo::m():java.lang.String";
                  return %11;
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
            func @"testAnonCaptureParam" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                  class.dec ()java.type:"void" -> {
                      %3 : Var<java.type:"LocalClassTest"> = var;
                      %4 : Var<java.type:"java.lang.String"> = var;
                      func @"<init>" (%5 : java.type:"LocalClassTest", %6 : java.type:"java.lang.String")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %3 %5;
                          var.store %4 %6;
                          return;
                      };
                      func @"m" (%7 : java.type:"LocalClassTest::$2")java.type:"java.lang.String" -> {
                          %8 : java.type:"java.lang.String" = var.load %4;
                          return %8;
                      };
                      yield;
                  };
                  %9 : java.type:"java.lang.String" = var.load %2;
                  %10 : java.type:"LocalClassTest::$2" = new %0 %9 @java.ref:"LocalClassTest::$2::(LocalClassTest, java.lang.String)";
                  %11 : java.type:"java.lang.String" = invoke %10 @java.ref:"LocalClassTest::$2::m():java.lang.String";
                  return %11;
              };
            """)
    String testAnonCaptureParam(String s) {
        return new Object() {
            String m() { return s; }
        }.m();
    }

    @CodeReflection
    @IR("""
            func @"testLocalCaptureParamAndField" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                  %3 : java.type:"java.lang.String" = constant @"Hello!";
                  %4 : Var<java.type:"java.lang.String"> = var %3 @"localConst";
                  class.dec ()java.type:"void" -> {
                      %5 : Var<java.type:"LocalClassTest"> = var;
                      %6 : Var<java.type:"java.lang.String"> = var;
                      func @"<init>" (%7 : java.type:"LocalClassTest", %8 : java.type:"java.lang.String")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %5 %7;
                          var.store %6 %8;
                          return;
                      };
                      func @"m" (%9 : java.type:"LocalClassTest::$3Foo")java.type:"java.lang.String" -> {
                          %10 : java.type:"java.lang.String" = var.load %4;
                          %11 : java.type:"java.lang.String" = var.load %6;
                          %12 : java.type:"java.lang.String" = concat %10 %11;
                          %13 : java.type:"LocalClassTest" = var.load %5;
                          %14 : java.type:"java.lang.String" = field.load %13 @java.ref:"LocalClassTest::nonConstString:java.lang.String";
                          %15 : java.type:"java.lang.String" = concat %12 %14;
                          %16 : java.type:"java.lang.String" = field.load @java.ref:"LocalClassTest::CONST_STRING:java.lang.String";
                          %17 : java.type:"java.lang.String" = concat %15 %16;
                          return %17;
                      };
                      yield;
                  };
                  %18 : java.type:"java.lang.String" = var.load %2;
                  %19 : java.type:"LocalClassTest::$3Foo" = new %0 %18 @java.ref:"LocalClassTest::$3Foo::(LocalClassTest, java.lang.String)";
                  %20 : java.type:"java.lang.String" = invoke %19 @java.ref:"LocalClassTest::$3Foo::m():java.lang.String";
                  return %20;
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
            func @"testAnonCaptureParamAndField" (%0 : java.type:"LocalClassTest", %1 : java.type:"java.lang.String")java.type:"java.lang.String" -> {
                  %2 : Var<java.type:"java.lang.String"> = var %1 @"s";
                  %3 : java.type:"java.lang.String" = constant @"Hello!";
                  %4 : Var<java.type:"java.lang.String"> = var %3 @"localConst";
                  class.dec ()java.type:"void" -> {
                      %5 : Var<java.type:"LocalClassTest"> = var;
                      %6 : Var<java.type:"java.lang.String"> = var;
                      func @"<init>" (%7 : java.type:"LocalClassTest", %8 : java.type:"java.lang.String")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %5 %7;
                          var.store %6 %8;
                          return;
                      };
                      func @"m" (%9 : java.type:"LocalClassTest::$3")java.type:"java.lang.String" -> {
                          %10 : java.type:"java.lang.String" = var.load %4;
                          %11 : java.type:"java.lang.String" = var.load %6;
                          %12 : java.type:"java.lang.String" = concat %10 %11;
                          %13 : java.type:"LocalClassTest" = var.load %5;
                          %14 : java.type:"java.lang.String" = field.load %13 @java.ref:"LocalClassTest::nonConstString:java.lang.String";
                          %15 : java.type:"java.lang.String" = concat %12 %14;
                          %16 : java.type:"java.lang.String" = field.load @java.ref:"LocalClassTest::CONST_STRING:java.lang.String";
                          %17 : java.type:"java.lang.String" = concat %15 %16;
                          return %17;
                      };
                      yield;
                  };
                  %18 : java.type:"java.lang.String" = var.load %2;
                  %19 : java.type:"LocalClassTest::$3" = new %0 %18 @java.ref:"LocalClassTest::$3::(LocalClassTest, java.lang.String)";
                  %20 : java.type:"java.lang.String" = invoke %19 @java.ref:"LocalClassTest::$3::m():java.lang.String";
                  return %20;
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
            func @"testLocalDependency" (%0 : java.type:"LocalClassTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"void" -> {
                  %3 : Var<java.type:"int"> = var %1 @"s";
                  %4 : Var<java.type:"int"> = var %2 @"i";
                  class.dec ()java.type:"void" -> {
                      %5 : Var<java.type:"LocalClassTest"> = var;
                      %6 : Var<java.type:"int"> = var;
                      func @"<init>" (%7 : java.type:"LocalClassTest", %8 : java.type:"int")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %5 %7;
                          var.store %6 %8;
                          return;
                      };
                      func @"i" (%9 : java.type:"LocalClassTest::$4Foo")java.type:"int" -> {
                          %10 : java.type:"int" = var.load %6;
                          return %10;
                      };
                      yield;
                  };
                  class.dec ()java.type:"void" -> {
                      %11 : Var<java.type:"LocalClassTest"> = var;
                      %12 : Var<java.type:"int"> = var;
                      %13 : Var<java.type:"int"> = var;
                      func @"<init>" (%14 : java.type:"LocalClassTest", %15 : java.type:"int", %16 : java.type:"int")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %11 %14;
                          var.store %12 %15;
                          var.store %13 %16;
                          return;
                      };
                      func @"s" (%17 : java.type:"LocalClassTest::$1Bar")java.type:"int" -> {
                          %18 : java.type:"int" = var.load %12;
                          return %18;
                      };
                      func @"foo" (%19 : java.type:"LocalClassTest::$1Bar")java.type:"LocalClassTest::$4Foo" -> {
                          %20 : java.type:"LocalClassTest" = var.load %11;
                          %21 : java.type:"int" = var.load %13;
                          %22 : java.type:"LocalClassTest::$4Foo" = new %20 %21 @java.ref:"LocalClassTest::$4Foo::(LocalClassTest, int)";
                          return %22;
                      };
                      yield;
                  };
                  %23 : java.type:"int" = var.load %3;
                  %24 : java.type:"int" = var.load %4;
                  %25 : java.type:"LocalClassTest::$1Bar" = new %0 %23 %24 @java.ref:"LocalClassTest::$1Bar::(LocalClassTest, int, int)";
                  return;
              };
            """)
    void testLocalDependency(int s, int i) { // static is like defined in a top level context, record sweet spot, worth keep pushing to model top level classes
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
            func @"testAnonDependency" (%0 : java.type:"LocalClassTest", %1 : java.type:"int", %2 : java.type:"int")java.type:"void" -> {
                  %3 : Var<java.type:"int"> = var %1 @"s";
                  %4 : Var<java.type:"int"> = var %2 @"i";
                  class.dec ()java.type:"void" -> {
                      %5 : Var<java.type:"LocalClassTest"> = var;
                      %6 : Var<java.type:"int"> = var;
                      func @"<init>" (%7 : java.type:"LocalClassTest", %8 : java.type:"int")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %5 %7;
                          var.store %6 %8;
                          return;
                      };
                      func @"i" (%9 : java.type:"LocalClassTest::$5Foo")java.type:"int" -> {
                          %10 : java.type:"int" = var.load %6;
                          return %10;
                      };
                      yield;
                  };
                  class.dec ()java.type:"void" -> {
                      %11 : Var<java.type:"LocalClassTest"> = var;
                      %12 : Var<java.type:"int"> = var;
                      %13 : Var<java.type:"int"> = var;
                      func @"<init>" (%14 : java.type:"LocalClassTest", %15 : java.type:"int", %16 : java.type:"int")java.type:"void" -> {
                          c.invoke @java.ref:"java.lang.Object::()";
                          var.store %11 %14;
                          var.store %12 %15;
                          var.store %13 %16;
                          return;
                      };
                      func @"s" (%17 : java.type:"LocalClassTest::$4")java.type:"int" -> {
                          %18 : java.type:"int" = var.load %12;
                          return %18;
                      };
                      func @"foo" (%19 : java.type:"LocalClassTest::$4")java.type:"LocalClassTest::$5Foo" -> {
                          %20 : java.type:"LocalClassTest" = var.load %11;
                          %21 : java.type:"int" = var.load %13;
                          %22 : java.type:"LocalClassTest::$5Foo" = new %20 %21 @java.ref:"LocalClassTest::$5Foo::(LocalClassTest, int)";
                          return %22;
                      };
                      yield;
                  };
                  %23 : java.type:"int" = var.load %3;
                  %24 : java.type:"int" = var.load %4;
                  %25 : java.type:"LocalClassTest::$4" = new %0 %23 %24 @java.ref:"LocalClassTest::$4::(LocalClassTest, int, int)";
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
