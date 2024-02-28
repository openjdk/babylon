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

/*
 * @test
 * @enablePreview
 * @build StringTemplateTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester StringTemplateTest
 */

import java.lang.runtime.CodeReflection;

import static java.lang.StringTemplate.RAW;
import static java.lang.StringTemplate.STR;

public class StringTemplateTest {

    @CodeReflection
    @IR("""
            func @"f" (%0 : int)void -> {
                  %1 : Var<int> = var %0 @"y";
                  %2 : java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::STR()java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException>";
                  %3 : java.lang.String = constant @"y = ";
                  %4 : java.lang.String = constant @"";
                  %5 : java.lang.String = java.stringTemplate %2 %3 %4 ()int -> {
                      %6 : int = var.load %1;
                      yield %6;
                  };
                  %7 : Var<java.lang.String> = var %5 @"s";
                  return;
              };
            """)
    static void f(int y) {
        String s = STR. "y = \{y}";
    }

    @CodeReflection
    @IR("""
            func @"f2" (%0 : int)void -> {
                  %1 : Var<int> = var %0 @"y";
                  %2 : java.lang.StringTemplate$Processor<java.lang.StringTemplate, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::RAW()java.lang.StringTemplate$Processor<java.lang.StringTemplate, java.lang.RuntimeException>";
                  %3 : java.lang.String = constant @"y = ";
                  %4 : java.lang.String = constant @"";
                  %5 : java.lang.StringTemplate = java.stringTemplate %2 %3 %4 ()int -> {
                      %6 : int = var.load %1;
                      yield %6;
                  };
                  %7 : Var<java.lang.StringTemplate> = var %5 @"st";
                  return;
              };
            """)
    static void f2(int y) {
        StringTemplate st = RAW. "y = \{y}";
    }

    @CodeReflection
    @IR("""
            func @"f3" (%0 : int, %1 : int, %2 : int)void -> {
                  %3 : Var<int> = var %0 @"x";
                  %4 : Var<int> = var %1 @"y";
                  %5 : Var<int> = var %2 @"z";
                  %6 : java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::STR()java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException>";
                  %7 : java.lang.String = constant @"x = ";
                  %8 : java.lang.String = constant @", z = ";
                  %9 : java.lang.String = constant @", x + z = ";
                  %10 : java.lang.String = constant @"";
                  %11 : java.lang.String = java.stringTemplate %6 %7 %8 %9 %10
                      ()int -> {
                          %12 : int = var.load %3;
                          yield %12;
                      }
                      ()int -> {
                          %13 : int = var.load %5;
                          yield %13;
                      }
                      ()int -> {
                          %14 : int = var.load %3;
                          %15 : int = var.load %5;
                          %16 : int = add %14 %15;
                          yield %16;
                      };
                  %17 : Var<java.lang.String> = var %11 @"s";
                  %18 : java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException> = field.load @"java.lang.StringTemplate::STR()java.lang.StringTemplate$Processor<java.lang.String, java.lang.RuntimeException>";
                  %19 : java.lang.String = constant @"y = ";
                  %20 : java.lang.String = constant @", ";
                  %21 : java.lang.String = constant @"";
                  %22 : java.lang.String = java.stringTemplate %18 %19 %20 %21
                      ()int -> {
                          %23 : int = var.load %4;
                          yield %23;
                      }
                      ()java.lang.String -> {
                          %24 : java.lang.String = var.load %17;
                          yield %24;
                      };
                  %25 : Var<java.lang.String> = var %22 @"s2";
                  return;
              };
            """)
    static void f3(int x, int y, int z) {
        String s = STR."x = \{x}, z = \{z}, x + z = \{x + z}";
        String s2 = STR."y = \{y}, \{s}";
    }
}
