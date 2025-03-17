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
 * @summary Smoke test for code reflection with unary operations.
 * @modules jdk.incubator.code
 * @build UnaryopTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester UnaryopTest
 */

import jdk.incubator.code.CodeReflection;

public class UnaryopTest {
    @CodeReflection
    @IR("""
            func @"test" (%0 : int)int -> {
                %1 : Var<int> = var %0 @"v" ;
                %2 : int = var.load %1 ;
                %3 : int = neg %2 ;
                return %3 ;
            };
            """)
    static int test(int v) {
        return -v;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : int)int -> {
                %1 : Var<int> = var %0 @"v";
                %2 : int = var.load %1;
                return %2;
            };
            """)
    static int test2(int v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test3"  (%0 : int)java.lang.Integer -> {
                %1 : Var<int> = var %0 @"v" ;
                %2 : int = var.load %1 ;
                %3 : java.lang.Integer = invoke %2 @"java.lang.Integer::valueOf(int)java.lang.Integer" ;
                return %3 ;
            };
            """)
    // Tests that numeric promotion occurs
    static Integer test3(int v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test4"  (%0 : java.lang.Integer)java.lang.Integer -> {
                %1 : Var<java.lang.Integer> = var %0 @"v" ;
                %2 : java.lang.Integer = var.load %1 ;
                %3 : int = invoke %2 @"java.lang.Integer::intValue()int" ;
                %4 : java.lang.Integer = invoke %3 @"java.lang.Integer::valueOf(int)java.lang.Integer" ;
                return %4 ;
            };
            """)
    // Tests that numeric promotion is retained
    static Integer test4(Integer v) {
        return +v;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : int)int -> {
                %1 : Var<int> = var %0 @"v" ;
                %2 : int = var.load %1 ;
                %3 : int = compl %2 ;
                return %3 ;
            };
            """)
    static int test5(int v) {
        return ~v;
    }

    @IR("""
            func @"test6" (%0 : byte)void -> {
                  %1 : Var<byte> = var %0 @"b";
                  %2 : byte = var.load %1;
                  %3 : int = constant @"1";
                  %4 : byte = conv %3;
                  %5 : byte = add %2 %4;
                  var.store %1 %5;
                  %6 : byte = var.load %1;
                  %7 : int = constant @"1";
                  %8 : byte = conv %7;
                  %9 : byte = sub %6 %8;
                  var.store %1 %9;
                  %10 : byte = var.load %1;
                  %11 : int = constant @"1";
                  %12 : byte = conv %11;
                  %13 : byte = add %10 %12;
                  var.store %1 %13;
                  %14 : byte = var.load %1;
                  %15 : int = constant @"1";
                  %16 : byte = conv %15;
                  %17 : byte = sub %14 %16;
                  var.store %1 %17;
                  return;
              };
            """)
    @CodeReflection
    static void test6(byte b) {
        b++;
        b--;
        ++b;
        --b;
    }

    @IR("""
            func @"test7" (%0 : short)void -> {
                  %1 : Var<short> = var %0 @"s";
                  %2 : short = var.load %1;
                  %3 : int = constant @"1";
                  %4 : short = conv %3;
                  %5 : short = add %2 %4;
                  var.store %1 %5;
                  %6 : short = var.load %1;
                  %7 : int = constant @"1";
                  %8 : short = conv %7;
                  %9 : short = sub %6 %8;
                  var.store %1 %9;
                  %10 : short = var.load %1;
                  %11 : int = constant @"1";
                  %12 : short = conv %11;
                  %13 : short = add %10 %12;
                  var.store %1 %13;
                  %14 : short = var.load %1;
                  %15 : int = constant @"1";
                  %16 : short = conv %15;
                  %17 : short = sub %14 %16;
                  var.store %1 %17;
                  return;
              };
            """)
    @CodeReflection
    static void test7(short s) {
        s++;
        s--;
        ++s;
        --s;
    }
}
