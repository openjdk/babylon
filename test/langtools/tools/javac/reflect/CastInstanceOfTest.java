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
import java.util.Collection;
import java.util.List;

/*
 * @test
 * @summary Smoke test for code reflection with cast expressions.
 * @build CastInstanceOfTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester CastInstanceOfTest
 */

public class CastInstanceOfTest {

    @CodeReflection
    @IR("""
            func @"test1" (%0 : CastInstanceOfTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.Object = var.load %2;
                %4 : java.lang.String = cast %3 @"java.lang.String";
                %5 : Var<java.lang.String> = var %4 @"s";
                %6 : java.lang.String = var.load %5;
                %7 : Var<java.lang.String> = var %6 @"ss";
                return;
            };
            """)
    void test1(Object o) {
        String s = (String) o;
        String ss = (String) s;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : CastInstanceOfTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.Object = var.load %2;
                %4 : java.util.List<java.lang.String> = cast %3 @"java.util.List";
                %5 : Var<java.util.List<java.lang.String>> = var %4 @"l";
                %6 : java.util.List<java.lang.String> = var.load %5;
                %7 : Var<java.util.Collection<java.lang.String>> = var %6 @"c1";
                %8 : java.util.List<java.lang.String> = var.load %5;
                %9 : Var<java.util.Collection<java.lang.String>> = var %8 @"c2";
                return;
            };
            """)
    void test2(Object o) {
        List<String> l = (List<String>) o;
        Collection<String> c1 = (List<String>) l;
        Collection<String> c2 = (Collection) l;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : CastInstanceOfTest, %1 : java.util.List<java.lang.String>)void -> {
                %2 : Var<java.util.List<java.lang.String>> = var %1 @"l";
                %3 : java.util.List<java.lang.String> = var.load %2;
                %4 : Var<java.util.List> = var %3 @"raw";
                %5 : java.util.List = var.load %4;
                %6 : Var<java.util.List<java.lang.Number>> = var %5 @"ln";
                return;
            };
            """)
    void test3(List<String> l) {
        List raw = l;
        List<Number> ln = raw;
    }

    @CodeReflection
    @IR("""
            func @"test4" (%0 : CastInstanceOfTest, %1 : int)void -> {
                %2 : Var<int> = var %1 @"i";
                %3 : int = var.load %2;
                %4 : long = conv %3;
                %5 : Var<long> = var %4 @"l";
                return;
            };
            """)
    void test4(int i) {
        long l = (int) i;
    }

    @CodeReflection
    @IR("""
            func @"test5" (%0 : CastInstanceOfTest, %1 : java.lang.Object)void -> {
                %2 : Var<java.lang.Object> = var %1 @"o";
                %3 : java.lang.Object = var.load %2;
                %4 : boolean = instanceof %3 @"java.lang.String";
                %5 : Var<boolean> = var %4 @"b";
                return;
            };
            """)
    void test5(Object o) {
        boolean b = o instanceof String;
    }

    @CodeReflection
    @IR("""
            func @"test6" (%0 : CastInstanceOfTest, %1 : java.util.List<java.lang.Object>)void -> {
                %2 : Var<java.util.List<java.lang.Object>> = var %1 @"l";
                %3 : java.util.List<java.lang.Object> = var.load %2;
                %4 : int = constant @"0";
                %5 : java.lang.Object = invoke %3 %4 @"java.util.List::get(int)java.lang.Object";
                %6 : boolean = instanceof %5 @"java.lang.String";
                %7 : Var<boolean> = var %6 @"b";
                return;
            };
            """)
    void test6(List<Object> l) {
        boolean b = l.get(0) instanceof String;
    }
}
