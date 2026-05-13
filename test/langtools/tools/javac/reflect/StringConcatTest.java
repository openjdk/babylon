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

/*
 * @test
 * @modules jdk.incubator.code
 * @build StringConcatTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester StringConcatTest
 */
public class StringConcatTest {

    @IR("""
            func @"test1" (%0 : java.type:"java.lang.String", %1 : java.type:"int")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"int"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"int" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                return %6;
            };
            """)
    @Reflect
    static String test1(String a, int b) {
        return a + b;
    }

    @IR("""
            func @"test2" (%0 : java.type:"java.lang.String", %1 : java.type:"char")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"char"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"char" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                var.store %2 %6;
                %7 : java.type:"java.lang.String" = var.load %2;
                return %7;
            };
            """)
    @Reflect
    static String test2(String a, char b) {
        a += b;
        return a;
    }

    @IR("""
            func @"test3" (%0 : java.type:"java.lang.String", %1 : java.type:"float")java.type:"java.lang.String" -> {
                %2 : Var<java.type:"java.lang.String"> = var %0 @"a";
                %3 : Var<java.type:"float"> = var %1 @"b";
                %4 : java.type:"java.lang.String" = var.load %2;
                %5 : java.type:"float" = var.load %3;
                %6 : java.type:"java.lang.String" = concat %4 %5;
                var.store %2 %6;
                %7 : java.type:"java.lang.String" = var.load %2;
                return %7;
            };
            """)
    @Reflect
    static String test3(String a, float b) {
        a = a + b;
        return a;
    }
}
