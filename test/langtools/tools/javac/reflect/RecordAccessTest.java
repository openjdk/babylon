/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
 * @summary Test for code reflection with record access.
 * @modules jdk.incubator.code
 * @build RecordAccessTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester RecordAccessTest
 */

import jdk.incubator.code.CodeReflection;

public class RecordAccessTest {

    record R() {
        static final String F = "";

        static void m() {
        }
    }

    @CodeReflection
    @IR("""
            func @"test1" (%0 : java.type:"RecordAccessTest")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = field.load @java.ref:"RecordAccessTest$R::F:java.lang.String";
                %2 : Var<java.type:"java.lang.String"> = var %1 @"f";
                return;
            };
            """)
    void test1() {
        String f = R.F;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : java.type:"RecordAccessTest")java.type:"void" -> {
                invoke @java.ref:"RecordAccessTest$R::m():void";
                return;
            };
            """)
    void test2() {
        R.m();
    }
}
