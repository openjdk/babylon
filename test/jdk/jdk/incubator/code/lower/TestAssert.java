/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @modules jdk.incubator.code
 * @summary test lowering of assert statements
 * @build TestAssert
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestAssert
 */

import jdk.incubator.code.Reflect;

public class TestAssert {

    @Reflect
    @LoweredModel(value = """
            func @"test1" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"java.lang.String" = constant @"Failed";
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(java.lang.Object)";
                throw %7;
            };
            """, ssa = false)
    static int test1(int i) {
        assert i >= 0 : "Failed";
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"test2" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"java.lang.AssertionError" = new @java.ref:"java.lang.AssertionError::()";
                throw %6;
            };
            """, ssa = false)
    static int test2(int i) {
        assert i >= 0;
        return i;
    }

}
