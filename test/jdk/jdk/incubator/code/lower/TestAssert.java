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
            func @"testNoDetail" (%0 : java.type:"int")java.type:"int" -> {
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
    static int testNoDetail(int i) {
        assert i >= 0;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testStringDetail" (%0 : java.type:"int")java.type:"int" -> {
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
    static int testStringDetail(int i) {
        assert i >= 0 : "Failed";
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testBooleanDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"boolean" = constant @false;
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(boolean)";
                throw %7;
            };
            """, ssa = false)
    static int testBooleanDetail(int i) {
        assert i >= 0 : false;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testCharDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"char" = constant @'a';
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(char)";
                throw %7;
            };
            """, ssa = false)
    static int testCharDetail(int i) {
        assert i >= 0 : 'a';
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testByteDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"int" = constant @10;
                %7 : java.type:"byte" = conv %6;
                %8 : java.type:"int" = conv %7;
                %9 : java.type:"java.lang.AssertionError" = new %8 @java.ref:"java.lang.AssertionError::(int)";
                throw %9;
            };
            """, ssa = false)
    static int testByteDetail(int i) {
        assert i >= 0 : (byte) 10;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testShortDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"int" = constant @10;
                %7 : java.type:"short" = conv %6;
                %8 : java.type:"int" = conv %7;
                %9 : java.type:"java.lang.AssertionError" = new %8 @java.ref:"java.lang.AssertionError::(int)";
                throw %9;
            };
            """, ssa = false)
    static int testShortDetail(int i) {
        assert i >= 0 : (short) 10;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testIntDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"int" = constant @10;
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(int)";
                throw %7;
            };
            """, ssa = false)
    static int testIntDetail(int i) {
        assert i >= 0 : 10;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testLongDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"long" = constant @10;
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(long)";
                throw %7;
            };
            """, ssa = false)
    static int testLongDetail(int i) {
        assert i >= 0 : 10l;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testFloatDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"float" = constant @1.0f;
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(float)";
                throw %7;
            };
            """, ssa = false)
    static int testFloatDetail(int i) {
        assert i >= 0 : 1.0f;
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"testDoubleDetail" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : java.type:"int" = var.load %1;
                %3 : java.type:"int" = constant @0;
                %4 : java.type:"boolean" = ge %2 %3;
                cbranch %4 ^block_1 ^block_2;

              ^block_1:
                %5 : java.type:"int" = var.load %1;
                return %5;

              ^block_2:
                %6 : java.type:"double" = constant @1.0;
                %7 : java.type:"java.lang.AssertionError" = new %6 @java.ref:"java.lang.AssertionError::(double)";
                throw %7;
            };
            """, ssa = false)
    static int testDoubleDetail(int i) {
        assert i >= 0 : 1.0;
        return i;
    }
}
