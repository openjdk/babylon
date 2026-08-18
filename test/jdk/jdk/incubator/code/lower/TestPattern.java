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
 * @summary test lowering of synchronized blocks
 * @build TestPattern
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestPattern
 */

import jdk.incubator.code.Reflect;

public class TestPattern {

    @Reflect
    @LoweredModel(value = """
            func @"match" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.String" = constant @null;
                %3 : Var<java.type:"java.lang.String"> = var %2 @"s";
                %4 : java.type:"java.lang.Object" = var.load %1;
                %5 : java.type:"boolean" = constant @false;
                %6 : java.type:"boolean" = instanceof %4 @java.type:"java.lang.String";
                cbranch %6 ^block_1 ^block_2(%5);

              ^block_1:
                %7 : java.type:"java.lang.String" = cast %4 @java.type:"java.lang.String";
                var.store %3 %7;
                %8 : java.type:"boolean" = constant @true;
                branch ^block_2(%8);

              ^block_2(%9 : java.type:"boolean"):
                cbranch %9 ^block_3 ^block_4;

              ^block_3:
                %10 : java.type:"java.lang.String" = var.load %3;
                invoke %10 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_5;

              ^block_4:
                branch ^block_5;

              ^block_5:
                return;
            };
            """)
    static void match(Object o) {
        if (o instanceof String s) {
            IO.println(s);
        }
    }


    @Reflect
    @LoweredModel(value = """
            func @"match2" (%0 : java.type:"java.lang.Object")java.type:"void" -> {
                %1 : java.type:"java.lang.String" = constant @null;
                %2 : java.type:"boolean" = constant @false;
                %3 : java.type:"boolean" = instanceof %0 @java.type:"java.lang.String";
                cbranch %3 ^block_1 ^block_2;

              ^block_1:
                %4 : java.type:"java.lang.String" = cast %0 @java.type:"java.lang.String";
                %5 : java.type:"boolean" = constant @true;
                invoke %4 @java.ref:"java.lang.IO::println(java.lang.Object):void";
                branch ^block_3;

              ^block_2:
                branch ^block_3;

              ^block_3:
                return;
            };
            """, transform = { LoweredModel.Transform.NORMALIZE_BLOCKS, LoweredModel.Transform.SSA})
    static void match2(Object o) {
        if (o instanceof String s) {
            IO.println(s);
        }
    }
}
