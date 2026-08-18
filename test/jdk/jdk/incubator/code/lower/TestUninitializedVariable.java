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
 * @build TestUninitializedVariable
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestUninitializedVariable
 */

import jdk.incubator.code.Reflect;

public class TestUninitializedVariable {

    @Reflect
    @LoweredModel(value = """
            func @"definitiveAssignment1" (%0 : java.type:"int")java.type:"int" -> {
                %1 : Var<java.type:"int"> = var %0 @"i";
                %2 : Var<java.type:"int"> = var @"assigned";
                %3 : java.type:"boolean" = constant @false;
                %4 : java.type:"int" = var.load %1;
                %5 : java.type:"int" = constant @0;
                %6 : java.type:"boolean" = gt %4 %5;
                cbranch %6 ^block_1 ^block_2(%3);

              ^block_1:
                %7 : java.type:"int" = var.load %1;
                var.store %2 %7;
                %8 : java.type:"int" = constant @1;
                %9 : java.type:"boolean" = gt %7 %8;
                branch ^block_2(%9);

              ^block_2(%10 : java.type:"boolean"):
                cbranch %10 ^block_3 ^block_4;

              ^block_3:
                %11 : java.type:"int" = var.load %2;
                return %11;

              ^block_4:
                branch ^block_5;

              ^block_5:
                %12 : java.type:"int" = constant @-1;
                return %12;
            };
            """)
    static int definitiveAssignment1(int i) {
        int assigned;
        if (i > 0 && (assigned = i) > 1) {
            return assigned;
        }
        return -1;
    }


    @Reflect
    @LoweredModel(value = """
            func @"definitiveAssignment2" (%0 : java.type:"int")java.type:"int" -> {
                %1 : java.type:"boolean" = constant @false;
                %2 : java.type:"int" = constant @0;
                %3 : java.type:"boolean" = gt %0 %2;
                cbranch %3 ^block_1 ^block_4;

              ^block_1:
                %4 : java.type:"int" = constant @1;
                %5 : java.type:"boolean" = gt %0 %4;
                branch ^block_2(%5);

              ^block_2(%6 : java.type:"boolean"):
                cbranch %6 ^block_3 ^block_4;

              ^block_3:
                return %0;

              ^block_4:
                %7 : java.type:"int" = constant @-1;
                return %7;
            };
            """, transform = { LoweredModel.Transform.NORMALIZE_BLOCKS, LoweredModel.Transform.SSA})
    static int definitiveAssignment2(int i) {
        int assigned;
        if (i > 0 && (assigned = i) > 1) {
            return assigned;
        }
        return -1;
    }
}
