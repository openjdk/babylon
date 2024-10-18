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
 * @summary test lowering of synchronized blocks
 * @build TestSynchronized
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestSynchronized
 */

import java.lang.runtime.CodeReflection;

public class TestSynchronized {

    @CodeReflection
    @LoweredModel(value = """
            func @"test1" (%0 : java.lang.Object, %1 : int)int -> {
                %2 : Var<java.lang.Object> = var %0 @"m";
                %3 : Var<int> = var %1 @"i";
                %4 : java.lang.Object = var.load %2;
                branch ^block_1(%4);

              ^block_1(%5 : java.lang.Object):
                monitor.enter %5;
                exception.region.enter ^block_2 ^block_4;

              ^block_2:
                %7 : int = var.load %3;
                %8 : int = constant @"1";
                %9 : int = add %7 %8;
                var.store %3 %9;
                monitor.exit %5;
                exception.region.exit ^block_3 ^block_4;

              ^block_3:
                %10 : int = var.load %3;
                return %10;

              ^block_4(%11 : java.lang.Throwable):
                exception.region.enter ^block_5 ^block_4;

              ^block_5:
                monitor.exit %5;
                exception.region.exit ^block_6 ^block_4;

              ^block_6:
                throw %11;
            };
            """, ssa = false)
    static int test1(Object m, int i) {
        synchronized (m) {
            i++;
        }
        return i;
    }


    @CodeReflection
    @LoweredModel(value = """
            func @"test2" (%0 : java.lang.Object, %1 : int)int -> {
                %2 : Var<java.lang.Object> = var %0 @"m";
                %3 : Var<int> = var %1 @"i";
                %4 : java.lang.Object = var.load %2;
                branch ^block_1(%4);

              ^block_1(%5 : java.lang.Object):
                monitor.enter %5;
                exception.region.enter ^block_2 ^block_8;

              ^block_2:
                %7 : int = var.load %3;
                %8 : int = constant @"0";
                %9 : boolean = gt %7 %8;
                cbranch %9 ^block_3 ^block_5;

              ^block_3:
                %10 : int = constant @"-1";
                monitor.exit %5;
                exception.region.exit ^block_4 ^block_8;

              ^block_4:
                return %10;

              ^block_5:
                branch ^block_6;

              ^block_6:
                %11 : int = var.load %3;
                %12 : int = constant @"1";
                %13 : int = add %11 %12;
                var.store %3 %13;
                monitor.exit %5;
                exception.region.exit ^block_7 ^block_8;

              ^block_7:
                %14 : int = var.load %3;
                return %14;

              ^block_8(%15 : java.lang.Throwable):
                exception.region.enter ^block_9 ^block_8;

              ^block_9:
                monitor.exit %5;
                exception.region.exit ^block_10 ^block_8;

              ^block_10:
                throw %15;
            };
            """, ssa = false)
    static int test2(Object m, int i) {
        synchronized (m) {
            if (i > 0) {
                return -1;
            }
            i++;
        }
        return i;
    }

}
