/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @build TestSynchronized
 * @build CodeReflectionTester
 * @run main CodeReflectionTester TestSynchronized
 */

import jdk.incubator.code.Reflect;

public class TestSynchronized {

    @Reflect
    @LoweredModel(value = """
            func @"test1" (%0 : java.type:"java.lang.Object", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"m";
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : java.type:"java.lang.Object" = var.load %2;
                branch ^block_1(%4);

              ^block_1(%5 : java.type:"java.lang.Object"):
                monitor.enter %5;
                %13 : java.type:"java.lang.Throwable" = constant @null;
                %6 : java.type:"void" = exception.region.enter ^block_2 ^block_4(%13);

              ^block_2:
                %7 : java.type:"int" = var.load %3;
                %8 : java.type:"int" = constant @1;
                %9 : java.type:"int" = add %7 %8;
                var.store %3 %9;
                monitor.exit %5;
                exception.region.exit %6 ^block_3;

              ^block_3:
                %10 : java.type:"int" = var.load %3;
                return %10;

              ^block_4(%11 : java.type:"java.lang.Throwable"):
                %14 : java.type:"java.lang.Throwable" = constant @null;
                %12 : java.type:"void" = exception.region.enter ^block_5 ^block_4(%14);

              ^block_5:
                monitor.exit %5;
                exception.region.exit %12 ^block_6;

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


    @Reflect
    @LoweredModel(value = """
            func @"test2" (%0 : java.type:"java.lang.Object", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"m";
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : java.type:"java.lang.Object" = var.load %2;
                branch ^block_1(%4);

              ^block_1(%5 : java.type:"java.lang.Object"):
                monitor.enter %5;
                %6 : java.type:"java.lang.Throwable" = constant @null;
                %7 : java.type:"void" = exception.region.enter ^block_2 ^block_7(%6);

              ^block_2:
                %8 : java.type:"int" = var.load %3;
                %9 : java.type:"int" = constant @0;
                %10 : java.type:"boolean" = gt %8 %9;
                cbranch %10 ^block_3 ^block_5;

              ^block_3:
                %11 : java.type:"int" = constant @-1;
                monitor.exit %5;
                exception.region.exit %7 ^block_4;

              ^block_4:
                return %11;

              ^block_5:
                %12 : java.type:"int" = var.load %3;
                %13 : java.type:"int" = constant @1;
                %14 : java.type:"int" = add %12 %13;
                var.store %3 %14;
                monitor.exit %5;
                exception.region.exit %7 ^block_6;

              ^block_6:
                %15 : java.type:"int" = var.load %3;
                return %15;

              ^block_7(%16 : java.type:"java.lang.Throwable"):
                %17 : java.type:"java.lang.Throwable" = constant @null;
                %18 : java.type:"void" = exception.region.enter ^block_8 ^block_7(%17);

              ^block_8:
                monitor.exit %5;
                exception.region.exit %18 ^block_9;

              ^block_9:
                throw %16;
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


    @Reflect
    @LoweredModel(value = """
            func @"test3" (%0 : java.type:"java.lang.Object", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"m";
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : java.type:"java.lang.Object" = var.load %2;
                %5 : java.type:"java.lang.Object" = constant @null;
                %6 : java.type:"boolean" = invoke %4 %5 @java.ref:"java.util.Objects::equals(java.lang.Object, java.lang.Object):boolean";
                cbranch %6 ^block_1 ^block_2;

              ^block_1:
                %7 : java.type:"java.lang.NullPointerException" = new @java.ref:"java.lang.NullPointerException::()";
                throw %7;

              ^block_2:
                %8 : java.type:"java.lang.Object" = var.load %2;
                branch ^block_3(%8);

              ^block_3(%9 : java.type:"java.lang.Object"):
                monitor.enter %9;
                %10 : java.type:"java.lang.Throwable" = constant @null;
                %11 : java.type:"void" = exception.region.enter ^block_4 ^block_10(%10);

              ^block_4:
                %12 : java.type:"int" = var.load %3;
                %13 : java.type:"int" = constant @0;
                %14 : java.type:"boolean" = gt %12 %13;
                cbranch %14 ^block_5 ^block_7;

              ^block_5:
                %15 : java.type:"int" = var.load %3;
                monitor.exit %9;
                exception.region.exit %11 ^block_6;

              ^block_6:
                branch ^block_9(%15);

              ^block_7:
                %16 : java.type:"int" = constant @0;
                monitor.exit %9;
                exception.region.exit %11 ^block_8;

              ^block_8:
                branch ^block_9(%16);

              ^block_9(%17 : java.type:"int"):
                return %17;

              ^block_10(%18 : java.type:"java.lang.Throwable"):
                %19 : java.type:"java.lang.Throwable" = constant @null;
                %20 : java.type:"void" = exception.region.enter ^block_11 ^block_10(%19);

              ^block_11:
                monitor.exit %9;
                exception.region.exit %20 ^block_12;

              ^block_12:
                throw %18;
            };
            """, ssa = false)
    static int test3(Object m, int i) {
        return switch (m) {
            default -> {
                synchronized (m) {
                    if (i > 0) {
                        yield i;
                    }
                    yield 0;
                }
            }
        };
    }


    @Reflect
    @LoweredModel(value = """
            func @"test4" (%0 : java.type:"java.lang.Object", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"m";
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : java.type:"java.lang.Object" = var.load %2;
                branch ^block_1(%4);

              ^block_1(%5 : java.type:"java.lang.Object"):
                monitor.enter %5;
                %6 : java.type:"java.lang.Throwable" = constant @null;
                %7 : java.type:"void" = exception.region.enter ^block_2 ^block_8(%6);

              ^block_2:
                %8 : java.type:"int" = var.load %3;
                %9 : java.type:"int" = constant @0;
                %10 : java.type:"boolean" = gt %8 %9;
                cbranch %10 ^block_3 ^block_5;

              ^block_3:
                %11 : java.type:"int" = constant @42;
                var.store %3 %11;
                monitor.exit %5;
                exception.region.exit %7 ^block_4;

              ^block_4:
                branch ^block_6;

              ^block_5:
                %12 : java.type:"int" = var.load %3;
                %13 : java.type:"int" = constant @1;
                %14 : java.type:"int" = add %12 %13;
                var.store %3 %14;
                monitor.exit %5;
                exception.region.exit %7 ^block_6;

              ^block_6:
                branch ^block_7;

              ^block_7:
                %15 : java.type:"int" = var.load %3;
                return %15;

              ^block_8(%16 : java.type:"java.lang.Throwable"):
                %17 : java.type:"java.lang.Throwable" = constant @null;
                %18 : java.type:"void" = exception.region.enter ^block_9 ^block_8(%17);

              ^block_9:
                monitor.exit %5;
                exception.region.exit %18 ^block_10;

              ^block_10:
                throw %16;
            };
            """, ssa = false)
    static int test4(Object m, int i) {
        x: synchronized (m) {
            if (i > 0) {
                i = 42;
                break x;
            }
            i++;
        }
        return i;
    }

    @Reflect
    @LoweredModel(value = """
            func @"test5" (%0 : java.type:"java.lang.Object", %1 : java.type:"int")java.type:"int" -> {
                %2 : Var<java.type:"java.lang.Object"> = var %0 @"m";
                %3 : Var<java.type:"int"> = var %1 @"i";
                %4 : java.type:"int" = constant @0;
                %5 : Var<java.type:"int"> = var %4 @"j";
                branch ^block_1;

              ^block_1:
                %6 : java.type:"int" = var.load %5;
                %7 : java.type:"int" = constant @10;
                %8 : java.type:"boolean" = lt %6 %7;
                cbranch %8 ^block_2 ^block_13;

              ^block_2:
                %9 : java.type:"java.lang.Object" = var.load %2;
                branch ^block_3(%9);

              ^block_3(%10 : java.type:"java.lang.Object"):
                monitor.enter %10;
                %11 : java.type:"java.lang.Throwable" = constant @null;
                %12 : java.type:"void" = exception.region.enter ^block_4 ^block_10(%11);

              ^block_4:
                %13 : java.type:"int" = var.load %3;
                %14 : java.type:"int" = constant @0;
                %15 : java.type:"boolean" = gt %13 %14;
                cbranch %15 ^block_5 ^block_7;

              ^block_5:
                %16 : java.type:"int" = constant @42;
                var.store %3 %16;
                monitor.exit %10;
                exception.region.exit %12 ^block_6;

              ^block_6:
                branch ^block_9;

              ^block_7:
                %17 : java.type:"int" = var.load %3;
                %18 : java.type:"int" = constant @1;
                %19 : java.type:"int" = add %17 %18;
                var.store %3 %19;
                monitor.exit %10;
                exception.region.exit %12 ^block_8;

              ^block_8:
                branch ^block_9;

              ^block_9:
                %20 : java.type:"int" = var.load %5;
                %21 : java.type:"int" = constant @1;
                %22 : java.type:"int" = add %20 %21;
                var.store %5 %22;
                branch ^block_1;

              ^block_10(%23 : java.type:"java.lang.Throwable"):
                %24 : java.type:"java.lang.Throwable" = constant @null;
                %25 : java.type:"void" = exception.region.enter ^block_11 ^block_10(%24);

              ^block_11:
                monitor.exit %10;
                exception.region.exit %25 ^block_12;

              ^block_12:
                throw %23;

              ^block_13:
                %26 : java.type:"int" = var.load %3;
                return %26;
            };
            """, ssa = false)
    static int test5(Object m, int i) {
        for (int j = 0; j < 10; j++) {
            synchronized (m) {
                if (i > 0) {
                    i = 42;
                    continue;
                }
                i++;
            }
        }
        return i;
    }

}
