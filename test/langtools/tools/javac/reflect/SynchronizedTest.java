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
 * @summary Smoke test for code reflection with synchronized blocks.
 * @modules jdk.incubator.code
 * @build SynchronizedTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester SynchronizedTest
 */

import jdk.incubator.code.CodeReflection;

public class SynchronizedTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : SynchronizedTest, %1 : int)int -> {
                %2 : Var<int> = var %1 @"i";
                java.synchronized
                    ()SynchronizedTest -> {
                        yield %0;
                    }
                    ()void -> {
                        %3 : int = var.load %2;
                        %4 : int = constant @"1";
                        %5 : int = add %3 %4;
                        var.store %2 %5;
                        yield;
                    };
                %6 : int = var.load %2;
                return %6;
            };
            """)
    int test1(int i) {
        synchronized (this) {
            i++;
        }
        return i;
    }

    static Object m() {
        return null;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : SynchronizedTest, %1 : int)int -> {
                %2 : Var<int> = var %1 @"i";
                java.synchronized
                    ()java.lang.Object -> {
                        %3 : java.lang.Object = invoke @"SynchronizedTest::m()java.lang.Object";
                        yield %3;
                    }
                    ()void -> {
                        %4 : int = var.load %2;
                        %5 : int = constant @"1";
                        %6 : int = add %4 %5;
                        var.store %2 %6;
                        yield;
                    };
                %7 : int = var.load %2;
                return %7;
            };
            """)
    int test2(int i) {
        synchronized (m()) {
            i++;
        }
        return i;
    }

}
