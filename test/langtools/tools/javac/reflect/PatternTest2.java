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

import jdk.incubator.code.Reflect;

/*
 * @test
 * @modules jdk.incubator.code
 * @build PatternTest2
 * @build CodeReflectionTester
 * @run main CodeReflectionTester PatternTest2
 */
public class PatternTest2 {
    record R<T extends Number> (T n) {}

    @IR("""
            func @"f" (%0 : java.type:"java.lang.Object")java.type:"boolean" -> {
                %1 : Var<java.type:"java.lang.Object"> = var %0 @"o";
                %2 : java.type:"java.lang.Object" = var.load %1;
                %3 : java.type:"java.lang.Integer" = constant @null;
                %4 : Var<java.type:"java.lang.Integer"> = var %3 @"i";
                %5 : java.type:"boolean" = pattern.match %2
                    ()java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>>" -> {
                        %6 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Type<java.lang.Integer>" = pattern.type @"i";
                        %7 : java.type:"jdk.incubator.code.dialect.java.JavaOp$Pattern$Record<PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>>" = pattern.record %6 @java.ref:"(PatternTest2$R::<T extends java.lang.Number> n)PatternTest2$R<PatternTest2$R::<T extends java.lang.Number>>";
                        yield %7;
                    }
                    (%8 : java.type:"java.lang.Integer")java.type:"void" -> {
                        var.store %4 %8;
                        yield;
                    };
                return %5;
            };
            """)
    @Reflect
    static boolean f(Object o) {
        return o instanceof R(Integer i);
    }
}
