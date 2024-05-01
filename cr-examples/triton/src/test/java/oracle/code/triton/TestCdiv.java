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

package oracle.code.triton;

import oracle.code.triton.TritonTestExtension.TritonTestData;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;

import static oracle.code.triton.Triton.*;
import static oracle.code.triton.TritonTest.consume;

@ExtendWith(TritonTestExtension.class)
public class TestCdiv {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"cdiv_int_int_int" (%0 : int, %1 : int)int -> {
                    %2 : int = arith.addi %0 %1;
                    %3 : int = arith.constant @"1";
                    %4 : int = arith.subi %2 %3;
                    %5 : int = arith.divsi %4 %1;
                    tt.return %5;
                };
                tt.func @"testScalar_int_int_void" (%6 : int, %7 : int)void -> {
                    %8 : int = tt.call %6 %7 @"cdiv_int_int_int";
                    tt.consume %8;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void testScalar(int a, int b) {
        var r1 = cdiv(a, b);
        consume(r1);
    }

    @Test
    public void testScalar(TritonTestData t) {
        List<CodeType> argTypes = List.of(
                JavaType.INT,
                JavaType.INT);

        t.test(argTypes);
    }


    @TritonCodeModel("""
            module ()void -> {
                tt.func @"cdiv_int_10_int" (%0 : int)int -> {
                    %1 : int = arith.constant @"10";
                    %2 : int = arith.addi %0 %1;
                    %3 : int = arith.constant @"1";
                    %4 : int = arith.subi %2 %3;
                    %5 : int = arith.divsi %4 %1;
                    tt.return %5;
                };
                tt.func @"testConstant_int_10_void" (%6 : int)void -> {
                    %7 : int = tt.call %6 @"cdiv_int_10_int";
                    tt.consume %7;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void testConstant(int a, int b) {
        var r1 = cdiv(a, b);
        consume(r1);
    }

    @Test
    public void testConstant(TritonTestData t) {
        List<CodeType> argTypes = List.of(
                JavaType.INT,
                new ConstantType(JavaType.INT, 10));

        t.test(argTypes);
    }


    @TritonCodeModel("""
            module ()void -> {
                tt.func @"cdiv_int_int_int" (%0 : int, %1 : int)int -> {
                    %2 : int = arith.addi %0 %1;
                    %3 : int = arith.constant @"1";
                    %4 : int = arith.subi %2 %3;
                    %5 : int = arith.divsi %4 %1;
                    tt.return %5;
                };
                tt.func @"cdiv_int_10_int" (%6 : int)int -> {
                    %7 : int = arith.constant @"10";
                    %8 : int = arith.addi %6 %7;
                    %9 : int = arith.constant @"1";
                    %10 : int = arith.subi %8 %9;
                    %11 : int = arith.divsi %10 %7;
                    tt.return %11;
                };
                tt.func @"cdiv_10_int_int" (%12 : int)int -> {
                    %13 : int = arith.constant @"10";
                    %14 : int = arith.addi %13 %12;
                    %15 : int = arith.constant @"1";
                    %16 : int = arith.subi %14 %15;
                    %17 : int = arith.divsi %16 %12;
                    tt.return %17;
                };
                tt.func @"testCalls_int_int_10_void" (%18 : int, %19 : int)void -> {
                    %20 : int = tt.call %18 %19 @"cdiv_int_int_int";
                    tt.consume %20;
                    %21 : int = tt.call %19 %18 @"cdiv_int_int_int";
                    tt.consume %21;
                    %22 : int = tt.call %18 @"cdiv_int_10_int";
                    tt.consume %22;
                    %23 : int = tt.call %18 @"cdiv_10_int_int";
                    tt.consume %23;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void testCalls(int a, int b, int c) {
        consume(cdiv(a, b));
        consume(cdiv(b, a));
        consume(cdiv(a, c));
        consume(cdiv(c, a));
    }

    @Test
    public void testCalls(TritonTestData t) {
        List<CodeType> argTypes = List.of(
                JavaType.INT,
                JavaType.INT,
                new ConstantType(JavaType.INT, 10));

        t.test(argTypes);
    }
}