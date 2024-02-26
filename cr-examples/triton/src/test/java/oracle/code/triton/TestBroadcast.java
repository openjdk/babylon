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

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;

import static oracle.code.triton.Triton.*;
import static oracle.code.triton.TritonTest.consume;

@ExtendWith(TritonTestExtension.class)
public class TestBroadcast {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"test1_ptr<int>_int_64_void" (%0 : ptr<int>, %1 : int)void -> {
                    %2 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %3 : tensor<x64, ptr<int>> = tt.splat %0;
                    %4 : tensor<x64, ptr<int>> = tt.addptr %3 %2;
                    tt.consume %4;
                    %5 : tensor<x64, int> = tt.splat %1;
                    %6 : tensor<x64, int> = arith.addi %5 %2;
                    tt.consume %6;
                    %7 : tensor<x64, int> = tt.splat %1;
                    %8 : tensor<x64, int> = arith.addi %2 %7;
                    tt.consume %8;
                    %9 : tensor<x64, int> = tt.splat %1;
                    %10 : tensor<x64, int> = arith.addi %9 %2;
                    tt.consume %10;
                    %11 : tensor<x64, int> = tt.splat %1;
                    %12 : tensor<x64, int> = arith.addi %2 %11;
                    tt.consume %12;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void test1(Ptr ptr, int a, @Constant int s) {
        var t = arange(0, s);
        consume(add(ptr, t));
        consume(add(a, t));
        consume(add(t, a));
        consume(add(broadcast(a, t.type()), t));
        consume(add(t, broadcast(a, t.type())));
    }

    @Test
    public void test1(TritonTestData t) {
        List<TypeElement> argTypes = List.of(
                new PtrType(JavaType.INT),
                JavaType.INT,
                new ConstantType(JavaType.INT, 64));

        t.test(argTypes);
    }

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"test2_int_64_32_void" (%1 : int)void -> {
                    %2 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %3 : tensor<x1, x64, int> = tt.expand_dims %2 @"0";
                    %4 : tensor<x32, int> = tt.make_range @start="0" @end="32";
                    %5 : tensor<x32, x1, int> = tt.expand_dims %4 @"1";
                    %6 : tensor<x1, x64, int> = tt.splat %1;
                    %7 : tensor<x1, x64, int> = arith.addi %3 %6;
                    tt.consume %7;
                    %8 : tensor<x32, x64, int> = tt.broadcast %3;
                    %9 : tensor<x32, x64, int> = tt.broadcast %5;
                    %10 : tensor<x32, x64, int> = arith.addi %8 %9;
                    tt.consume %10;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void test2(int a, @Constant int M, @Constant int N) {
        var m = arange(0, M);
        var me = expand(m, 0);

        var n = arange(0, N);
        var ne = expand(n, 1);

        var t4 = add(me, a);
        consume(t4);

        var t3 = add(me, ne);
        consume(t3);
    }

    @Test
    public void test2(TritonTestData t) {
        List<TypeElement> argTypes = List.of(
                JavaType.INT,
                new ConstantType(JavaType.INT, 64),
                new ConstantType(JavaType.INT, 32)
        );

        t.test(argTypes);
    }
}