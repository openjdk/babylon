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

import java.lang.reflect.Type;
import java.lang.runtime.CodeReflection;
import java.util.List;

import static oracle.code.triton.Triton.*;
import static oracle.code.triton.TritonTest.consume;

@ExtendWith(TritonTestExtension.class)
public class TestCountedLoop {

    @TritonCodeModel(value = """
            module ()void -> {
                tt.func @"test1_int_64_void" (%0 : int)void -> {
                    %1 : int = arith.constant @"64";
                    %2 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %3 : tensor<x64, int> = tt.make_range @start="0" @end="64";
                    %4 : int = arith.constant @"0";
                    %5 : int = arith.constant @"1";
                    %6 : java.lang.reflect.code.CoreOps$Tuple<tensor<x64, int>, tensor<x64, int>> = scf.for %4 %0 %5 %2 %3 (%7 : int, %8 : tensor<x64, int>, %9 : tensor<x64, int>)java.lang.reflect.code.CoreOps$Tuple<tensor<x64, int>, tensor<x64, int>> -> {
                        %10 : tensor<x64, int> = tt.splat %7;
                        %11 : tensor<x64, int> = arith.addi %8 %10;
                        %12 : tensor<x64, int> = tt.splat %1;
                        %13 : tensor<x64, int> = arith.addi %9 %12;
                        scf.yield %11 %13;
                    };
                    %14 : tensor<x64, int> = tuple.load %6 @"0";
                    %15 : tensor<x64, int> = tuple.load %6 @"1";
                    tt.consume %14;
                    tt.consume %15;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void test1(int n, @Constant int s) {
        var a = arange(0, s);
        var b = arange(0, s);
        for (int i = 0; i < n; i++) {
            a = Triton.add(a, i);
            b = Triton.add(b, s);
        }
        consume(a);
        consume(b);
    }

    @Test
    public void test1(TritonTestData t) {
        List<Type> argTypes = List.of(
                int.class,
                new ConstantType(int.class, 64));

        t.test(argTypes);
    }
}