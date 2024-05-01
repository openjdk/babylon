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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.JavaType;
import java.lang.runtime.CodeReflection;
import java.util.List;

import static oracle.code.triton.Triton.*;
import static oracle.code.triton.TritonTest.consume;

@ExtendWith(TritonTestExtension.class)
public class TestZeros {

    @TritonCodeModel("""
            module ()void -> {
                tt.func @"test1_32_64_void" ()void -> {
                    %0 : tensor<x32, x64, float> = arith.constant @"0.0";
                    tt.consume %0;
                    tt.return;
                };
                unreachable;
            };
            """)
    @CodeReflection
    static void test1(@Constant int M, @Constant int N) {
        var t = zeros(float.class, M, N);
        consume(t);
    }

    @Test
    public void test1(TritonTestExtension.TritonTestData t) {
        List<CodeType> argTypes = List.of(
                new ConstantType(JavaType.INT, 32),
                new ConstantType(JavaType.INT, 64));

        t.test(argTypes);
    }

}
