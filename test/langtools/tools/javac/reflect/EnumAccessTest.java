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

import java.lang.constant.DirectMethodHandleDesc;
import jdk.incubator.code.CodeReflection;

import static java.lang.constant.DirectMethodHandleDesc.Kind;
import static java.lang.constant.DirectMethodHandleDesc.Kind.VIRTUAL;

/*
 * @test
 * @summary Smoke test for code reflection with enum access.
 * @modules jdk.incubator.code
 * @build EnumAccessTest
 * @build CodeReflectionTester
 * @run main CodeReflectionTester EnumAccessTest
 */

public class EnumAccessTest {
    @CodeReflection
    @IR("""
            func @"test1" (%0 : EnumAccessTest)java.lang.constant.DirectMethodHandleDesc$Kind -> {
                %1 : java.lang.constant.DirectMethodHandleDesc$Kind = field.load @"java.lang.constant.DirectMethodHandleDesc$Kind::VIRTUAL()java.lang.constant.DirectMethodHandleDesc$Kind";
                return %1;
            };
            """)
    DirectMethodHandleDesc.Kind test1() {
        return DirectMethodHandleDesc.Kind.VIRTUAL;
    }

    @CodeReflection
    @IR("""
            func @"test2" (%0 : EnumAccessTest)java.lang.constant.DirectMethodHandleDesc$Kind -> {
                %1 : java.lang.constant.DirectMethodHandleDesc$Kind = field.load @"java.lang.constant.DirectMethodHandleDesc$Kind::VIRTUAL()java.lang.constant.DirectMethodHandleDesc$Kind";
                return %1;
            };
            """)
    DirectMethodHandleDesc.Kind test2() {
        return Kind.VIRTUAL;
    }

    @CodeReflection
    @IR("""
            func @"test3" (%0 : EnumAccessTest)java.lang.constant.DirectMethodHandleDesc$Kind -> {
                %1 : java.lang.constant.DirectMethodHandleDesc$Kind = field.load @"java.lang.constant.DirectMethodHandleDesc$Kind::VIRTUAL()java.lang.constant.DirectMethodHandleDesc$Kind";
                return %1;
            };
            """)
    DirectMethodHandleDesc.Kind test3() {
        return VIRTUAL;
    }
}