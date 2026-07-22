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

import jdk.incubator.code.Op;

import java.lang.reflect.Method;
import jdk.incubator.code.Reflect;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Optional;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Method::getCodeModel returns the same instance.
 * @modules jdk.incubator.code
 * @run junit CodeModelSameInstanceTest
 */
public class CodeModelSameInstanceTest {

    @Reflect
    static int add(int a, int b) {
        return a + b;
    }

    @Test
    public void test() {
        Optional<Method> om = Arrays.stream(this.getClass().getDeclaredMethods()).filter(m -> m.getName().equals("add"))
                .findFirst();
        Method m = om.get();
        Object[] codeModels = IntStream.range(0, 1024).mapToObj(_ -> Op.ofMethod(m)).toArray();
        for (int i = 1; i < codeModels.length; i++) {
            Assertions.assertSame(codeModels[i-1], codeModels[i]);
        }
    }
}
