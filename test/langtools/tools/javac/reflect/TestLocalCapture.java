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
 * @summary Smoke test for captured values in local classes.
 * @run testng TestLocalCapture
 */

import org.testng.annotations.*;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp.FuncOp;
import java.lang.runtime.CodeReflection;
import java.util.stream.IntStream;

import static org.testng.Assert.*;

public class TestLocalCapture {

    int x = 13;
    final int CONST = 1;

    @CodeReflection
    public int sum(int y, int z) {
        final int localConst = 2;
        class Foo {
            int sum(int z) { return localConst + x + y + z + CONST; };
        }
        return new Foo().sum(z);
    }

    @Test(dataProvider = "ints")
    public void testLocalCapture(int y) throws ReflectiveOperationException {
        Method sum = TestLocalCapture.class.getDeclaredMethod("sum", int.class, int.class);
        FuncOp model = sum.getCodeModel().get();
        int found = (int)Interpreter.invoke(MethodHandles.lookup(), model, this, y, 17);
        int expected = sum(y, 17);
        assertEquals(found, expected);
    }

    @DataProvider(name = "ints")
    public Object[][] ints() {
        return IntStream.range(0, 50)
                .mapToObj(i -> new Object[] { i })
                .toArray(Object[][]::new);
    }
}
