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

import org.testng.Assert;
import org.testng.annotations.Test;
import org.testng.annotations.DataProvider;

import java.io.IOException;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.runtime.CodeReflection;
import java.util.List;

/*
 * @test
 * @run testng TestThrowing
 */

public class TestThrowing {

    @Test(dataProvider = "methods-exceptions")
    public void testThrowsCorrectException(String methodName, Class<?> expectedExceptionType) throws NoSuchMethodException {
        Method method = TestThrowing.class.getDeclaredMethod(methodName);
        try {
            Interpreter.invoke(method.getCodeModel().orElseThrow(), List.of());
            Assert.fail("invoke should throw");
        } catch (Throwable throwable) {
            Assert.assertEquals(throwable.getClass(), expectedExceptionType);
        }
    }

    @DataProvider(name = "methods-exceptions")
    static Object[][] testData() throws NoSuchMethodException {
        return new Object[][]{
                {"throwsError", Error.class},
                {"throwsRuntimeException", RuntimeException.class},
                {"throwsCheckedException", IOException.class},
        };
    }

    @CodeReflection
    static void throwsError() {
        throw new Error();
    }

    @CodeReflection
    static void throwsRuntimeException() {
        throw new RuntimeException();
    }

    @CodeReflection
    static void throwsCheckedException() throws IOException {
        throw new IOException();
    }
}
