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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.runtime.CodeReflection;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/*
 * @test
 * @run testng TestThrowing
 */

public class TestThrowing {

    @Test(dataProvider = "methods-exceptions")
    public void testThrowsCorrectException(String methodName, Class<? extends Throwable> expectedExceptionType) throws NoSuchMethodException {
        Method method = TestThrowing.class.getDeclaredMethod(methodName);
        Assert.assertThrows(expectedExceptionType, () -> Interpreter.invoke(MethodHandles.lookup(), method.getCodeModel().orElseThrow()));
    }

    @DataProvider(name = "methods-exceptions")
    static Object[][] testData() throws NoSuchMethodException {
        return new Object[][]{
                {"throwsError", TestError.class},
                {"throwsRuntimeException", TestRuntimeException.class},
                {"throwsCheckedException", TestCheckedException.class},
        };
    }

    public static class TestError extends Error {

    }

    public static class TestRuntimeException extends RuntimeException {

    }

    public static class TestCheckedException extends Exception {

    }

    @CodeReflection
    static void throwsError() {
        throw new TestError();
    }

    @CodeReflection
    static void throwsRuntimeException() {
        throw new TestRuntimeException();
    }

    @CodeReflection
    static void throwsCheckedException() throws TestCheckedException {
        throw new TestCheckedException();
    }
}
