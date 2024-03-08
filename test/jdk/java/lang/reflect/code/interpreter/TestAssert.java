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
import java.util.Optional;
import java.util.stream.Stream;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/*
 * @test
 * @run testng/othervm -ea TestAssert
 */
public class TestAssert {

    @Test
    public void testAssertThrows(){
        try {
            Class<TestAssert> clazz = TestAssert.class;
            Method method = clazz.getDeclaredMethod("assertThrow");
            CoreOps.FuncOp f = method.getCodeModel().orElseThrow();
            //Interpreter.invoke(MethodHandles.lookup(), f);
            Assert.assertThrows(AssertionError.class, () -> Interpreter.invoke(MethodHandles.lookup(), method.getCodeModel().orElseThrow()));
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
        return;
    }

    @CodeReflection
    public static void assertThrow() {
        assert false;
    }
}
