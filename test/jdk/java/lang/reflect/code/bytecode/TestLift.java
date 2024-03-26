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
import org.testng.annotations.BeforeClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestLift
 */

public class TestLift {

    @CodeReflection
    static int sum(int i, int j) {
        i = i + j;
        return i;
    }

    @CodeReflection
    static Class<?> objectArray(int i, int j) {
        Class<?>[] ifaces = new Class[1 + i + j];
        ifaces[0] = Function.class;
        return ifaces[0];
    }

    @CodeReflection
    static String constructor(int i, int j) {
        return new String("hello world".getBytes(), i, j);
    }

    @CodeReflection
    static int logicalOps(int i, int j) {
        return 13 & i | j ^ 13;
    }

    @DataProvider(name = "testMethods")
    public static Object[] testMethods() {
        return Stream.of(TestLift.class.getDeclaredMethods()).filter(m -> m.isAnnotationPresent(CodeReflection.class)).map(Method::getName).toArray();
    }

    private static byte[] CLASS_DATA;

    @BeforeClass
    public void setup() throws Exception {
        CLASS_DATA = TestLift.class.getResourceAsStream("TestLift.class").readAllBytes();
    }

    @Test(dataProvider = "testMethods")
    public void testLift(String methodName) throws Exception {
        Method testMethod = TestLift.class.getDeclaredMethod(methodName, int.class, int.class);
        CoreOps.FuncOp flift = null;
        try {
            flift = BytecodeLift.lift(CLASS_DATA, methodName);
        } catch (Exception e) {
            System.out.println("Lift failed, expected:");
            testMethod.getCodeModel().ifPresent(f -> f.writeTo(System.out));
            throw e;
        }
        try {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    Assert.assertEquals(Interpreter.invoke(flift, i, j), testMethod.invoke(null, i, j));
                }
            }
        } catch (Exception e) {
            flift.writeTo(System.out);
            throw e;
        }
    }
}
