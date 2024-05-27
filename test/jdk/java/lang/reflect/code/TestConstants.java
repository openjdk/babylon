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
 * @run testng TestConstants
 */

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class TestConstants {

    @CodeReflection
    public static boolean c_boolean() {
        return true;
    }

    @CodeReflection
    public static boolean c_boolean_f() {
        return false;
    }

    @CodeReflection
    public static byte c_byte() {
        return 42;
    }

    @CodeReflection
    public static byte c_byte_neg() {
        return -42;
    }

    @CodeReflection
    public static short c_short() {
        return 42;
    }

    @CodeReflection
    public static short c_short_neg() {
        return -42;
    }

    @CodeReflection
    public static char c_char() {
        return 'A';
    }

    @CodeReflection
    public static int c_int() {
        return 42;
    }

    @CodeReflection
    public static int c_int_neg() {
        return -42;
    }

    @CodeReflection
    public static long c_long() {
        return 42L;
    }

    @CodeReflection
    public static long c_long_neg() {
        return -42L;
    }

    @CodeReflection
    public static float c_float() {
        return 1.0f;
    }

    @CodeReflection
    public static float c_float_neg() {
        return -1.0f;
    }

    @CodeReflection
    public static double c_double() {
        return 1.0;
    }

    @CodeReflection
    public static double c_double_neg() {
        return -1.0;
    }

    @CodeReflection
    public static String c_String() {
        String s = "42";
        s = null;
        return s;
    }

    @CodeReflection
    public static Class<?> c_Class() {
        return String.class;
    }

    @CodeReflection
    public static Class<?> c_Class_primitive() {
        return float.class;
    }

    @DataProvider
    static Object[][] provider() {
        return new Object[][] {
                { boolean.class },
                { byte.class },
                { short.class },
                { char.class },
                { int.class },
                { long.class },
                { float.class },
                { double.class },
                { String.class },
                { Class.class }
        };
    }

    @Test(dataProvider = "provider")
    public void testString(Class<?> c) throws Exception {
        String name = "c_" + c.getSimpleName();
        List<Method> ms = Stream.of(TestConstants.class.getDeclaredMethods())
                .filter(m -> m.getName().startsWith(name))
                .toList();

        for (Method m : ms) {
            CoreOp.FuncOp f = m.getCodeModel().get();

            f.writeTo(System.out);

            Assert.assertEquals(Interpreter.invoke(f), m.invoke(null));
        }
    }

    @CodeReflection
    public static String compareNull(String s) {
        if (s == null) {
            return "NULL";
        } else {
            return s;
        }
    }

    @Test
    public void testCompareNull() {
        CoreOp.FuncOp f = getFuncOp("compareNull");

        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        lf.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(lf, (Object) null), compareNull(null));
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestConstants.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
