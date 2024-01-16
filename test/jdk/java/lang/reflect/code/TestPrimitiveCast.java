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
 * @run testng TestPrimitiveCast
 */

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

import static java.util.stream.Collectors.joining;

public class TestPrimitiveCast {

    static final Function<Object, String> FROM_DOUBLE = v -> fromDouble((double) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromDouble(double v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_FLOAT = v -> fromFloat((float) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromFloat(float v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_LONG = v -> fromLong((long) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromLong(long v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_INT = v -> fromInt((int) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromInt(int v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_SHORT = v -> fromShort((short) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromShort(short v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_CHAR = v -> fromChar((char) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromChar(char v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    static final Function<Object, String> FROM_BYTE = v -> fromByte((byte) v);

    @CodeReflection
    @SuppressWarnings("cast")
    static String fromByte(byte v) {
        double d = (double) v;
        float f = (float) v;
        long l = (long) v;
        int i = (int) v;
        short s = (short) v;
        char c = (char) v;
        byte b = (byte) v;
        return collect(d, f, l, i, s, c, b);
    }

    @DataProvider
    static Object[][] fromMethods() {
        return new Object[][] {
                { "fromDouble", Math.PI, FROM_DOUBLE},
                { "fromDouble", 65.1, FROM_DOUBLE},
                { "fromFloat", (float) Math.PI, FROM_FLOAT},
                { "fromFloat", 65.1f, FROM_FLOAT},
                { "fromLong", Long.MAX_VALUE, FROM_LONG},
                { "fromInt", Integer.MAX_VALUE, FROM_INT},
                { "fromShort", Short.MAX_VALUE, FROM_SHORT},
                { "fromChar", Character.MAX_VALUE, FROM_CHAR},
                { "fromByte", Byte.MAX_VALUE, FROM_BYTE},
        };
    };

    @Test(dataProvider = "fromMethods")
    public void testFromDouble(String name, Object value, Function<Object, String> m) {
        CoreOps.FuncOp f = getFuncOp(name);
        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), f, value), m.apply(value));
    }


    static String collect(Object... values) {
        return Stream.of(values).map(Object::toString).collect(joining(" "));
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestPrimitiveCast.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
