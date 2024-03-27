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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestConcat
 */

public class TestConcat {

    static final String TESTSTR = "TESTING STRING";

    static final int TESTINT = 42;
    @CodeReflection
    static String f() {
       int test = 1;
       //String res = TESTINT + TESTSTR + test;
       String res = "HI " + TESTSTR + test;
       return res;
    }


    @CodeReflection
    public static String byteConcat1(byte b, String s) {
       return b + s;
    }

    @CodeReflection
    public static String byteConcat2(String s, byte b) {
        return s + b;
    }

    @CodeReflection
    public static String shortConcat1(short b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String shortConcat2(String s, short b) {
        return s + b;
    }

    @CodeReflection
    public static String intConcat1(int b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String intConcat2(String s, int b) {
        return s + b;
    }

    @CodeReflection
    public static String longConcat1(long b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String longConcat2(String s, long b) {
        return s + b;
    }

    @CodeReflection
    public static String floatConcat1(float b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String floatConcat2(String s, float b) {
        return s + b;
    }

    @CodeReflection
    public static String doubleConcat1(double b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String doubleConcat2(String s, double b) {
        return s + b;
    }

    @CodeReflection
    public static String booleanConcat1(boolean b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String booleanConcat2(String s, boolean b) {
        return s + b;
    }

    @CodeReflection
    public static String charConcat1(char b, String s) {
        return b + s;
    }
    @CodeReflection
    public static String charConcat2(String s, char b) {
        return s + b;
    }
    @CodeReflection
    public static String objectConcat1(Object b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String objectConcat2(String s, Object b) {
        return s + b;
    }

    @CodeReflection
    public static String objectConcat3(TestObject b, String s) {
        return b + s;
    }

    @CodeReflection
    public static String objectConcat4(String s, TestObject b) {
        return s + b;
    }

    @CodeReflection
    public static String stringConcat(String b, String s) {
        return b + s;
    }



    @Test
    public void testInterpretConcat() {
        //CoreOps.FuncOp fModel = getFuncOp("f");
        var res1 = testRun("f",List.of());
        var res2 = f();
        Assert.assertEquals(res1,res2);
    }

    @Test
    public static void testByteConcat() {
       byte b = 42;
       Object res1 = testRun("byteConcat1", List.of(byte.class, String.class), b, TESTSTR);
       Assert.assertEquals(res1, byteConcat1(b, TESTSTR));

       var res2 = testRun("byteConcat2", List.of(String.class, byte.class), TESTSTR, b);
       Assert.assertEquals(res2, byteConcat2(TESTSTR,b));
    }

    @Test
    public static void testShortConcat() {
        short b = 42;
        Object res1 = testRun("shortConcat1", List.of(short.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, shortConcat1(b, TESTSTR));

        var res2 = testRun("shortConcat2", List.of(String.class, short.class), TESTSTR, b);
        Assert.assertEquals(res2, shortConcat2(TESTSTR,b));
    }

    @Test
    public static void testIntConcat() {
        int b = 42;
        Object res1 = testRun("intConcat1", List.of(int.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, intConcat1(b, TESTSTR));

        var res2 = testRun("intConcat2", List.of(String.class, int.class), TESTSTR, b);
        Assert.assertEquals(res2, intConcat2(TESTSTR,b));
    }

    @Test
    public static void testLongConcat() {
        long b = 42;
        Object res1 = testRun("longConcat1", List.of(long.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, longConcat1(b, TESTSTR));

        var res2 = testRun("longConcat2", List.of(String.class, long.class), TESTSTR, b);
        Assert.assertEquals(res2, longConcat2(TESTSTR,b));
    }

    @Test
    public static void testFloatConcat() {
        float b = 42.0f;
        Object res1 = testRun("floatConcat1", List.of(float.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, floatConcat1(b, TESTSTR));

        var res2 = testRun("floatConcat2", List.of(String.class, float.class), TESTSTR, b);
        Assert.assertEquals(res2, floatConcat2(TESTSTR,b));
    }

    @Test
    public static void testDoubleConcat() {
        double b = 42.0f;
        Object res1 = testRun("doubleConcat1", List.of(double.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, doubleConcat1(b, TESTSTR));

        var res2 = testRun("doubleConcat2", List.of(String.class, double.class), TESTSTR, b);
        Assert.assertEquals(res2, doubleConcat2(TESTSTR,b));
    }

    @Test
    public static void testBooleanConcat() {
        boolean b = false;
        Object res1 = testRun("booleanConcat1", List.of(boolean.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, booleanConcat1(b, TESTSTR));

        var res2 = testRun("booleanConcat2", List.of(String.class, boolean.class), TESTSTR, b);
        Assert.assertEquals(res2, booleanConcat2(TESTSTR,b));
    }
    @Test
    public static void testCharConcat() {
        char b = 'z';
        Object res1 = testRun("charConcat1", List.of(char.class, String.class), b, TESTSTR);
        Assert.assertEquals(res1, charConcat1(b, TESTSTR));

        var res2 = testRun("charConcat2", List.of(String.class, char.class), TESTSTR, b);
        Assert.assertEquals(res2, charConcat2(TESTSTR,b));
    }

    @Test
    public static void testObjectConcat() {

        Object o = new Object() {
            @Override
            public String toString() {
                return "I'm a test string.";
            }
        };

        Object res1 = testRun("objectConcat1", List.of(Object.class, String.class), o, TESTSTR);
        Assert.assertEquals(res1, objectConcat1(o, TESTSTR));

        var res2 = testRun("objectConcat2", List.of(String.class, Object.class), TESTSTR, o);
        Assert.assertEquals(res2, objectConcat2(TESTSTR,o));
    }

    @Test
    public static void testObjectConcat2() {

        TestObject o = new TestObject();

        Object res1 = testRun("objectConcat3", List.of(TestObject.class, String.class), o, TESTSTR);
        Assert.assertEquals(res1, objectConcat3(o, TESTSTR));

        var res2 = testRun("objectConcat4", List.of(String.class, TestObject.class), TESTSTR, o);
        Assert.assertEquals(res2, objectConcat4(TESTSTR,o));
    }

    @Test
    public static void testStringConcat() {
        String s = "teststring.";

        Object res1 = testRun("stringConcat", List.of(String.class, String.class), s, TESTSTR);
        Assert.assertEquals(res1, stringConcat(s, TESTSTR));

    }

    private static Object testRun(String methodName, List<Class<?>> params, Object...args) {
        try {

            Class<TestConcat> clazz = TestConcat.class;
            Method method = clazz.getDeclaredMethod(methodName,params.toArray(new Class[params.size()]));
            CoreOps.FuncOp f = method.getCodeModel().orElseThrow();
            return Interpreter.invoke(MethodHandles.lookup(), f ,args);

        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }


    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestConcat.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

    public static final class TestObject {
        TestObject(){}

        @Override
        public String toString() {
           return "TestObject String";
        }
    }
}
