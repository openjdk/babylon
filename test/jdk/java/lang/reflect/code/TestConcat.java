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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.runtime.CodeReflection;
import java.util.*;
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


    record Triple(Class<?> first, Class<?> second, String third) {
    }

    static final Map<Class<?>, Object> valMap;
    static {
        valMap = new HashMap<>();
        valMap.put(byte.class, (byte) 42);
        valMap.put(short.class, (short) 42);
        valMap.put(int.class, 42);
        valMap.put(long.class, (long) 42);
        valMap.put(float.class, 42f);
        valMap.put(double.class, 42d);
        valMap.put(char.class, 'z');
        valMap.put(boolean.class, false);
        valMap.put(Object.class, new Object() {
                    @Override
                    public String toString() {
                        return "I'm a test string.";
                    }
                });
        valMap.put(TestObject.class, new TestObject());
        valMap.put(String.class, TESTSTR);
    }
    private static String testName(Class<?> n, Integer i){
        return n.getSimpleName().toLowerCase() + "Concat" + i;
    }
    @DataProvider(name = "testData")
    public static Object[][]  testData() {
        Set<Class<?>> types = Set.of(byte.class, short.class, int.class, long.class, float.class,
                double.class, char.class, boolean.class, Object.class);



        //Types from types concatenated to strings left-to-right and right-to-left
        Stream<Triple> s1 = types.stream().map(t -> new Triple(t,String.class, testName(t, 1)));
        Stream<Triple> s2 = types.stream().map(t -> new Triple(String.class, t, testName(t, 2)));

        //Custom Object and basic string concat tests
        Stream<Triple> s3 = Stream.of(new Triple(TestObject.class, String.class, testName(Object.class, 3)),
                                      new Triple(String.class, TestObject.class, testName(Object.class, 4)),
                                      new Triple(String.class, String.class, "stringConcat"));

        Object[] t = Stream.concat(Stream.concat(s1,s2),s3).toArray();

        Object[][] args = new Object[t.length][];

        for(int i = 0; i < args.length; i++) {
            args[i] = new Object[]{ t[i] };
        }

        return args;

    }

    @Test(dataProvider = "testData")
    public static void testRun(Triple t) {
        try {

            Object[] args = new Object[] {valMap.get(t.first), valMap.get(t.second)};
            Class<TestConcat> clazz = TestConcat.class;
            Method method = clazz.getDeclaredMethod(t.third, t.first, t.second);
            CoreOps.FuncOp f = method.getCodeModel().orElseThrow();
            var res1 = Interpreter.invoke(MethodHandles.lookup(), f, args);
            var res2 = method.invoke(null, args);

            Assert.assertEquals(res1, res2);

        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            throw new RuntimeException(e);
        }
    }

    public static final class TestObject {
        TestObject(){}

        @Override
        public String toString() {
            return "TestObject String";
        }
    }
}
