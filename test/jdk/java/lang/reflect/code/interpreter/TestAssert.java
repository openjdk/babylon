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
import java.lang.reflect.code.Op;
import java.util.List;
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

    public static final String FAILURESTRING = "failure";
    public static final char FAILURECHAR = 'o';

    public static final float FAILUREFLOAT = -1.0f;
    public static final double FAILUREDOUBLE = -1.0d;
    public static final byte FAILUREBYTE = -1;
    public static final short FAILURESHORT = -1;
    public static final int FAILUREINT = -1;

    public static final long FAILURELONG = -1;

    public static final String FAILUREOBJECTMSG = "FAILURE OBJECT";

    public static final Object FAILUREOBJECT = new FailureObject();


    @Test
    public void testAssertThrows(){
        testThrows("assertThrow");
    }

    @Test
    public void testAssertString(){
        AssertionError ae = testThrows("assertThrowWithMessage");
        if (ae.getMessage() == null || !ae.getMessage().equals(FAILURESTRING)) {
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertChar() {
        AssertionError ae = testThrows("assertChar");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILURECHAR))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertFloat() {
        AssertionError ae = testThrows("assertFloat");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREFLOAT))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertDouble() {
        AssertionError ae = testThrows("assertDouble");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREDOUBLE))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertByte() {
        AssertionError ae = testThrows("assertByte");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREBYTE))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertShort() {
        AssertionError ae = testThrows("assertShort");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILURESHORT))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertInt() {
        AssertionError ae = testThrows("assertInt");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREINT))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertLong() {
        AssertionError ae = testThrows("assertLong");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILURELONG))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertObject() {
        AssertionError ae = testThrows("assertObject");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREOBJECT))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertExpr1() {
        AssertionError ae = testThrows("assertExpr1");
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREINT + FAILURELONG))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertExpr2() {
        AssertionError ae = testThrows("assertExpr2", List.of(int.class), 52);
        if (ae.getMessage() == null || !ae.getMessage().equals(String.valueOf(FAILUREINT))){
            Assert.fail("Assertion failure messages do not match.");
        }
    }

    @Test
    public void testAssertExpr3() {
         testRun("assertExpr3", List.of(int.class), 52);
    }

    private static AssertionError testThrows(String methodName) {
        return testThrows(methodName, List.of());
    }
    private static void testRun(String methodName, List<Class<?>> params, Object...args) {
        try {
            Class<TestAssert> clazz = TestAssert.class;
            Method method = clazz.getDeclaredMethod(methodName,params.toArray(new Class[params.size()]));
            CoreOps.FuncOp f = method.getCodeModel().orElseThrow();

            //Ensure we're fully lowered before testing.
            final var fz = f.transform((b, o) -> {
                if (o instanceof Op.Lowerable l) {
                    b = l.lower(b);
                } else {
                    b.op(o);
                }
                return b;
            });

            Interpreter.invoke(MethodHandles.lookup(), fz ,args);
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    private static AssertionError testThrows(String methodName, List<Class<?>> params, Object...args) {
        try {
            Class<TestAssert> clazz = TestAssert.class;
            Method method = clazz.getDeclaredMethod(methodName,params.toArray(new Class[params.size()]));
            CoreOps.FuncOp f = method.getCodeModel().orElseThrow();

            //Ensure we're fully lowered before testing.
            final var fz = f.transform((b, o) -> {
                if (o instanceof Op.Lowerable l) {
                    b = l.lower(b);
                } else {
                    b.op(o);
                }
                return b;
            });


            AssertionError ae = (AssertionError) retCatch(() -> Interpreter.invoke(MethodHandles.lookup(), fz ,args));
            Assert.assertNotNull(ae);
            return ae;
        } catch (NoSuchMethodException e) {
            throw new RuntimeException(e);
        }
    }

    private static Throwable retCatch(Runnable r) {
        try {
            r.run();
        } catch (Throwable t) {
            return t;
        }
        return null;
    }



    @CodeReflection
    public static void assertThrow() {
        assert false;
    }

    @CodeReflection
    public static void assertThrowWithMessage() {
        assert false : FAILURESTRING;
    }

    @CodeReflection
    public static void assertChar() {
        char c = FAILURECHAR;
        assert false : c;
    }

    @CodeReflection
    public static void assertFloat() {
        float f = FAILUREFLOAT;
        assert false : f;
    }

    @CodeReflection
    public static void assertDouble() {
        double d = FAILUREDOUBLE;
        assert false : d;
    }

    @CodeReflection
    public static void assertByte() {
        byte b = FAILUREBYTE;
        assert false : b;
    }

    @CodeReflection
    public static void assertShort() {
        short s = FAILURESHORT;
        assert false : s;
    }

    @CodeReflection
    public static void assertInt() {
        int i = FAILUREINT;
        assert false : i;
    }

    @CodeReflection
    public static void assertLong() {
        long l = FAILURELONG;
        assert false : l;
    }

    @CodeReflection
    public static void assertObject() {
        Object o = FAILUREOBJECT;
        assert false : o;
    }

    @CodeReflection
    public static void assertExpr1() {
        int i = FAILUREINT;
        long l = FAILURELONG;
        assert false : i + l;
        String y = "test";
    }

    @CodeReflection
    public static void assertExpr2(int iz) {
        int i = FAILUREINT;
        long l = FAILURELONG;
        assert false : (i > iz) ? i + l : i;
        String s = "";
    }

    @CodeReflection
    public static void assertExpr3(int iz) {
        int i = FAILUREINT;
        long l = FAILURELONG;
        assert true : (i > iz) ? i + l : i;
        String s = "";
    }

    static class FailureObject {
        @Override
        public String toString(){
           return FAILUREOBJECTMSG;
        }
    }
}
