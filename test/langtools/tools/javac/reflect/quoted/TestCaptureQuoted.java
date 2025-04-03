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
 * @summary Smoke test for captured values in quoted lambdas.
 * @modules jdk.incubator.code
 * @run testng TestCaptureQuoted
 */

import jdk.incubator.code.Quotable;
import jdk.incubator.code.op.CoreOp.Var;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;

import org.testng.annotations.*;
import static org.testng.Assert.*;

public class TestCaptureQuoted {

    @Test(dataProvider = "ints")
    public void testCaptureIntParam(int x) {
        Quoted quoted = (int y) -> x + y;
        assertEquals(quoted.capturedValues().size(), 1);
        assertEquals(((Var)quoted.capturedValues().values().iterator().next()).value(), x);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(res, x + 1);
    }

    @Test(dataProvider = "ints")
    public void testCaptureIntField(int x) {
        class Context {
            final int x;

            Context(int x) {
                this.x = x;
            }

            Quoted quoted() {
                return (int y) -> x + y;
            }
        }
        Context context = new Context(x);
        Quoted quoted = context.quoted();
        assertEquals(quoted.capturedValues().size(), 1);
        assertEquals(quoted.capturedValues().values().iterator().next(), context);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(res, x + 1);
    }

    @Test
    public void testCaptureThisRefAndIntConstant() {
        final int x = 100;
        String hello = "hello";
        Quoted quoted = (Integer y) -> y.intValue() + hashCode() + hello.length() + x;
        assertEquals(quoted.capturedValues().size(), 3);
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(it.next(), this);
        assertEquals(((Var)it.next()).value(), hello);
        assertEquals(((Var)it.next()).value(), x);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(res, x + 1 + hashCode() + hello.length());
    }

    @Test
    public void testCaptureThisInInvocationArg() {
        Quoted quoted = (Number y) -> y.intValue() + Integer.valueOf(hashCode());
        assertEquals(quoted.capturedValues().size(), 1);
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(it.next(), this);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(res, 1 + hashCode());
    }

    record R(int i) {}

    @Test
    public void testCaptureThisInNewArg() {
        Quoted quoted = (Number y) -> y.intValue() + new R(hashCode()).i;
        assertEquals(quoted.capturedValues().size(), 1);
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(it.next(), this);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(res, 1 + hashCode());
    }


    @DataProvider(name = "ints")
    public Object[][] ints() {
        return IntStream.range(0, 50)
                .mapToObj(i -> new Object[] { i })
                .toArray(Object[][]::new);
    }
}
