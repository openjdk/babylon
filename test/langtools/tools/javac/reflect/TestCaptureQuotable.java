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
 * @summary Smoke test for captured values in quotable lambdas.
 * @modules jdk.incubator.code
 * @run junit TestCaptureQuotable
 */

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import jdk.incubator.code.dialect.core.CoreOp.Var;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.function.ToIntFunction;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestCaptureQuotable {

    @ParameterizedTest
    @MethodSource("ints")
    public void testCaptureIntParam(int x) {
        Quotable quotable = (Quotable & IntUnaryOperator)y -> x + y;
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(1, quoted.capturedValues().size());
        assertEquals(x, ((Var)quoted.capturedValues().values().iterator().next()).value());
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(x + 1, res);
    }

    @Test
    public void testCaptureThisRefAndIntConstant() {
        final int x = 100;
        String hello = "hello";
        Quotable quotable = (Quotable & ToIntFunction<Number>)y -> y.intValue() + hashCode() + hello.length() + x;
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(3, quoted.capturedValues().size());
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(this, it.next());
        assertEquals(hello, ((Var)it.next()).value());
        assertEquals(x, ((Var)it.next()).value());
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(x + 1 + hashCode() + hello.length(), res);
    }

    @Test
    public void testCaptureThisInInvocationArg() {
        Quotable quotable = (Quotable & ToIntFunction<Number>)y -> y.intValue() + Integer.valueOf(hashCode());
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(1, quoted.capturedValues().size());
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(this, it.next());
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(1 + hashCode(), res);
    }

    record R(int i) {}

    @Test
    public void testCaptureThisInNewArg() {
        Quotable quotable = (Quotable & ToIntFunction<Number>)y -> y.intValue() + new R(hashCode()).i;
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(1, quoted.capturedValues().size());
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        assertEquals(this, it.next());
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(1 + hashCode(), res);
    }

    @Test
    public void testCaptureMany() {
        int[] ia = new int[8];
        int i1 = ia[0] = 0;
        int i2 = ia[1] = 1;
        int i3 = ia[2] = 2;
        int i4 = ia[3] = 3;
        int i5 = ia[4] = 4;
        int i6 = ia[5] = 5;
        int i7 = ia[6] = 6;
        int i8 = ia[7] = 7;

        Quotable quotable = (Quotable & IntSupplier) () -> i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8;
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(ia.length, quoted.capturedValues().size());
        assertEquals(new ArrayList<>(quoted.capturedValues().keySet()), quoted.op().capturedValues());
        Iterator<Object> it = quoted.capturedValues().values().iterator();
        int i = 0;
        while (it.hasNext()) {
            int actual = (int) ((Var)it.next()).value();
            assertEquals(ia[i++], actual);
        }
    }

    static class Context {
        final int x;

        Context(int x) {
            this.x = x;
        }

        Quotable quotable() {
            return (Quotable & IntUnaryOperator) y -> x + y;
        }
    }

    @ParameterizedTest
    @MethodSource("ints")
    public void testCaptureIntField(int x) {
        Context context = new Context(x);
        Quotable quotable = context.quotable();
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(1, quoted.capturedValues().size());
        assertEquals(context, quoted.capturedValues().values().iterator().next());
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(x + 1, res);
    }

    public static IntStream ints() {
        return IntStream.range(0, 50);
    }

    @ParameterizedTest
    @MethodSource("ints")
    public void testCaptureReferenceReceiver(int i) {
        int prevCount = Box.count;
        Quotable quotable = (Quotable & IntUnaryOperator)new Box(i)::add;
        Quoted quoted = Op.ofQuotable(quotable).get();
        assertEquals(prevCount + 1, Box.count); // no duplicate receiver computation!
        assertEquals(1, quoted.capturedValues().size());
        assertEquals(i, ((Box)((Var)quoted.capturedValues().values().iterator().next()).value()).i);
        List<Object> arguments = new ArrayList<>();
        arguments.add(1);
        arguments.addAll(quoted.capturedValues().values());
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                arguments);
        assertEquals(i + 1, res);
    }

    record Box(int i) {

        static int count = 0;

        Box {
           count++; // keep track of side-effects
        }

        int add(int i) {
            return i + this.i;
        }
    }
}
