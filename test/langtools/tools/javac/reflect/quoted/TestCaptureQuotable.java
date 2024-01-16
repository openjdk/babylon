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
 * @run testng TestCaptureQuotable
 */

import org.testng.annotations.*;

import java.lang.reflect.code.op.CoreOps.Var;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

import static org.testng.Assert.*;

public class TestCaptureQuotable {

    @Test(dataProvider = "ints")
    public void testCaptureIntParam(int x) {
        Quotable quotable = (Quotable & IntUnaryOperator)y -> x + y;
        Quoted quoted = quotable.quoted();
        assertEquals(quoted.capturedValues().size(), 1);
        assertEquals(((Var)quoted.capturedValues().values().iterator().next()).value(), x);
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                quoted.capturedValues(), 1);
        assertEquals(res, x + 1);
    }

    @Test(dataProvider = "ints")
    public void testCaptureIntField(int x) {
        class Context {
            final int x;

            Context(int x) {
                this.x = x;
            }

            Quotable quotable() {
                return (Quotable & IntUnaryOperator) y -> x + y;
            }
        }
        Context context = new Context(x);
        Quotable quotable = context.quotable();
        Quoted quoted = quotable.quoted();
        assertEquals(quoted.capturedValues().size(), 1);
        assertEquals(quoted.capturedValues().values().iterator().next(), context);
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                quoted.capturedValues(), 1);
        assertEquals(res, x + 1);
    }

    @DataProvider(name = "ints")
    public Object[][] ints() {
        return IntStream.range(0, 50)
                .mapToObj(i -> new Object[] { i })
                .toArray(Object[][]::new);
    }

    @Test(dataProvider = "ints")
    public void testCaptureReferenceReceiver(int i) {
        int prevCount = Box.count;
        Quotable quotable = (Quotable & IntUnaryOperator)new Box(i)::add;
        Quoted quoted = quotable.quoted();
        assertEquals(Box.count, prevCount + 1); // no duplicate receiver computation!
        assertEquals(quoted.capturedValues().size(), 1);
        assertEquals(((Box)((Var)quoted.capturedValues().values().iterator().next()).value()).i, i);
        int res = (int)Interpreter.invoke(MethodHandles.lookup(), (Op & Op.Invokable) quoted.op(),
                quoted.capturedValues(), 1);
        assertEquals(res, i + 1);
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
