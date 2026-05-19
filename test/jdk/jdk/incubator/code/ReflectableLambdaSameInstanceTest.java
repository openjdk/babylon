/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/*
 * @test
 * @summary test that invoking Op#ofLambda returns the same instance
 * @modules jdk.incubator.code
 * @library lib
 * @run junit ReflectableLambdaSameInstanceTest
 */

public class ReflectableLambdaSameInstanceTest {

    @Reflect
    private static final Runnable q1 = () -> { };

    @Test
    public void testWithOneThread() {
        Assertions.assertSame(Op.ofLambda(q1).get(), Op.ofLambda(q1).get());
    }

    @Reflect
    private static final IntUnaryOperator q2 = x -> x;

    @Test
    public void testWithMultiThreads() {
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofLambda(q2).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assertions.assertSame(quotedObjects[i], quotedObjects[i - 1]);
        }
    }

    @Reflect
    static IntSupplier q() {
        return () -> 8;
    }

    @Test
    public void testMultiThreadsViaInterpreter() throws NoSuchMethodException {
        var qm = this.getClass().getDeclaredMethod("q");
        var q = Op.ofMethod(qm).get();
        IntSupplier quotable = (IntSupplier) Util.interpretOp(MethodHandles.lookup(), q);
        Object[] quotedObjects = IntStream.range(0, 1024).parallel().mapToObj(__ -> Op.ofLambda(quotable).get()).toArray();
        for (int i = 1; i < quotedObjects.length; i++) {
            Assertions.assertSame(quotedObjects[i-1], quotedObjects[i]);
        }
    }
}
