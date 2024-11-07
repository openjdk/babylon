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
 * @modules jdk.incubator.code
 * @run testng TestNestedCapturingLambda
 */

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.Quotable;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.function.IntSupplier;
import java.util.stream.Stream;

public class TestNestedCapturingLambda {

    @FunctionalInterface
    interface QIntSupplier extends IntSupplier, Quotable {
    }

    @CodeReflection
    static public int f(int a) {
        if (a > 0) {
            QIntSupplier s = () -> a;
            test(s, a);
            return s.getAsInt();
        } else {
            return 0;
        }
    }

    static void test(QIntSupplier s, int a) {
        @SuppressWarnings("unchecked")
        CoreOp.Var<Integer> capture = (CoreOp.Var<Integer>) s.quoted().capturedValues().values().iterator().next();
        Assert.assertEquals(capture.value().intValue(), a);
    }

    @Test
    public void testf() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("f");

        MethodHandle mh = generate(f);

        Assert.assertEquals((int) mh.invoke(42), f(42));
        Assert.assertEquals((int) mh.invoke(-1), f(-1));
    }

    static MethodHandle generate(CoreOp.FuncOp f) {
        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestNestedCapturingLambda.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
