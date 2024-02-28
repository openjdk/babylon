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

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestSlots
 */

public class TestSlots {
    @CodeReflection
    static double f(double i, double j) {
        i = i + j;

        double k = 4.0;
        k += i;
        return k;
    }

    @Test
    public void testF() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f");

        MethodHandle mh = generate(f);

        Assert.assertEquals(f(1.0d, 2.0d), (double) mh.invoke(1.0d, 2.0d));
    }

    @CodeReflection
    static double f2(double x, double y) {
        return x * (-Math.sin(x * y) + y) * 4.0d;
    }

    @Test
    public void testF2() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f2");

        MethodHandle mh = generate(f);

        Assert.assertEquals(f2(1.0d, 2.0d), (double) mh.invoke(1.0d, 2.0d));
    }

    @CodeReflection
    static double f3(/* independent */ double x, int y) {
        /* dependent */
        double o = 1.0;
        for (int i = 0; i < y; i = i + 1) {
            if (i > 1) {
                if (i < 5) {
                    o = o * x;
                }
            }
        }
        return o;
    }

    @Test
    public void testF3() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f3");

        MethodHandle mh = generate(f);

        for (int i = 0; i < 7; i++) {
            Assert.assertEquals(f3(2.0d, i), (double) mh.invoke(2.0d, i));
        }
    }

    @CodeReflection
    static int f4(/* Unused */ int a, int b) {
        return b;
    }

    @Test
    public void testF4() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f4");

        MethodHandle mh;
        try {
            mh = generate(f);
        } catch (VerifyError e) {
            Assert.fail("invalid class file generated", e);
            return;
        }

        Assert.assertEquals(f4(1, 2), (int) mh.invoke(1, 2));
    }

    @CodeReflection
    static double f5(/* Unused */ double a, double b) {
        return b;
    }

    @Test
    public void testF5() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f5");

        MethodHandle mh;
        try {
            mh = generate(f);
        } catch (VerifyError e) {
            Assert.fail("invalid class file generated", e);
            return;
        }

        Assert.assertEquals(f5(1.0, 2.0), (double) mh.invoke(1.0, 2.0));
    }

    static MethodHandle generate(CoreOps.FuncOp f) {
        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        lf.writeTo(System.out);

//        lf = SSA.transform(lf);
//        lf.writeTo(System.out);

        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestSlots.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

}
