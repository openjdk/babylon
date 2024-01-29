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

import java.lang.reflect.code.Block;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestForwardAutoDiff
 */

public class TestForwardAutoDiff {
    static final double PI_4 = Math.PI / 4;

    @Test
    public void testExpression() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f");
        f.writeTo(System.out);

        f = SSA.transform(f);
        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 0.0, 1.0), f(0.0, 1.0));
        Assert.assertEquals(Interpreter.invoke(f, PI_4, PI_4), f(PI_4, PI_4));

        Block.Parameter x = f.body().entryBlock().parameters().get(0);
        Block.Parameter y = f.body().entryBlock().parameters().get(1);

        CoreOps.FuncOp dff_dx = ExpressionElimination.eliminate(ForwardDifferentiation.diff(f, x));
        dff_dx.writeTo(System.out);
        MethodHandle dff_dx_mh = generate(dff_dx);
        Assert.assertEquals((double) dff_dx_mh.invoke(0.0, 1.0), df_dx(0.0, 1.0));
        Assert.assertEquals((double) dff_dx_mh.invoke(PI_4, PI_4), df_dx(PI_4, PI_4));

        CoreOps.FuncOp dff_dy = ExpressionElimination.eliminate(ForwardDifferentiation.diff(f, y));
        dff_dy.writeTo(System.out);
        MethodHandle dff_dy_mh = generate(dff_dy);
        Assert.assertEquals((double) dff_dy_mh.invoke(0.0, 1.0), df_dy(0.0, 1.0));
        Assert.assertEquals((double) dff_dy_mh.invoke(PI_4, PI_4), df_dy(PI_4, PI_4));
    }

    @CodeReflection
    static double f(double x, double y) {
        return x * (-Math.sin(x * y) + y) * 4.0d;
    }

    static double df_dx(double x, double y) {
        return (-Math.sin(x * y) + y - x * Math.cos(x * y) * y) * 4.0d;
    }

    static double df_dy(double x, double y) {
        return x * (1 - Math.cos(x * y) * x) * 4.0d;
    }

    @Test
    public void testControlFlow() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("fcf");
        f.writeTo(System.out);

        f = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        f.writeTo(System.out);

        f = SSA.transform(f);
        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 2.0, 6), fcf(2.0, 6));
        Assert.assertEquals(Interpreter.invoke(f, 2.0, 5), fcf(2.0, 5));
        Assert.assertEquals(Interpreter.invoke(f, 2.0, 4), fcf(2.0, 4));

        Block.Parameter x = f.body().entryBlock().parameters().get(0);

        CoreOps.FuncOp df_dx = ForwardDifferentiation.diff(f, x);
        df_dx.writeTo(System.out);
        MethodHandle df_dx_mh = generate(df_dx);

        Assert.assertEquals((double) df_dx_mh.invoke(2.0, 6), dfcf_dx(2.0, 6));
        Assert.assertEquals((double) df_dx_mh.invoke(2.0, 5), dfcf_dx(2.0, 5));
        Assert.assertEquals((double) df_dx_mh.invoke(2.0, 4), dfcf_dx(2.0, 4));
    }

    @CodeReflection
    static double fcf(/* independent */ double x, int y) {
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

    static double dfcf_dx(/* independent */ double x, int y) {
        double d_o = 0.0;
        double o = 1.0;
        for (int i = 0; i < y; i = i + 1) {
            if (i > 1) {
                if (i < 5) {
                    d_o = d_o * x + o * 1.0;
                    o = o * x;
                }
            }
        }
        return d_o;
    }

    static MethodHandle generate(CoreOps.FuncOp f) {
        return BytecodeGenerator.generate(MethodHandles.lookup(), f);
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestForwardAutoDiff.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
