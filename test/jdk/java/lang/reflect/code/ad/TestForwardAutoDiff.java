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

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.analysis.SSA;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestForwardAutoDiff
 * @run junit/othervm -Dbabylon.ssa=cytron TestForwardAutoDiff
 */

public class TestForwardAutoDiff {
    static final double PI_4 = Math.PI / 4;

    @Test
    public void testExpression() throws Throwable {
        CoreOp.FuncOp f = getFuncOp("f");
        System.out.println(f.toText());

        f = SSA.transform(f);
        System.out.println(f.toText());

        Assertions.assertEquals(f(0.0, 1.0), Interpreter.invoke(MethodHandles.lookup(), f, 0.0, 1.0));
        Assertions.assertEquals(f(PI_4, PI_4), Interpreter.invoke(MethodHandles.lookup(), f, PI_4, PI_4));

        Block.Parameter x = f.body().entryBlock().parameters().get(0);
        Block.Parameter y = f.body().entryBlock().parameters().get(1);

        CoreOp.FuncOp dff_dx = ExpressionElimination.eliminate(ForwardDifferentiation.partialDiff(f, x));
        System.out.println(dff_dx.toText());
        MethodHandle dff_dx_mh = generate(dff_dx);
        Assertions.assertEquals(df_dx(0.0, 1.0), (double) dff_dx_mh.invoke(0.0, 1.0));
        Assertions.assertEquals(df_dx(PI_4, PI_4), (double) dff_dx_mh.invoke(PI_4, PI_4));

        CoreOp.FuncOp dff_dy = ExpressionElimination.eliminate(ForwardDifferentiation.partialDiff(f, y));
        System.out.println(dff_dy.toText());
        MethodHandle dff_dy_mh = generate(dff_dy);
        Assertions.assertEquals(df_dy(0.0, 1.0), (double) dff_dy_mh.invoke(0.0, 1.0));
        Assertions.assertEquals(df_dy(PI_4, PI_4), (double) dff_dy_mh.invoke(PI_4, PI_4));
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
        CoreOp.FuncOp f = getFuncOp("fcf");
        System.out.println(f.toText());

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        System.out.println(f.toText());

        f = SSA.transform(f);
        System.out.println(f.toText());

        Assertions.assertEquals(fcf(2.0, 6), Interpreter.invoke(MethodHandles.lookup(), f, 2.0, 6));
        Assertions.assertEquals(fcf(2.0, 5), Interpreter.invoke(MethodHandles.lookup(), f, 2.0, 5));
        Assertions.assertEquals(fcf(2.0, 4), Interpreter.invoke(MethodHandles.lookup(), f, 2.0, 4));

        Block.Parameter x = f.body().entryBlock().parameters().get(0);

        CoreOp.FuncOp df_dx = ForwardDifferentiation.partialDiff(f, x);
        System.out.println(df_dx.toText());
        MethodHandle df_dx_mh = generate(df_dx);

        Assertions.assertEquals(dfcf_dx(2.0, 6), (double) df_dx_mh.invoke(2.0, 6));
        Assertions.assertEquals(dfcf_dx(2.0, 5), (double) df_dx_mh.invoke(2.0, 5));
        Assertions.assertEquals(dfcf_dx(2.0, 4), (double) df_dx_mh.invoke(2.0, 4));
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

    static MethodHandle generate(CoreOp.FuncOp f) {
        return BytecodeGenerator.generate(MethodHandles.lookup(), f);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestForwardAutoDiff.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
