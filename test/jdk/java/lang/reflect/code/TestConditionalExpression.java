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
 * @run testng TestConditionalExpression
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.Op;
import jdk.incubator.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestConditionalExpression {
    @CodeReflection
    public static int simpleExpression(boolean b, int x, int y) {
        return b ? x : y;
    }

    @Test
    public void testSimpleExpression() {
        CoreOp.FuncOp f = getFuncOp("simpleExpression");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, true, 1, 2), simpleExpression(true, 1, 2));
        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), lf, false, 1, 2), simpleExpression(false, 1, 2));
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestConditionalExpression.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
