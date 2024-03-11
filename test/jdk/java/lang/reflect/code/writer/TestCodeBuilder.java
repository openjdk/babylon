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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.op.ExtendedOps;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.lang.reflect.code.writer.OpBuilder;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestCodeBuilder
 */

public class TestCodeBuilder {

    @CodeReflection
    static double f(double i, double j) {
        int sum = 0;
        for (int k = 0; k < 10; k++) {
            sum += j;
        }

        String s = "HELLO";
        int k = s.length();
        s = null;
        return i + j + k;
    }

    @Test
    public void testF() {
        CoreOps.FuncOp f = getFuncOp("f");
        test(f);

        f = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable l) {
                return l.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });
        test(f);

        f = SSA.transform(f);
        test(f);
    }

    static void test(CoreOps.FuncOp fExpected) {
        CoreOps.FuncOp fb = OpBuilder.createBuilderFunction(fExpected);
        CoreOps.FuncOp fActual = (CoreOps.FuncOp) Interpreter.invoke(MethodHandles.lookup(),
                fb, ExtendedOps.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY);
        Assert.assertEquals(fActual.toText(), fExpected.toText());
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestCodeBuilder.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

}
