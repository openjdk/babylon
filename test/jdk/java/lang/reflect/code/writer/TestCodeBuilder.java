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
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.op.ExtendedOp;
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
    static void constants() {
        boolean bool = false;
        byte b = 1;
        char c = 'a';
        short s = 1;
        int i = 1;
        long l = 1L;
        float f = 1.0f;
        double d = 1.0;
        String str = "1";
        Object obj = null;
        Class<?> klass = Object.class;
    }

    @Test
    public void testConstants() {
        testWithTransforms(getFuncOp("constants"));
    }

    static record X(int f) {
        void m() {}
    }

    @CodeReflection
    static void reflect() {
        X x = new X(1);
        int i = x.f;
        x.m();
        X[] ax = new X[1];
        int l = ax.length;
        x = ax[0];

        Object o = x;
        x = (X) o;
        if (o instanceof X) {
            return;
        }
        if (o instanceof X(var a)) {
            return;
        }
    }

    @Test
    public void testReflect() {
        testWithTransforms(getFuncOp("reflect"));
    }

    @CodeReflection
    static int bodies(int m, int n) {
        int sum = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum += i + j;
            }
        }
        return m > 10 ? sum : 0;
    }

    @Test
    public void testBodies() {
        testWithTransforms(getFuncOp("bodies"));
    }

    public void testWithTransforms(CoreOp.FuncOp f) {
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

    static void test(CoreOp.FuncOp fExpected) {
        CoreOp.FuncOp fb = OpBuilder.createBuilderFunction(fExpected);
        CoreOp.FuncOp fActual = (CoreOp.FuncOp) Interpreter.invoke(MethodHandles.lookup(),
                fb, ExtendedOp.FACTORY, CoreTypeFactory.CORE_TYPE_FACTORY);
        Assert.assertEquals(fActual.toText(), fExpected.toText());
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestCodeBuilder.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
