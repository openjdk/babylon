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
 * @run testng TestUninitializedVariable
 */

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

public class TestUninitializedVariable {

    @CodeReflection
    static int simple(int i) {
        int x;
        x = i; // drop store
        return x;
    }

    @CodeReflection
    static int controlFlow(int i) {
        int x;
        if (i > 0) {
            x = i;  // drop store
        } else {
            x = -i;
        }
        return x;
    }

    @DataProvider
    Object[][] methods() {
        return new Object[][] {
                { "simple" },
                { "controlFlow" }
        };
    }

    @Test(dataProvider = "methods")
    public void testInterpret(String method) {
        CoreOp.FuncOp f = removeFirstStore(getFuncOp(method).transform(OpTransformer.LOWERING_TRANSFORMER));
        f.writeTo(System.out);

        Assert.assertThrows(Interpreter.InterpreterException.class, () -> Interpreter.invoke(f, 1));
    }

    @Test(dataProvider = "methods")
    public void testSSA(String method) {
        CoreOp.FuncOp f = removeFirstStore(getFuncOp(method).transform(OpTransformer.LOWERING_TRANSFORMER));
        f.writeTo(System.out);

        Assert.assertThrows(IllegalStateException.class, () -> SSA.transform(f));
    }

    static CoreOp.FuncOp removeFirstStore(CoreOp.FuncOp f) {
        AtomicBoolean b = new AtomicBoolean();
        return f.transform((block, op) -> {
            if (op instanceof CoreOp.VarAccessOp.VarStoreOp vop && !b.getAndSet(true)) {
                // Drop first encountered var store
            } else {
                block.op(op);
            }
            return block;
        });
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestUninitializedVariable.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
