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
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestArrayCreation
 */

public class TestArrayCreation {
    @CodeReflection
    public static String[] f() {
        return new String[10];
    }

    @Test
    public void testf() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f");

        MethodHandle mh = generate(f);

        Assert.assertEquals((String[]) mh.invoke(), f());
    }

    @CodeReflection
    public static String[][] f2() {
        return new String[10][];
    }

    @Test
    public void testf2() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f2");

        MethodHandle mh = generate(f);

        Assert.assertEquals((String[][]) mh.invoke(), f2());
    }

    @CodeReflection
    public static String[][] f3() {
        return new String[10][10];
    }

    @Test
    public void testf3() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f3");

        MethodHandle mh = generate(f);

        Assert.assertEquals((String[][]) mh.invoke(), f3());
    }

    @CodeReflection
    public static String[][] f4() {
        return new String[][]{{"one", "two"}, {"three"}};
    }

    @Test
    public void testf4() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("f4");

        MethodHandle mh = generate(f);

        Assert.assertEquals((String[][]) mh.invoke(), f4());
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

        return BytecodeGenerator.generate(MethodHandles.lookup(), lf);
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestArrayCreation.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
