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

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run testng TestArrayTypes
 */

public class TestArrayTypes {
    @CodeReflection
    public static Class<?> f() {
        return String[].class;
    }

    @Test
    public void testf() {
        CoreOp.FuncOp f = getFuncOp("f");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), f), f());
    }

    @CodeReflection
    public static Class<?> f2() {
        return int[][].class;
    }

    @Test
    public void testf2() {
        CoreOp.FuncOp f = getFuncOp("f2");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), f), f2());
    }

    @CodeReflection
    public static Class<?> f3() {
        return CoreOp.ArrayLengthOp[][][][][][][].class;
    }

    @Test
    public void testf3() {
        CoreOp.FuncOp f = getFuncOp("f3");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), f), f3());
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestArrayTypes.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
