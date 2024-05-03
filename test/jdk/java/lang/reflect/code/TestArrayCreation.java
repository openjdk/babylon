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

import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.interpreter.Interpreter;
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
    public void testf() {
        CoreOp.FuncOp f = getFuncOp("f");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f), f());
    }

    @CodeReflection
    public static String[][] f2() {
        return new String[10][];
    }

    @Test
    public void testf2() {
        CoreOp.FuncOp f = getFuncOp("f2");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f), f2());
    }

    @CodeReflection
    public static String[][] f3() {
        return new String[10][10];
    }

    @Test
    public void testf3() {
        CoreOp.FuncOp f = getFuncOp("f3");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f), f3());
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestArrayCreation.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
