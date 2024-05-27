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
 * @run testng TestBinops
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestBinops {

    @CodeReflection
    public static boolean not(boolean b) {
        return !b;
    }

    @Test
    public void testNot() {
        CoreOp.FuncOp f = getFuncOp("not");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, true), not(true));
        Assert.assertEquals(Interpreter.invoke(f, false), not(false));
    }

    @CodeReflection
    public static int mod(int a, int b) {
        return a % b;
    }

    @Test
    public void testMod() {
        CoreOp.FuncOp f = getFuncOp("mod");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 10, 3), mod(10, 3));
    }

    @CodeReflection
    public static int bitand(int a, int b) {
        return a & b;
    }

    @Test
    public void testBitand() {
        CoreOp.FuncOp f = getFuncOp("bitand");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 10, 3), bitand(10, 3));
    }

    @CodeReflection
    public static int bitor(int a, int b) {
        return a | b;
    }

    @Test
    public void testBitor() {
        CoreOp.FuncOp f = getFuncOp("bitor");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 10, 3), bitor(10, 3));
    }

    @CodeReflection
    public static int bitxor(int a, int b) {
        return a ^ b;
    }

    @Test
    public void testBitxor() {
        CoreOp.FuncOp f = getFuncOp("bitxor");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 10, 3), bitxor(10, 3));
    }

    @CodeReflection
    public static boolean booland(boolean a, boolean b) {
        return a & b;
    }

    @Test
    public void testBooland() {
        CoreOp.FuncOp f = getFuncOp("booland");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, true, false), booland(true, false));
    }

    @CodeReflection
    public static boolean boolor(boolean a, boolean b) {
        return a | b;
    }

    @Test
    public void testBoolor() {
        CoreOp.FuncOp f = getFuncOp("boolor");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, false, true), boolor(false, true));
    }

    @CodeReflection
    public static boolean boolxor(boolean a, boolean b) {
        return a ^ b;
    }

    @Test
    public void testBoolxor() {
        CoreOp.FuncOp f = getFuncOp("boolxor");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, true, true), boolxor(true, true));
    }

    @CodeReflection
    public static double doublemod(double a, double b) {
        return a % b;
    }

    @Test
    public void testDoublemod() {
        CoreOp.FuncOp f = getFuncOp("doublemod");

        f.writeTo(System.out);

        Assert.assertEquals(Interpreter.invoke(f, 15.6, 2.1), doublemod(15.6, 2.1));
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestBinops.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
