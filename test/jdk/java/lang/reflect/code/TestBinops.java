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
 * @run junit TestBinops
 */

import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
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

        System.out.println(f.toText());

        Assertions.assertEquals(not(true), Interpreter.invoke(MethodHandles.lookup(), f, true));
        Assertions.assertEquals(not(false), Interpreter.invoke(MethodHandles.lookup(), f, false));
    }

    @CodeReflection
    public static int neg(int a) {
        return -a;
    }

    @Test
    public void testNeg() {
        CoreOp.FuncOp f = getFuncOp("neg");

        System.out.println(f.toText());

        Assertions.assertEquals(neg(42), Interpreter.invoke(MethodHandles.lookup(), f, 42));
    }

    @CodeReflection
    public static int compl(int a) {
        return ~a;
    }

    @Test
    public void testCompl() {
        CoreOp.FuncOp f = getFuncOp("compl");

        System.out.println(f.toText());

        Assertions.assertEquals(compl(42), Interpreter.invoke(MethodHandles.lookup(), f, 42));
    }

    @CodeReflection
    public static int mod(int a, int b) {
        return a % b;
    }

    @Test
    public void testMod() {
        CoreOp.FuncOp f = getFuncOp("mod");

        System.out.println(f.toText());

        Assertions.assertEquals(mod(10, 3), Interpreter.invoke(MethodHandles.lookup(), f, 10, 3));
    }

    @CodeReflection
    public static int bitand(int a, int b) {
        return a & b;
    }

    @Test
    public void testBitand() {
        CoreOp.FuncOp f = getFuncOp("bitand");

        System.out.println(f.toText());

        Assertions.assertEquals(bitand(10, 3), Interpreter.invoke(MethodHandles.lookup(), f, 10, 3));
    }

    @CodeReflection
    public static int bitor(int a, int b) {
        return a | b;
    }

    @Test
    public void testBitor() {
        CoreOp.FuncOp f = getFuncOp("bitor");

        System.out.println(f.toText());

        Assertions.assertEquals(bitor(10, 3), Interpreter.invoke(MethodHandles.lookup(), f, 10, 3));
    }

    @CodeReflection
    public static int bitxor(int a, int b) {
        return a ^ b;
    }

    @Test
    public void testBitxor() {
        CoreOp.FuncOp f = getFuncOp("bitxor");

        System.out.println(f.toText());

        Assertions.assertEquals(bitxor(10, 3), Interpreter.invoke(MethodHandles.lookup(), f, 10, 3));
    }

    @CodeReflection
    public static boolean booland(boolean a, boolean b) {
        return a & b;
    }

    @Test
    public void testBooland() {
        CoreOp.FuncOp f = getFuncOp("booland");

        System.out.println(f.toText());

        Assertions.assertEquals(booland(true, false), Interpreter.invoke(MethodHandles.lookup(), f, true, false));
    }

    @CodeReflection
    public static boolean boolor(boolean a, boolean b) {
        return a | b;
    }

    @Test
    public void testBoolor() {
        CoreOp.FuncOp f = getFuncOp("boolor");

        System.out.println(f.toText());

        Assertions.assertEquals(boolor(false, true), Interpreter.invoke(MethodHandles.lookup(), f, false, true));
    }

    @CodeReflection
    public static boolean boolxor(boolean a, boolean b) {
        return a ^ b;
    }

    @Test
    public void testBoolxor() {
        CoreOp.FuncOp f = getFuncOp("boolxor");

        System.out.println(f.toText());

        Assertions.assertEquals(boolxor(true, true), Interpreter.invoke(MethodHandles.lookup(), f, true, true));
    }

    @CodeReflection
    public static double doublemod(double a, double b) {
        return a % b;
    }

    @Test
    public void testDoublemod() {
        CoreOp.FuncOp f = getFuncOp("doublemod");

        System.out.println(f.toText());

        Assertions.assertEquals(doublemod(15.6, 2.1), Interpreter.invoke(MethodHandles.lookup(), f, 15.6, 2.1));
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestBinops.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
