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

import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.classfile.components.ClassPrinter;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.bytecode.BytecodeGenerator;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.net.URL;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestLiftSimple
 */

public class TestLiftSimple {

    @CodeReflection
    static int f(int i, int j) {
        i = i + j;
        return i;
    }

    @Test
    public void testF() throws Throwable {
        CoreOps.FuncOp flift = getFuncOp("f");
        flift.writeTo(System.out);
        Assert.assertEquals(Interpreter.invoke(flift, 1, 1), f(1, 1));
    }

    @CodeReflection
    static int f2() {
        Class<?>[] ifaces = new Class[1];
        ifaces[0] = Function.class;
        return ifaces.length;
    }

    @Test
    public void testF2() {
        CoreOps.FuncOp flift = getFuncOp("f2");
        flift.writeTo(System.out);
        Assert.assertEquals(Interpreter.invoke(flift), f2());
    }

    @CodeReflection
    static String f3() {
        return new String("hello".getBytes(), 1, 3);
    }

    @Test
    public void testF3() {
        CoreOps.FuncOp flift = getFuncOp("f3");
        flift.writeTo(System.out);
        Assert.assertEquals(Interpreter.invoke(flift), f3());
    }

    static CoreOps.FuncOp getFuncOp(String method) {
        byte[] classdata = getClassdata();
        CoreOps.FuncOp flift = BytecodeLift.lift(classdata, method);
        flift.writeTo(System.out);
        return flift;
    }

    static byte[] getClassdata() {
        URL resource = TestLiftSimple.class.getClassLoader()
                .getResource(TestLiftSimple.class.getName().replace('.', '/') + ".class");
        try {
            return resource.openStream().readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
