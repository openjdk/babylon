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
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.net.URL;

/*
 * @test
 * @run testng TestLiftTryFinally
 */

public class TestLiftTryFinally {

    static int f(int i, int j) {
        try {
            i = i + j;
        } finally {
            i = i + j;
        }
        return i;
    }

    @Test
    public void testF() throws Throwable {
        CoreOps.FuncOp flift = getFuncOp("f");
        flift.writeTo(System.out);
        Assert.assertEquals(Interpreter.invoke(flift, 1, 1), f(1, 1));
    }

    static CoreOps.FuncOp getFuncOp(String method) {
        byte[] classdata = getClassdata();
        CoreOps.FuncOp flift = BytecodeLift.lift(classdata, method);
        flift.writeTo(System.out);
//        CoreOps.FuncOp fliftcoreSSA = SSA.transform(flift);
//        fliftcoreSSA.writeTo(System.out);
        return flift;
    }

    static byte[] getClassdata() {
        URL resource = TestLiftTryFinally.class.getClassLoader()
                .getResource(TestLiftTryFinally.class.getName().replace('.', '/') + ".class");
        try {
            return resource.openStream().readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
