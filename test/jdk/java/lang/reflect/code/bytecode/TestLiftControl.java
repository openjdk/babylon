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

import java.io.IOException;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.net.URL;

/*
 * @test
 * @run testng TestLiftControl
 */

public class TestLiftControl {

    static int ifelseCompare(int a, int b, int n) {
        if (n < 10) {
            a += 1;
        } else {
            b += 2;
        }
        return a + b;
    }

    @Test
    public void testifElseCompare() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("ifelseCompare");

        Assert.assertEquals((int) Interpreter.invoke(f, 0, 0, 1), ifelseCompare(0, 0, 1));
        Assert.assertEquals((int) Interpreter.invoke(f, 0, 0, 11), ifelseCompare(0, 0, 11));
    }

    // @@@ Cannot use boolean since it erases to int
    static int ifelseEquality(int v, int t1, int t1_1, int t2_1) {
        if (t1 != 0) {
            if (t1_1 != 0) {
                v += 1;
            } else {
                v += 2;
            }
        } else {
            if (t2_1 != 0) {
                v += 3;
            } else {
                v += 4;
            }
        }
        return v;
    }

    @Test
    public void testIfelseEquality() throws Throwable {
        CoreOps.FuncOp f = getFuncOp("ifelseEquality");

        Assert.assertEquals((int) Interpreter.invoke(f, 0, 1, 0, 0), ifelseEquality(0, 1, 0, 0));
        Assert.assertEquals((int) Interpreter.invoke(f, 0, 1, 1, 0), ifelseEquality(0, 1, 1, 0));
        Assert.assertEquals((int) Interpreter.invoke(f, 0, 0, 0, 1), ifelseEquality(0, 0, 0, 1));
        Assert.assertEquals((int) Interpreter.invoke(f, 0, 0, 0, 0), ifelseEquality(0, 0, 0, 0));
    }

    static CoreOps.FuncOp getFuncOp(String method) {
        byte[] classdata = getClassdata();
        CoreOps.FuncOp flift = BytecodeLift.lift(classdata, method);
        flift.writeTo(System.out);
        CoreOps.FuncOp fliftcoreSSA = SSA.transform(flift);
        fliftcoreSSA.writeTo(System.out);
        return fliftcoreSSA;
    }

    static byte[] getClassdata() {
        URL resource = TestLiftControl.class.getClassLoader()
                .getResource(TestLiftControl.class.getName().replace('.', '/') + ".class");
        try {
            return resource.openStream().readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
