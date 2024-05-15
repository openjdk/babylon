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
 * @run testng TestNaming
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.StringWriter;
import java.lang.reflect.Method;
import java.lang.reflect.code.CodeItem;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.OpTransformer;
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.writer.OpWriter;
import java.lang.runtime.CodeReflection;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Stream;

public class TestNaming {

    @CodeReflection
    static int f(int n, int m) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sum += i;
                sum += j;
            }
        }
        return sum;
    }

    @Test
    public void testHigh() {
        CoreOp.FuncOp f = getFuncOp("f");

        testModel(f);
    }

    @Test
    public void testLow() {
        CoreOp.FuncOp f = getFuncOp("f");

        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        f = SSA.transform(f);

        testModel(f);
    }

    static void testModel(Op op) {
        Function<CodeItem, String> cNamer = OpWriter.computeGlobalNames(op);

        StringWriter w = new StringWriter();
        new OpWriter(w, OpWriter.CodeItemNamerOption.of(cNamer::apply)).writeOp(op);
        w.write("\n");
        String actual = w.toString();

        String expected = op.toText();

        Assert.assertEquals(actual, expected);
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestNaming.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
