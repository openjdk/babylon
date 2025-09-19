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
 * @run junit TestOptions
 */

import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.extern.OpWriter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

public class TestOptions {

    @CodeReflection
    static int f(int n) {
        return n;
    }

    @Test
    public void testDropWriteVoid() {
        CoreOp.FuncOp f = getFuncOp("f");

        Assertions.assertFalse(OpWriter.toText(f).contains("void"));
        Assertions.assertFalse(OpWriter.toText(f, OpWriter.VoidOpResultOption.DROP_VOID).contains("void"));
        Assertions.assertTrue(OpWriter.toText(f, OpWriter.VoidOpResultOption.WRITE_VOID).contains("void"));
    }

    @Test
    public void testDropWriteDescendants() {
        CoreOp.FuncOp f = getFuncOp("f");

        Assertions.assertTrue(OpWriter.toText(f).lines().count() > 1);
        Assertions.assertTrue(OpWriter.toText(f, OpWriter.OpDescendantsOption.WRITE_DESCENDANTS).lines().count() > 1);
        Assertions.assertTrue(OpWriter.toText(f, OpWriter.OpDescendantsOption.DROP_DESCENDANTS).lines().count() == 1);
    }


    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestOptions.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
