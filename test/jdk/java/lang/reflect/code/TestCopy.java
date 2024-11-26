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
 * @run testng TestCopy
 */

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.reflect.Method;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.op.ExternalizableOp;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.function.IntUnaryOperator;
import java.util.stream.Stream;

public class TestCopy {

    @CodeReflection
    static int f(int i) {
        IntUnaryOperator f = j -> i + j;
        return f.applyAsInt(42);
    }

    @Test
    public void testCopy() {
        CoreOp.FuncOp f = getFuncOp("f");

        Op copy = f.copy();

        Assert.assertEquals(f.toText(), copy.toText());
    }

    @Test
    public void testCopyWithDefinition() {
        CoreOp.FuncOp f = getFuncOp("f");

        ExternalizableOp.ExternalizedOp odef = ExternalizableOp.ExternalizedOp.externalizeOp(CopyContext.create(), f);
        Op copy = CoreOp.FACTORY.constructOp(odef);

        Assert.assertEquals(f.toText(), copy.toText());
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestCopy.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
