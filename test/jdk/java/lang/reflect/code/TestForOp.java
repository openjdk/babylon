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

import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestForOp
 */

public class TestForOp {

    @CodeReflection
    public static int f() {
        int j = 0;
        for (int i = 0; i < 10; i++) {
            j += i;
        }
        return j;
    }

    @Test
    public void testf() {
        CoreOp.FuncOp f = getFuncOp("f");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Assertions.assertEquals(f(), Interpreter.invoke(MethodHandles.lookup(), lf));
    }

    @CodeReflection
    public static int f2() {
        int k = 0;
        for (int i = 0, j = 0; i < 10; i++, j++) {
            k += i;
            k += j;
        }
        return k;
    }

    @Test
    public void testf2() {
        CoreOp.FuncOp f = getFuncOp("f2");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Assertions.assertEquals(f2(), Interpreter.invoke(MethodHandles.lookup(), lf));
    }

    @CodeReflection
    public static int f3() {
        int k = 0;
        int i = 0;
        int j = 0;
        for (i = 0, j = 0; i < 10; i++, j++) {
            k += i;
            k += j;
        }
        return k;
    }

    @Test
    public void testf3() {
        CoreOp.FuncOp f = getFuncOp("f3");

        System.out.println(f.toText());

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);

        System.out.println(lf.toText());

        Assertions.assertEquals(f3(), Interpreter.invoke(MethodHandles.lookup(), lf));
    }

    static CoreOp.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestForOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
