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

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.runtime.CodeReflection;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @run testng TestConditionalOp
 */

public class TestConditionalOp {

    @CodeReflection
    static boolean f(boolean a, boolean b, boolean c, List<String> l) {
        return F.a(a, l) || (F.b(b, l) && F.c(c, l));
    }

    static class F {
        static boolean a(boolean a, List<String> l) {
            l.add("a");
            return a;
        }

        static boolean b(boolean b, List<String> l) {
            l.add("b");
            return b;
        }

        static boolean c(boolean c, List<String> l) {
            l.add("c");
            return c;
        }
    }

    static CoreOps.FuncOp getFuncOp(String name) {
        Optional<Method> om = Stream.of(TestConditionalOp.class.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }

    @Test
    public void testf() {
        CoreOps.FuncOp f = getFuncOp("f");

        f.writeTo(System.out);

        CoreOps.FuncOp lf = f.transform((block, op) -> {
            if (op instanceof Op.Lowerable lop) {
                return lop.lower(block);
            } else {
                block.op(op);
                return block;
            }
        });

        lf.writeTo(System.out);

        for (int i = 0; i < 8; i++) {
            boolean a = (i & 1) != 0;
            boolean b = (i & 2) != 0;
            boolean c = (i & 4) != 0;
            List<String> la = new ArrayList<>();
            boolean ra = (boolean) Interpreter.invoke(MethodHandles.lookup(), lf, a, b, c, la);

            List<String> le = new ArrayList<>();
            boolean re = f(a, b, c, le);

            Assert.assertEquals(ra, re);
            Assert.assertEquals(la, le);
        }
    }
}
