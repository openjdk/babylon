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
 * @run testng TestInvokeSuper
 */

import jdk.incubator.code.Op;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.interpreter.Interpreter;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestInvokeSuper {

    interface I {
        default String f() { return "I"; }
    }
    static class A {
        String f() { return "A"; }
    }

    public static class B extends A implements I {
        final boolean invokeClass;

        public B(boolean invokeClass) {
            this.invokeClass = invokeClass;
        }

        @CodeReflection
        public String f() {
            return invokeClass ? super.f() : I.super.f();
        }
    }

    @Test
    public void testInvokeSuper() {
        CoreOp.FuncOp f = getFuncOp(B.class, "f");
        f = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        f.writeTo(System.out);

        for (boolean invokeClass : new boolean[] {true, false}) {
            B b = new B(invokeClass);
            Assert.assertEquals(Interpreter.invoke(MethodHandles.lookup(), f, b), b.f());
        }
    }

    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
