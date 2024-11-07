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
 * @run testng TestInvokeSuper
 */

import org.testng.Assert;
import org.testng.annotations.Ignore;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.java.lang.reflect.code.OpTransformer;
import jdk.incubator.code.java.lang.reflect.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.java.lang.reflect.code.op.CoreOp;
import java.lang.runtime.CodeReflection;
import java.util.Optional;
import java.util.stream.Stream;

public class TestInvokeSuper {

    interface I {
        default String f() { return "I"; }
    }
    static class A {
        String f() { return "A"; }
    }

    static class B extends A implements I {
        final boolean invokeClass;

        public B(boolean invokeClass) {
            this.invokeClass = invokeClass;
        }

        @CodeReflection
        public String f() {
            return invokeClass ? super.f() : I.super.f();
        }
    }

    @Ignore
    @Test
    public void testInvokeSuper() throws Throwable {
        CoreOp.FuncOp f = getFuncOp(B.class, "f");
        MethodHandle mh = generate(f);

        for (boolean invokeClass : new boolean[] {true, false}) {
            B b = new B(invokeClass);
            Assert.assertEquals(mh.invoke(b), b.f());
        }
    }

    static MethodHandle generate(CoreOp.FuncOp f) {
        f.writeTo(System.out);

        CoreOp.FuncOp lf = f.transform(OpTransformer.LOWERING_TRANSFORMER);
        lf.writeTo(System.out);

        return BytecodeGenerator.generate(MethodHandles.lookup().in(B.class), lf);
    }

    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return m.getCodeModel().get();
    }
}
