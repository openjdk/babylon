/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
 * @library lib
 * @run junit TestInvokeInnerCtor
 * @run main Unreflect TestInvokeInnerCtor
 * @run junit TestInvokeInnerCtor
 */

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

public class TestInvokeInnerCtor {

    static class Base {
        class Inner {
            private String s;

            Inner(String s) {
                this.s = s;
            }

            @Override
            public boolean equals(Object obj) {
                if (obj instanceof Inner in) {
                    return Objects.equals(this.s, in.s);
                }
                return false;
            }
        }
    }

    static class Sub extends Base {
        @Reflect
        Inner make(String s) {
            return new Inner(s);
        }

        @Reflect
        String localCaptureParam(String s) {
            class Foo {
                Foo(int i) {}
                String m() { return s; }
            }
            return new Foo(10).m();
        }
    }

    @Test
    public void test() {
        CoreOp.FuncOp f = getFuncOp(Sub.class, "make");
        System.out.println(f.toText());
        Assertions.assertEquals(new Sub().make("Test"), Interpreter.invoke(MethodHandles.lookup(), f, new Sub(), "Test"));
    }

    @Test
    public void testLocalCaptureParam() {
        CoreOp.FuncOp f = getFuncOp(Sub.class, "localCaptureParam");
        System.out.println(f.toText());
        Assertions.assertEquals(new Sub().localCaptureParam("Test"), Interpreter.invoke(MethodHandles.lookup(), f, new Sub(), "Test"));
    }

    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();
        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
