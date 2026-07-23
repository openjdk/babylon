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
 * @run junit TestAssertOp
 * @run junit/othervm -ea TestAssertOp
 * @run main Unreflect TestAssertOp
 * @run junit TestAssertOp
 * @run junit/othervm -ea TestAssertOp
 */

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class TestAssertOp {

    @Reflect
    int check(int i) {
        assert i >= 0 : "Failed";
        return i;
    }
    @Test
    public void test() {
        CoreOp.FuncOp f = getFuncOp(TestAssertOp.class, "check");

        boolean assertEnabled = false;
        assert assertEnabled = true;

        if (assertEnabled) {
            Assertions.assertThrows(AssertionError.class, () -> Interpreter.invoke(MethodHandles.lookup(), f, new TestAssertOp(), -42));
        } else {
            Assertions.assertEquals(new TestAssertOp().check(-42), Interpreter.invoke(MethodHandles.lookup(), f, new TestAssertOp(), -42));
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
