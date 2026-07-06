/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Optional;
import java.util.stream.Stream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestInvokeOp
 */
public class TestInvokeOp {

    @Test
    void test() {
        var f = getFuncOp(this.getClass(), "f");
        var invokeOps = f.elements().filter(ce -> ce instanceof JavaOp.InvokeOp).map(ce -> ((JavaOp.InvokeOp) ce)).toList();

        Assertions.assertEquals(invokeOps.get(0).operands(), invokeOps.get(0).argOperands());

        Assertions.assertEquals(invokeOps.get(1).operands().subList(0, 1), invokeOps.get(1).argOperands());

        Assertions.assertEquals(invokeOps.get(2).operands(), invokeOps.get(2).argOperands());

        Assertions.assertEquals(invokeOps.get(3).operands().subList(0, 1), invokeOps.get(3).argOperands());

        for (JavaOp.InvokeOp invokeOp : invokeOps) {
            var l = new ArrayList<>(invokeOp.argOperands());
            if (invokeOp.isVarArgs()) {
                l.addAll(invokeOp.varArgOperands());
            }
            Assertions.assertEquals(invokeOp.operands(), l);
        }
    }

    @Reflect
    void f() {
        s(1);
        s(4, 2, 3);
        i();
        i(0.0, 0.0);
    }

    static void s(int a, long... l) {}
    void i(double... d) {}

    static CoreOp.FuncOp getFuncOp(Class<?> c, String name) {
        Optional<Method> om = Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();

        Method m = om.get();
        return Op.ofMethod(m).get();
    }
}
