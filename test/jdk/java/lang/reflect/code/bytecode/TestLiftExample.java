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
import java.lang.reflect.code.analysis.SSA;
import java.lang.reflect.code.bytecode.BytecodeLift;
import java.lang.reflect.code.interpreter.Interpreter;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.util.function.Function;

/*
 * @test
 * @run testng TestLiftExample
 */

public class TestLiftExample {

    public static <T, R> Function<T, R> proxy(Function<T, R> f) {
        @SuppressWarnings("unchecked")
        Function<T, R> pf = (Function<T, R>) Proxy.newProxyInstance(
                TestLiftExample.class.getClassLoader(),
                new Class[]{Function.class},
                // @@@ Change to lambda
                handler(f));
        return pf;
    }

    static InvocationHandler handler(Function<?, ?> f) {
        return (proxy, method, args) -> {
            if (method.getName().equals("apply")) {
                int r = (int) method.invoke(f, args);
                return r + 1;
            } else {
                return method.invoke(f, args);
            }
        };
    }

    @Test
    public void testF() throws Throwable {
        URL resource = TestLiftExample.class.getClassLoader().getResource(TestLiftExample.class.getName().replace('.', '/') + ".class");
        byte[] classdata = resource.openStream().readAllBytes();
        CoreOps.FuncOp flift = BytecodeLift.lift(classdata, "proxy");
        flift.writeTo(System.out);
        CoreOps.FuncOp fliftcoreSSA = SSA.transform(flift);
        fliftcoreSSA.writeTo(System.out);

        Function<Integer, Integer> f = i -> i;
        @SuppressWarnings("unchecked")
        Function<Integer, Integer> pf = (Function<Integer, Integer>) Interpreter.invoke(MethodHandles.lookup(),
                fliftcoreSSA, f);

        Assert.assertEquals((int) pf.apply(1), 2);
    }
}
