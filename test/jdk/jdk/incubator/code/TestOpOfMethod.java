/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.stream.IntStream;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestOpOfMethod
 */
public class TestOpOfMethod {

    @Reflect
    static void f() {
    }
    @Reflect
    static void g() {
    }

    @Test
    public void testInstancesReflectSameMethodHaveSameModel() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        Method f2 = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();
        CoreOp.FuncOp fm2 = Op.ofMethod(f2).orElseThrow();
        Assertions.assertSame(fm, fm2);
    }

    @Test
    public void testInstancesReflectDiffMethodsHaveDiffModels() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        CoreOp.FuncOp fm = Op.ofMethod(f).orElseThrow();

        Method g = this.getClass().getDeclaredMethod("g");
        CoreOp.FuncOp gm = Op.ofMethod(g).orElseThrow();

        Assertions.assertNotSame(gm, fm);
    }

    @Test
    public void testOpOfMethodIsThreadSafe() throws NoSuchMethodException {
        Method f = this.getClass().getDeclaredMethod("f");
        List<Optional<CoreOp.FuncOp>> fops = IntStream.range(1, 3).parallel().mapToObj(_ -> Op.ofMethod(f)).toList();
        Assertions.assertSame(fops.getFirst(), fops.getLast());
    }
}
