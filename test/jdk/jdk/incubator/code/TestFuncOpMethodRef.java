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
 * @modules jdk.incubator.code/jdk.incubator.code.internal
 * @library lib
 * @run junit TestFuncOpMethodRef
 */

import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static jdk.incubator.code.dialect.core.CoreOp.*;

public class TestFuncOpMethodRef {

    final JavaType THIS_CLASS = JavaType.type(this.getClass().describeConstable().get());
    static final JavaType J_U_MAP = JavaType.type(Map.class.describeConstable().get());

    static void test(FuncOp op, MethodRef expectedMethodRef) {
        Assertions.assertTrue(op.source().isPresent());
        Assertions.assertEquals(expectedMethodRef, op.source().get());
    }

    @Reflect
    static int f(List<Integer> l) {
        return l.getFirst();
    }

    @Test
    void testStatic() throws NoSuchMethodException {
        FuncOp fop = Op.ofMethod(this.getClass().getDeclaredMethod("f", List.class)).get();
        MethodRef fmr = MethodRef.method(THIS_CLASS, "f", JavaType.INT,
                JavaType.parameterized(JavaType.J_U_LIST, JavaType.J_L_INTEGER));
        test(fop, fmr);

        FuncOp tfop = fop.transform(CodeTransformer.COPYING_TRANSFORMER);
        test(tfop, fmr);
    }

    @Reflect
    int g(String k, Map<String, Integer> m) {
        return m.get(k);
    }

    @Test
    void testInstance() throws NoSuchMethodException {
        FuncOp gop = Op.ofMethod(this.getClass().getDeclaredMethod("g", String.class, Map.class)).get();
        MethodRef gmr = MethodRef.method(THIS_CLASS, "g", JavaType.INT,
                JavaType.J_L_STRING, JavaType.parameterized(J_U_MAP, JavaType.J_L_STRING, JavaType.J_L_INTEGER));
        test(gop, gmr);

        FuncOp tgop = gop.transform(CodeTransformer.COPYING_TRANSFORMER);
        test(tgop, gmr);
    }
}