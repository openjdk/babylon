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

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/*
 * @test
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestProtectedAccess
 */
public class TestProtectedAccess extends PrintWriter {

    public TestProtectedAccess() {
        super(new StringWriter());
        print("hello");
    }

    @Reflect
    public String accessProtectedField() {
        return out.toString();
    }

    @Reflect
    public boolean accessProtectedMethod() {
        setError();
        return checkError();
    }

    @Test
    public void testProtectedFieldAccess() throws Throwable {
        Method accessMethod = TestProtectedAccess.class.getDeclaredMethod("accessProtectedField");
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                         Op.ofMethod(accessMethod).orElseThrow());
        Assertions.assertEquals(new TestProtectedAccess().accessProtectedField(),
                                handle.invoke(new TestProtectedAccess()));
    }

    @Test
    public void testProtectedMethodAccess() throws Throwable {
        Method accessMethod = TestProtectedAccess.class.getDeclaredMethod("accessProtectedMethod");
        MethodHandle handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                                                         Op.ofMethod(accessMethod).orElseThrow());
        Assertions.assertEquals(new TestProtectedAccess().accessProtectedMethod(),
                                (boolean) handle.invoke(new TestProtectedAccess()));
    }

    @Test
    public void testArrayCloneAccess() throws Throwable {
        var handle = BytecodeGenerator.generate(MethodHandles.lookup(), CoreOp.func("arrayCloneTest",
                CoreType.functionType(JavaType.J_L_OBJECT, JavaType.array(JavaType.J_L_STRING))).body(bb ->
                        bb.add(CoreOp.return_(
                                bb.add(JavaOp.invoke(
                                        MethodRef.method(JavaType.array(JavaType.J_L_STRING),
                                                         "clone",
                                                         JavaType.J_L_OBJECT),
                                        bb.parameters().getFirst()))))));
        String[] input = { "a", "b" };
        Assertions.assertArrayEquals(input, (String[])handle.invoke(input));
    }
}
