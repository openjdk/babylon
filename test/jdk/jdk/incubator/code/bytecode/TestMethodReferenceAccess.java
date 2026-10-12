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

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.stream.Stream;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import resolve.Parent;

/*
 * @test
 * @bug 8392556
 * @summary Test for code reflection with method reference expressions.
 * @library ../lib/
 * @modules jdk.incubator.code
 * @enablePreview
 * @run junit TestMethodReferenceAccess
 */
public class TestMethodReferenceAccess {

    @Reflect
    public static Parent direct(Parent p) {
        return p.setValue("ok");
    }

    @Reflect
    public static Parent genericReference(Parent p) {
        Optional.of("ok").ifPresent(p::setValue);
        return p;
    }

    @Reflect
    public static int nonGenericReference(Parent p) {
        return Optional.of("ok").map(p::length).get();
    }

    @Reflect
    public static <T extends Parent> T genericReferenceWithTypeVarReceiver(T p) {
        Optional.of("ok").ifPresent(p::setValue);
        return p;
    }

    @Test
    public void testDirectCall() throws Throwable {
        var parent = new Parent();
        var handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                Op.ofMethod(getMethod(TestMethodReferenceAccess.class, "direct").orElseThrow()).orElseThrow());
        Assertions.assertEquals(TestMethodReferenceAccess.direct(parent), handle.invoke(parent));
    }

    @Test
    public void testGenericMethodReference() throws Throwable {
        var parent = new Parent();
        var handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                Op.ofMethod(getMethod(TestMethodReferenceAccess.class, "genericReference").orElseThrow()).orElseThrow());
        Assertions.assertEquals(TestMethodReferenceAccess.genericReference(parent), handle.invoke(parent));
    }

    @Test
    public void testNonGenericMethodReference() throws Throwable {
        var parent = new Parent();
        var handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                Op.ofMethod(getMethod(TestMethodReferenceAccess.class, "nonGenericReference").orElseThrow()).orElseThrow());
        Assertions.assertEquals(TestMethodReferenceAccess.nonGenericReference(parent), handle.invoke(parent));
    }

    @Test
    public void testGenericMethodReferenceWithTypeVarReceiver() throws Throwable {
        var parent = new Parent();
        var handle = BytecodeGenerator.generate(MethodHandles.lookup(),
                Op.ofMethod(getMethod(TestMethodReferenceAccess.class, "genericReferenceWithTypeVarReceiver").orElseThrow()).orElseThrow());
        Assertions.assertEquals(TestMethodReferenceAccess.genericReferenceWithTypeVarReceiver(parent), handle.invoke(parent));
    }

    static Optional<Method> getMethod(Class<?> c, String name) {
        return Stream.of(c.getDeclaredMethods())
                .filter(m -> m.getName().equals(name))
                .findFirst();
    }
}
