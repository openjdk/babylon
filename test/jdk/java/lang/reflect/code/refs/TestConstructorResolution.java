/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.MethodRef;
import org.junit.jupiter.api.Test;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestConstructorResolution
 */
public class TestConstructorResolution {
    public static class C {
        C() { }
        public C(int x) { }
    }

    public static class C_Sub extends C {
        private C_Sub(int x, int y, int z) { } // no default constructor
    }

    @Test
    public void testClassDeclaredFieldsPrivateLookup() throws ReflectiveOperationException {
        lookupInternal(C.class, C.class, MethodHandles.lookup());
    }

    @Test
    public void testClassDeclaredFieldsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(C.class, C.class, publicLookup());
    }

    @Test
    public void testClassInheritedFieldsPrivateLookup() throws ReflectiveOperationException {
        lookupInternal(C_Sub.class, C.class, MethodHandles.lookup());
    }

    @Test
    public void testClassInheritedFieldsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(C_Sub.class, C.class, publicLookup());
    }

    static void lookupInternal(Class<?> refC, Class<?> cl, Lookup lookup) throws ReflectiveOperationException {
        for (Constructor<?> c : cl.getDeclaredConstructors()) {
            MethodRef constructorRef = MethodRef.constructor(refC, c.getParameterTypes());
            if (refC.equals(cl) && (Modifier.isPublic(c.getModifiers()) || (lookup.lookupModes() & Lookup.ORIGINAL) != 0)) {
                Constructor<?> resolvedC = constructorRef.resolveToConstructor(lookup);
                assertEquals(c, resolvedC);
                MethodHandle resolvedMH = constructorRef.resolveToHandle(lookup, InvokeKind.SUPER);
                Constructor<?> targetC = lookup.revealDirect(resolvedMH).reflectAs(Constructor.class, lookup);
                assertEquals(targetC, c);
            } else {
                assertThrows(ReflectiveOperationException.class, () -> constructorRef.resolveToConstructor(lookup));
            }
        }
    }

    static Lookup publicLookup() {
        return MethodHandles.publicLookup().in(TestConstructorResolution.class);
    }
}
