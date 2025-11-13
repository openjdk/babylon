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
import java.lang.invoke.MethodType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestMethodResolution
 */
public class TestMethodResolution {
    public static class C {
        static public int s_x() { return 42; }
        public int x() { return 42; }
        long y() { return 42L; };
        static long s_y() { return 42L; };
    }

    public static class C_Sub extends C { }

    @Test
    public void testClassDeclaredMethodsPrivateLookup() throws ReflectiveOperationException {
        lookupInternal(C.class, C.class, MethodHandles.lookup());
    }

    @Test
    public void testClassDeclaredMethodsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(C.class, C.class, publicLookup());
    }

    @Test
    public void testClassInheritedMethodsPrivateLookup() throws ReflectiveOperationException {
        lookupInternal(C_Sub.class, C.class, MethodHandles.lookup());
    }

    @Test
    public void testClassInheritedMethodsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(C_Sub.class, C.class, publicLookup());
    }

    public interface I {
        int x();
        default int xd() {
            return 42;
        }
        static int s_x() { return 42; }
    }

    public interface I_Sub extends I { }

    public static abstract class CI_Sub implements I_Sub { }

    @Test
    public void testInterfaceDeclaredMethodsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(I.class, I.class, publicLookup());
    }

    @Test
    public void testInterfaceInheritedMethodsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(I_Sub.class, I.class, publicLookup());
    }

//    @Test
//    public void testClassInterfaceInheritedMethodsPublicLookup() throws ReflectiveOperationException {
//        lookupInternal(CI_Sub.class, I.class, publicLookup());
//    }
//    @@@: this is commented for now -- revealDirect seems to have issues when cracking interface methods from subclasses

    static void lookupInternal(Class<?> refC, Class<?> cl, Lookup lookup) throws ReflectiveOperationException {
        for (Method m : cl.getDeclaredMethods()) {
            MethodRef methodRef = MethodRef.method(refC, m.getName(), MethodType.methodType(m.getReturnType(), m.getParameterTypes()));
            boolean implLookup = (lookup.lookupModes() & Lookup.ORIGINAL) != 0;
            boolean intfMethodSub = cl.isInterface() && Modifier.isStatic(m.getModifiers()) && !cl.equals(refC);
            if (!intfMethodSub && (Modifier.isPublic(m.getModifiers()) || implLookup)) {
                Method resolvedM = methodRef.resolveToMethod(lookup);
                assertEquals(m, resolvedM);
                final List<InvokeKind> kinds = kindsToTest(m, implLookup);
                for (InvokeKind kind : kinds) {
                    MethodHandle resolvedMH = methodRef.resolveToHandle(lookup.in(refC), kind);
                    Method targetM = lookup.revealDirect(resolvedMH).reflectAs(Method.class, lookup);
                    assertEquals(targetM, m);
                }
            } else {
                assertThrows(ReflectiveOperationException.class, () -> methodRef.resolveToMethod(lookup));
            }
        }
    }

    static List<InvokeKind> kindsToTest(Method m, boolean implLookup) {
        if (Modifier.isStatic(m.getModifiers())) {
            return List.of(InvokeKind.STATIC);
        } else if (implLookup) {
            return List.of(InvokeKind.INSTANCE, InvokeKind.SUPER);
        } else {
            return List.of(InvokeKind.INSTANCE);
        }
    }

    static Lookup publicLookup() {
        return MethodHandles.publicLookup().in(TestMethodResolution.class);
    }
}
