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

import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaType;
import org.junit.jupiter.api.Test;

import java.lang.constant.ClassDesc;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.VarHandle;
import java.lang.invoke.VarHandle.VarHandleDesc;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/*
 * @test
 * @modules jdk.incubator.code
 * @modules java.base/java.lang.invoke:open
 * @run junit TestFieldResolution
 */
public class TestFieldResolution {
    public static class C {
        public static int s_x;
        public int x;
        long y;
        static long s_y;
    }

    public static class C_Sub extends C { }

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

    public interface I {
        int X = 42;
    }

    public interface I_Sub extends I { }

    public static class CI_Sub implements I_Sub { }

    @Test
    public void testInterfaceDeclaredFieldsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(I.class, I.class, publicLookup());
    }

    @Test
    public void testInterfaceInheritedFieldsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(I_Sub.class, I.class, publicLookup());
    }

    @Test
    public void testClassInterfaceInheritedFieldsPublicLookup() throws ReflectiveOperationException {
        lookupInternal(CI_Sub.class, I.class, publicLookup());
    }

    static void lookupInternal(Class<?> refC, Class<?> cl, MethodHandles.Lookup lookup) throws ReflectiveOperationException {
        for (Field f : cl.getDeclaredFields()) {
            FieldRef fieldRef = FieldRef.field(refC, f.getName(), f.getType());
            if (Modifier.isPublic(f.getModifiers()) || (lookup.lookupModes() & Lookup.ORIGINAL) != 0) {
                Field resolvedF = fieldRef.resolveToField(lookup);
                assertEquals(f, resolvedF);
                VarHandle resolvedVH = fieldRef.resolveToHandle(lookup);
                try {
                    VarHandleDesc vhDesc = resolvedVH.describeConstable().get();
                    FieldRef vhRef = FieldRef.field(
                            JavaType.type(varHandleDescDeclaringClass(vhDesc)),
                            vhDesc.constantName(),
                            JavaType.type(vhDesc.varType()));
                    assertEquals(vhRef.resolveToField(lookup), f);
                } catch (InternalError ex) {
                    // @@@: this is a workaround -- there seems to be an issue with describeConstable for some VHs
                }
            } else {
                assertThrows(ReflectiveOperationException.class, () -> fieldRef.resolveToField(lookup));
            }
        }
    }

    static MethodHandles.Lookup publicLookup() {
        return MethodHandles.publicLookup().in(TestFieldResolution.class);
    }

    static ClassDesc varHandleDescDeclaringClass(VarHandleDesc varHandleDesc) throws ReflectiveOperationException {
        Field f = VarHandleDesc.class.getDeclaredField("declaringClass");
        f.setAccessible(true);
        return (ClassDesc) f.get(varHandleDesc);
    }
}
