/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
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

package jdk.incubator.code.dialect.java;

import jdk.incubator.code.dialect.java.impl.FieldRefImpl;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import jdk.incubator.code.TypeElement;

/**
 * The symbolic reference to a Java field, called the <em>target field</em>.
 * <p>
 * All field references are defined in terms of the following attributes:
 * <ul>
 *     <li>an <em>owner type</em>, the type of which the target field is a member;</li>
 *     <li>a <em>name</em>, the name of the target field.</li>
 *     <li>a <em>type</em>, the type of the target field.</li>
 * </ul>
 * <p>
 * Field references can be <em>resolved</em> to their corresponding {@linkplain #resolveToField(Lookup) target field}.
 * Or they can be turned into a {@linkplain #resolveToHandle(Lookup) var handle} that can be used to access the target field.
 */
public sealed interface FieldRef extends JavaRef
        permits FieldRefImpl {

    /**
     * {@return the owner type of this method reference}
     */
    TypeElement refType();

    /**
     * {@return the name of this method reference}
     */
    String name();

    /**
     * {@return the type of this method reference}
     */
    TypeElement type();

    // Conversions

    /**
     * Resolves the target field associated with this field reference.
     * @return the field associated with this field reference
     * @param l the lookup used for resolving this field reference
     * @throws ReflectiveOperationException if a resolution error occurs
     * @throws UnsupportedOperationException if this reference is not a constructor reference
     */
    Field resolveToField(MethodHandles.Lookup l) throws ReflectiveOperationException;

    /**
     * {@return a var handle used to access the target field associated with this field reference}
     * The var handle is obtained by invoking the corresponding method on the provided lookup, in the following order:
     * <ol>
     *     <li>first {@link MethodHandles.Lookup#findStaticVarHandle(Class, String, Class)} is used;</li>
     *     <li>if the above step fails, then {@link MethodHandles.Lookup#findVarHandle(Class, String, Class)} is used;</li>
     *     <li>otherwise, resolution fails and an exception is thrown</li>.
     * </ol>
     * @param l the lookup used for resolving this field reference
     * @throws ReflectiveOperationException if a resolution error occurs
     */
    VarHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Factories

    /**
     * {@return a field reference obtained from the provided field}
     * @param f a reflective field
     */
    static FieldRef field(Field f) {
        return field(f.getDeclaringClass(), f.getName(), f.getType());
    }

    /**
     * {@return a field reference obtained from the provided owner, name and type}
     * @param refType the reference owner type
     * @param name the reference name
     * @param type the reference type
     */
    static FieldRef field(Class<?> refType, String name, Class<?> type) {
        return field(JavaType.type(refType), name, JavaType.type(type));
    }

    /**
     * {@return a field reference obtained from the provided owner, name and type}
     * @param refType the reference owner type
     * @param name the reference name
     * @param type the reference type
     */
    static FieldRef field(TypeElement refType, String name, TypeElement type) {
        return new FieldRefImpl(refType, name, type);
    }
}
