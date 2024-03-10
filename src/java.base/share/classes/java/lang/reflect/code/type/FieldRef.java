/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package java.lang.reflect.code.type;

import java.lang.reflect.code.type.impl.FieldRefImpl;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.lang.reflect.code.TypeElement;

/**
 * The symbolic reference to a Java field.
 */
public sealed interface FieldRef permits FieldRefImpl {
    TypeElement refType();

    String name();

    TypeElement type();

    // Conversions

    Field resolveToMember(MethodHandles.Lookup l) throws ReflectiveOperationException;

    VarHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Factories

    static FieldRef field(Field f) {
        return field(f.getDeclaringClass(), f.getName(), f.getType());
    }

    static FieldRef field(Class<?> refType, String name, Class<?> type) {
        return field(JavaType.type(refType), name, JavaType.type(type));
    }

    static FieldRef field(TypeElement refType, String name, TypeElement type) {
        return new FieldRefImpl(refType, name, type);
    }

    // Copied code in jdk.compiler module throws UOE
    static FieldRef ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseFieldRef(s);
    }
}