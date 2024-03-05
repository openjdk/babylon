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

package java.lang.reflect.code.type.impl;

import java.lang.reflect.code.type.FieldRef;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Field;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;

public final class FieldRefImpl implements FieldRef {
    final TypeElement refType;
    final String name;
    final TypeElement type;

    public FieldRefImpl(TypeElement refType, String name, TypeElement type) {
        this.refType = refType;
        this.name = name;
        this.type = type;
    }

    @Override
    public TypeElement refType() {
        return refType;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public TypeElement type() {
        return type;
    }

    @Override
    public Field resolveToMember(MethodHandles.Lookup l) throws ReflectiveOperationException {
        Class<?> refC = resolve(l, refType);
        Class<?> typeC = resolve(l, type);

        Field f = refC.getDeclaredField(name);
        if (!f.getType().equals(typeC)) {
            throw new NoSuchFieldException();
        }

        return f;
    }

    @Override
    public VarHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException {
        Class<?> refC = resolve(l, refType);
        Class<?> typeC = resolve(l, type);

        VarHandle vh = null;
        ReflectiveOperationException c = null;

        try {
            vh = l.findStaticVarHandle(refC, name, typeC);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            c = e;
        }

        if (c != null) {
            c = null;
            try {
                vh = l.findVarHandle(refC, name, typeC);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                c = e;
            }
        }

        if (c != null) {
            throw c;
        }

        assert vh != null;
        return vh;
    }

    static Class<?> resolve(MethodHandles.Lookup l, TypeElement t) throws ReflectiveOperationException {
        if (t instanceof JavaType jt) {
            return jt.resolve(l);
        } else {
            // @@@
            throw new ReflectiveOperationException();
        }
    }

    @Override
    public String toString() {
        return refType + "::" + name + "()" + type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        FieldRefImpl fieldDesc = (FieldRefImpl) o;

        if (!refType.equals(fieldDesc.refType)) return false;
        if (!name.equals(fieldDesc.name)) return false;
        return type.equals(fieldDesc.type);
    }

    @Override
    public int hashCode() {
        int result = refType.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + type.hashCode();
        return result;
    }
}