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

package jdk.incubator.code.type.impl;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.ConstructorRef;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.type.MethodRef;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.joining;

public final class ConstructorRefImpl implements ConstructorRef {
    final TypeElement refType;
    final FunctionType type;

    public ConstructorRefImpl(TypeElement refType, FunctionType type) {
        this.refType = refType;
        this.type = type;
    }

    @Override
    public TypeElement refType() {
        return refType;
    }

    @Override
    public FunctionType type() {
        return type;
    }

    @Override
    public Constructor<?> resolveToConstructor(MethodHandles.Lookup l) throws ReflectiveOperationException {
        MethodHandleInfo methodHandleInfo = l.revealDirect(resolveToHandle(l));
        return methodHandleInfo.reflectAs(Constructor.class, l);
    }

    @Override
    public MethodHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException {
        Class<?> refC = resolve(l, refType);
        MethodType mt = MethodRef.toNominalDescriptor(type).resolveConstantDesc(l);
        return l.findConstructor(refC, mt);
    }

    static Class<?> resolve(MethodHandles.Lookup l, TypeElement t) throws ReflectiveOperationException {
        if (t instanceof JavaType jt) {
            return (Class<?>)jt.erasure().resolve(l);
        } else {
            // @@@
            throw new ReflectiveOperationException();
        }
    }

    @Override
    public String toString() {
        return refType.externalize() + "::<new>" +
            type.parameterTypes().stream().map(t -> t.externalize().toString())
                    .collect(joining(", ", "(", ")")) + type.returnType().externalize();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ConstructorRefImpl that = (ConstructorRefImpl) o;

        if (!refType.equals(that.refType)) return false;
        return type.equals(that.type);
    }

    @Override
    public int hashCode() {
        int result = refType.hashCode();
        result = 31 * result + type.hashCode();
        return result;
    }
}
