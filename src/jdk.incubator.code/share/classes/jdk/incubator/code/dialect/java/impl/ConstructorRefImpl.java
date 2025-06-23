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

package jdk.incubator.code.dialect.java.impl;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ConstructorRef;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.ExternalizableTypeElement;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.util.Objects;

import static java.util.stream.Collectors.joining;

public final class ConstructorRefImpl implements ConstructorRef {

    final FunctionType type;

    static final MethodHandle MULTI_NEW_ARRAY_MH;

    static {
        try {
            MULTI_NEW_ARRAY_MH = MethodHandles.lookup().findStatic(Array.class, "newInstance",
                    MethodType.methodType(Object.class, Class.class, int[].class));
        } catch (ReflectiveOperationException ex) {
            throw new ExceptionInInitializerError(ex);
        }
    }


    public ConstructorRefImpl(FunctionType functionType) {
        this.type = functionType;
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
        Class<?> refC = resolve(l, type.returnType());
        if (type.returnType() instanceof ArrayType at) {
            if (at.dimensions() == 1) {
                return MethodHandles.arrayConstructor(refC);
            } else {
                int dims = type.parameterTypes().size();
                Class<?> elementType = refC;
                for (int i = 0 ; i < type.parameterTypes().size(); i++) {
                    elementType = elementType.componentType();
                }
                // only the use-site knows how many dimensions are specified
                return MULTI_NEW_ARRAY_MH.asType(MULTI_NEW_ARRAY_MH.type().changeReturnType(refC))
                        .bindTo(elementType)
                        .asCollector(int[].class, dims);
            }
        } else {
            // MH lookup wants a void-returning lookup type
            MethodType mt = MethodRef.toNominalDescriptor(type).resolveConstantDesc(l).changeReturnType(void.class);
            return l.findConstructor(refC, mt);
        }
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
    public ExternalizedTypeElement externalize() {
        return JavaTypeUtils.constructorRef(ExternalizableTypeElement.externalize(type.returnType()),
                type.parameterTypes().stream().map(ExternalizableTypeElement::externalize).toList());
    }

    @Override
    public String toString() {
        return JavaTypeUtils.toExternalRefString(externalize());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof ConstructorRefImpl that)) return false;
        return Objects.equals(type, that.type);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(type);
    }
}
