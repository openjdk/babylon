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

package jdk.incubator.code.dialect.java.impl;

import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.MethodRef;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Array;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.extern.ExternalizedTypeElement;

public record MethodRefImpl(TypeElement refType, String name, FunctionType type) implements MethodRef {

    static final MethodHandle MULTI_NEW_ARRAY_MH;

    static {
        try {
            MULTI_NEW_ARRAY_MH = MethodHandles.lookup().findStatic(Array.class, "newInstance",
                    MethodType.methodType(Object.class, Class.class, int[].class));
        } catch (ReflectiveOperationException ex) {
            throw new ExceptionInInitializerError(ex);
        }
    }

    @Override
    public boolean isConstructor() {
        return name.equals(INIT_NAME);
    }

    @Override
    public Method resolveToMethod(MethodHandles.Lookup l) throws ReflectiveOperationException {
        if (isConstructor()) {
            throw new UnsupportedOperationException("Not a method reference");
        }
        MethodHandle mh = ResolutionHelper.resolveMethod(l, this);
        return l.revealDirect(mh)
                .reflectAs(Method.class, l);
    }

    @Override
    public Constructor<?> resolveToConstructor(MethodHandles.Lookup l) throws ReflectiveOperationException {
        if (!isConstructor()) {
            throw new UnsupportedOperationException("Not a constructor reference");
        }
        return l.revealDirect(resolveToHandle(l, InvokeKind.SUPER))
                .reflectAs(Constructor.class, l);
    }

    @Override
    public MethodHandle resolveToHandle(MethodHandles.Lookup l, JavaOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException {
        if (!isConstructor()) {
            return ResolutionHelper.resolveMethod(l, this, kind);
        } else {
            if (kind != JavaOp.InvokeOp.InvokeKind.SUPER) {
                throw new IllegalArgumentException("Bad invoke kind for constructor: " + kind);
            }
            return resolveToConstructorHandle(l);
        }
    }

    private MethodHandle resolveToConstructorHandle(MethodHandles.Lookup l) throws ReflectiveOperationException {
        Class<?> refC = ResolutionHelper.resolveClass(l, type.returnType());
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

    @Override
    public ExternalizedTypeElement externalize() {
        if (!isConstructor()) {
            return JavaTypeUtils.methodRef(name, refType.externalize(),
                    type.returnType().externalize(),
                    type.parameterTypes().stream().map(TypeElement::externalize).toList());
        } else {
            return JavaTypeUtils.constructorRef(type.returnType().externalize(),
                    type.parameterTypes().stream().map(TypeElement::externalize).toList());
        }
    }

    @Override
    public String toString() {
        return JavaTypeUtils.toExternalRefString(externalize());
    }
}