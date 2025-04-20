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

package jdk.incubator.code.type;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.type.impl.ConstructorRefImpl;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.util.List;

import static jdk.incubator.code.type.FunctionType.functionType;

/**
 * The symbolic reference to a Java constructor.
 */
public sealed interface ConstructorRef extends TypeVarRef.Owner permits ConstructorRefImpl {

    default TypeElement refType() {
        return type().returnType();
    }

    FunctionType type();

    // Resolutions to constructors and method handles

    // Resolve to constructor on referenced class
    Constructor<?> resolveToConstructor(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Resolve to constructor on referenced class
    MethodHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Factories

    static ConstructorRef constructor(Constructor<?> c) {
        return constructor(c.getDeclaringClass(),
                c.getParameterTypes());
    }

    static ConstructorRef constructor(MethodType mt) {
        return constructor(mt.returnType(), mt.parameterList());
    }

    static ConstructorRef constructor(Class<?> refType, Class<?>... params) {
        return constructor(refType, List.of(params));
    }

    static ConstructorRef constructor(Class<?> refType, List<Class<?>> params) {
        return constructor(JavaType.type(refType), params.stream().map(JavaType::type).toList());
    }

    static ConstructorRef constructor(TypeElement refType, List<? extends TypeElement> params) {
        return constructor(functionType(refType, params));
    }

    static ConstructorRef constructor(TypeElement refType, TypeElement... params) {
        return constructor(functionType(refType, params));
    }

    static ConstructorRef constructor(FunctionType type) {
        return new ConstructorRefImpl(type);
    }

    static ConstructorRef ofString(String s) {
        return jdk.incubator.code.parser.impl.DescParser.parseConstructorRef(s);
    }
}