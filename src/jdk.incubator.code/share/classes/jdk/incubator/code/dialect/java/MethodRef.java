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

package jdk.incubator.code.dialect.java;

import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.impl.MethodRefImpl;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.FunctionType;

import java.util.List;

import static jdk.incubator.code.dialect.core.CoreType.functionType;

/**
 * The symbolic reference to a Java method.
 */
public sealed interface MethodRef extends JavaRef, TypeVariableType.Owner
        permits MethodRefImpl {

    TypeElement refType();

    String name();

    FunctionType type();

    boolean isConstructor();

    // Resolutions to methods, constructors and method handles

    Method resolveToDeclaredMethod(MethodHandles.Lookup l) throws ReflectiveOperationException;

    Constructor<?> resolveToDeclaredConstructor(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // For InvokeKind.SUPER the specialCaller == l.lookupClass() for Lookup::findSpecial
    MethodHandle resolveToHandle(MethodHandles.Lookup l, JavaOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException;

    // Method factories

    static MethodRef method(Method m) {
        return method(m.getDeclaringClass(), m.getName(),
                m.getReturnType(),
                m.getParameterTypes());
    }

    static MethodRef method(Class<?> refType, String name, MethodType mt) {
        return method(refType, name, mt.returnType(), mt.parameterList());
    }

    static MethodRef method(Class<?> refType, String name, Class<?> retType, Class<?>... params) {
        return method(refType, name, retType, List.of(params));
    }

    static MethodRef method(Class<?> refType, String name, Class<?> retType, List<Class<?>> params) {
        return method(JavaType.type(refType), name, JavaType.type(retType), params.stream().map(JavaType::type).toList());
    }


    static MethodRef method(TypeElement refType, String name, FunctionType type) {
        return new MethodRefImpl(refType, name, type);
    }

    static MethodRef method(TypeElement refType, String name, TypeElement retType, TypeElement... params) {
        return method(refType, name, functionType(retType, params));
    }

    static MethodRef method(TypeElement refType, String name, TypeElement retType, List<? extends TypeElement> params) {
        return method(refType, name, functionType(retType, params));
    }

    // Constructor factories

    static MethodRef constructor(Constructor<?> c) {
        return constructor(c.getDeclaringClass(),
                c.getParameterTypes());
    }

    static MethodRef constructor(MethodType mt) {
        return constructor(mt.returnType(), mt.parameterList());
    }

    static MethodRef constructor(Class<?> refType, Class<?>... params) {
        return constructor(refType, List.of(params));
    }

    static MethodRef constructor(Class<?> refType, List<Class<?>> params) {
        return constructor(JavaType.type(refType), params.stream().map(JavaType::type).toList());
    }

    static MethodRef constructor(TypeElement refType, List<? extends TypeElement> params) {
        return constructor(functionType(refType, params));
    }

    static MethodRef constructor(TypeElement refType, TypeElement... params) {
        return constructor(functionType(refType, params));
    }

    static MethodRef constructor(FunctionType type) {
        return new MethodRefImpl(type.returnType(), INIT_NAME, type);
    }


    // MethodTypeDesc factories
    // @@@ Where else to place them?

    static FunctionType ofNominalDescriptor(MethodTypeDesc d) {
        return CoreType.functionType(
                JavaType.type(d.returnType()),
                d.parameterList().stream().map(JavaType::type).toList());
    }

    static MethodTypeDesc toNominalDescriptor(FunctionType t) {
        return MethodTypeDesc.of(
                toClassDesc(t.returnType()),
                t.parameterTypes().stream().map(MethodRef::toClassDesc).toList());
    }

    private static ClassDesc toClassDesc(TypeElement e) {
        if (!(e instanceof JavaType jt)) {
            throw new IllegalArgumentException();
        }

        return jt.toNominalDescriptor();
    }

    String INIT_NAME = "<init>";
}