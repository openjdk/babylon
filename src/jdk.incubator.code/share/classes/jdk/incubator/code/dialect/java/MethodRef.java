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

import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp.InvokeOp.InvokeKind;
import jdk.incubator.code.dialect.java.impl.MethodRefImpl;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.FunctionType;

import java.util.List;

import static jdk.incubator.code.dialect.core.CoreType.functionType;

/**
 * The symbolic reference to a Java method, called the <em>target method</em>.
 * <p>
 * All method references are defined in terms of the following attributes:
 * <ul>
 *     <li>an <em>owner type</em>, the type of which the target method is a member;</li>
 *     <li>a <em>name</em>, the name of the target method.</li>
 *     <li>a <em>type</em>, the type of the target method.</li>
 * </ul>
 * Some method references, called <em>constructor references</em> are used to model a Java constructor, called
 * the <em>target constructor</em>. The name of a constructor reference is always the special name {@code "<init>"}.
 * <p>
 * Method references can be <em>resolved</em> to their corresponding {@linkplain #resolveToDeclaredMethod(Lookup) target method} or
 * {@linkplain #resolveToDeclaredMethod(Lookup) target constructor}. Or they can be turned into a
 * {@linkplain #resolveToHandle(Lookup, InvokeKind) method handle} that can be used to invoke the target method or constructor.
 */
public sealed interface MethodRef extends JavaRef, TypeVariableType.Owner
        permits MethodRefImpl {

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
    FunctionType type();

    /**
     * {@return {@code true}, if this method reference is a constructor reference}
     */
    boolean isConstructor();

    // Resolutions to methods, constructors and method handles

    /**
     * Resolves the target method associated with this method reference. Resolution looks for a method with the given
     * {@linkplain #name() name} and {@linkplain #type() type} declared in the given {@linkplain #refType() owner type}.
     * If no such method can be found, {@link NoSuchMethodException} is thrown.
     * @return the method associated with this method reference
     * @param l the lookup used for resolving this method reference
     * @throws ReflectiveOperationException if a resolution error occurs
     * @throws UnsupportedOperationException if this reference is a constructor reference
     */
    Method resolveToDeclaredMethod(MethodHandles.Lookup l) throws ReflectiveOperationException;

    /**
     * Resolves the target constructor associated with this constructor reference. Resolution looks for a constructor with the given
     * {@linkplain #type() type} declared in the given {@linkplain #refType() owner type}.
     * If no such constructor can be found, {@link NoSuchMethodException} is thrown.
     * @return the constructor associated with this constructor reference
     * @param l the lookup used for resolving this constructor reference
     * @throws ReflectiveOperationException if a resolution error occurs
     * @throws UnsupportedOperationException if this reference is not a constructor reference
     */
    Constructor<?> resolveToDeclaredConstructor(MethodHandles.Lookup l) throws ReflectiveOperationException;

    /**
     * {@return a method handle used to invoke the target method or constructor associated with this method reference}
     * The method handle is obtained by invoking the corresponding method on the provided lookup, as determined by
     * the provided {@code kind}:
     * <ul>
     *     <li>if <code>kind == SUPER && isConstructor()</code>, then {@link MethodHandles.Lookup#findConstructor(Class, MethodType)} is used;</li>
     *     <li>if <code>kind == STATIC && !isConstructor()</code>, then {@link MethodHandles.Lookup#findStatic(Class, String, MethodType)} is used;</li>
     *     <li>if <code>kind == INSTANCE && !isConstructor()</code>, then {@link MethodHandles.Lookup#findVirtual(Class, String, MethodType)} is used;</li>
     *     <li>if <code>kind == SUPER && !isConstructor()</code>, then {@link MethodHandles.Lookup#findSpecial(Class, String, MethodType, Class)} is used;</li>
     *     <li>otherwise, the provided {@code kind} is unsupported for this method reference, and {@link IllegalArgumentException} is thrown</li>.
     * </ul>
     * @param l the lookup used for resolving this method reference
     * @throws ReflectiveOperationException if a resolution error occurs
     * @throws IllegalArgumentException if the provided {@code kind} is unsupported for this method reference
     */
    MethodHandle resolveToHandle(MethodHandles.Lookup l, JavaOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException;

    // Method factories

    /**
     * {@return a method reference obtained from the provided method}
     * @param m a reflective method
     */
    static MethodRef method(Method m) {
        return method(m.getDeclaringClass(), m.getName(),
                m.getReturnType(),
                m.getParameterTypes());
    }

    /**
     * {@return a method reference obtained from the provided owner, name and type}
     * @param refType the reference owner type
     * @param name the reference name
     * @param mt the reference type
     */
    static MethodRef method(Class<?> refType, String name, MethodType mt) {
        return method(refType, name, mt.returnType(), mt.parameterList());
    }

    /**
     * {@return a method reference obtained from the provided owner, name, return type and parameter types}
     * @param refType the reference owner type
     * @param name the reference name
     * @param retType the reference return type
     * @param params the reference parameter types
     */
    static MethodRef method(Class<?> refType, String name, Class<?> retType, Class<?>... params) {
        return method(refType, name, retType, List.of(params));
    }

    /**
     * {@return a method reference obtained from the provided owner, name, return type and parameter types}
     * @param refType the reference owner type
     * @param name the reference name
     * @param retType the reference return type
     * @param params the reference parameter types
     */
    static MethodRef method(Class<?> refType, String name, Class<?> retType, List<Class<?>> params) {
        return method(JavaType.type(refType), name, JavaType.type(retType), params.stream().map(JavaType::type).toList());
    }

    /**
     * {@return a method reference obtained from the provided owner, name and type}
     * @param refType the reference owner type
     * @param name the reference name
     * @param type the reference type
     */
    static MethodRef method(TypeElement refType, String name, FunctionType type) {
        return new MethodRefImpl(refType, name, type);
    }

    /**
     * {@return a method reference obtained from the provided owner, name, return type and parameter types}
     * @param refType the reference owner type
     * @param name the reference name
     * @param retType the reference return type
     * @param params the reference parameter types
     */
    static MethodRef method(TypeElement refType, String name, TypeElement retType, TypeElement... params) {
        return method(refType, name, functionType(retType, params));
    }

    /**
     * {@return a method reference obtained from the provided owner, name, return type and parameter types}
     * @param refType the reference owner type
     * @param name the reference name
     * @param retType the reference return type
     * @param params the reference parameter types
     */
    static MethodRef method(TypeElement refType, String name, TypeElement retType, List<? extends TypeElement> params) {
        return method(refType, name, functionType(retType, params));
    }

    // Constructor factories

    /**
     * {@return a method reference obtained from the provided constructor}
     * @param c a reflective constructor
     */
    static MethodRef constructor(Constructor<?> c) {
        return constructor(c.getDeclaringClass(),
                c.getParameterTypes());
    }

    /**
     * {@return a method reference obtained from the provided type}
     * The owner type of the returned method reference is the return type of the provided type.
     * @param mt the reference type
     */
    static MethodRef constructor(MethodType mt) {
        return constructor(mt.returnType(), mt.parameterList());
    }

    /**
     * {@return a method reference obtained from the provided owner and parameter types}
     * @param refType the reference owner type
     * @param params the reference parameter types
     */
    static MethodRef constructor(Class<?> refType, Class<?>... params) {
        return constructor(refType, List.of(params));
    }

    /**
     * {@return a method reference obtained from the provided owner and parameter types}
     * @param refType the reference owner type
     * @param params the reference parameter types
     */
    static MethodRef constructor(Class<?> refType, List<Class<?>> params) {
        return constructor(JavaType.type(refType), params.stream().map(JavaType::type).toList());
    }

    /**
     * {@return a method reference obtained from the provided owner and parameter types}
     * @param refType the reference owner type
     * @param params the reference parameter types
     */
    static MethodRef constructor(TypeElement refType, List<? extends TypeElement> params) {
        return constructor(functionType(refType, params));
    }

    /**
     * {@return a method reference obtained from the provided owner and parameter types}
     * @param refType the reference owner type
     * @param params the reference parameter types
     */
    static MethodRef constructor(TypeElement refType, TypeElement... params) {
        return constructor(functionType(refType, params));
    }

    /**
     * {@return a method reference obtained from the provided type}
     * The owner type of the returned method reference is the return type of the provided type.
     * @param type the reference type
     */
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

    /**
     * The name of a constructor reference
     */
    String INIT_NAME = "<init>";
}
