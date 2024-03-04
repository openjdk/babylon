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

import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.reflect.code.type.impl.MethodRefImpl;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.code.TypeElement;
import java.util.List;

import static java.lang.reflect.code.type.FunctionType.functionType;

/**
 * The symbolic reference to a Java method.
 */
// @@@ require invoke kind:
//    special, static, virtual
//    interface_special, interface_static, interface_virtual
//  Otherwise it is not possible to generate correct bytecode invoke instruction with
//  a symbolic reference to a method or an interface method, specifically a
//  constant pool entry of CONSTANT_Methodref_info or CONSTANT_InterfaceMethodref_info.
//
//  We can infer the kind, if we can resolve the types and lookup the declared method
public sealed interface MethodRef permits MethodRefImpl {

    TypeElement refType();

    String name();

    FunctionType type();

    // Conversions

    Executable resolveToMember(MethodHandles.Lookup l) throws ReflectiveOperationException;

    MethodHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Factories

    static MethodRef method(Method m) {
        return method(m.getDeclaringClass(), m.getName(), m.getReturnType(), m.getParameterTypes());
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

    // Copied code in jdk.compiler module throws UOE
    static MethodRef ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseMethodRef(s);
    }


    // MethodTypeDesc factories
    // @@@ Where else to place them?

    static FunctionType ofNominalDescriptor(MethodTypeDesc d) {
        return FunctionType.functionType(
                JavaType.ofNominalDescriptor(d.returnType()),
                d.parameterList().stream().map(JavaType::ofNominalDescriptor).toList());
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
}