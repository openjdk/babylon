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

package java.lang.reflect.code.descriptor;

import java.lang.reflect.code.descriptor.impl.MethodDescImpl;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Executable;
import java.lang.reflect.Method;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.util.List;

import static java.lang.reflect.code.descriptor.MethodTypeDesc.methodType;

/**
 * The symbolic description of a Java method.
 */
// @@@ require invoke kind:
//    special, static, virtual
//    interface_special, interface_static, interface_virtual
//  Otherwise it is not possible to generate correct bytecode invoke instruction with
//  a symbolic reference to a method or an interface method, specifically a
//  constant pool entry of CONSTANT_Methodref_info or CONSTANT_InterfaceMethodref_info.
//
//  We can infer the kind, if we can resolve the types and lookup the declared method
public sealed interface MethodDesc permits MethodDescImpl {

    TypeElement refType();

    String name();

    MethodTypeDesc type();

    // Conversions

    Executable resolveToMember(MethodHandles.Lookup l) throws ReflectiveOperationException;

    MethodHandle resolve(MethodHandles.Lookup l) throws ReflectiveOperationException;

    // Factories

    static MethodDesc method(Method m) {
        return method(m.getDeclaringClass(), m.getName(), m.getReturnType(), m.getParameterTypes());
    }

    static MethodDesc method(Class<?> refType, String name, MethodType mt) {
        return method(JavaType.type(refType), name, MethodTypeDesc.methodType(mt));
    }

    static MethodDesc method(Class<?> refType, String name, Class<?> retType, Class<?>... params) {
        return method(JavaType.type(refType), name, methodType(retType, params));
    }

    static MethodDesc method(Class<?> refType, String name, Class<?> retType, List<Class<?>> params) {
        return method(JavaType.type(refType), name, methodType(retType, params));
    }

    static MethodDesc initMethod(MethodTypeDesc mt) {
        return new MethodDescImpl(
                mt.returnType(),
                "<init>",
                MethodTypeDesc.methodType(JavaType.VOID, mt.parameters()));
    }

    static MethodDesc method(TypeElement refType, String name, MethodTypeDesc type) {
        return new MethodDescImpl(refType, name, type);
    }

    static MethodDesc method(TypeElement refType, String name, TypeElement retType, TypeElement... params) {
        return method(refType, name, methodType(retType, params));
    }

    static MethodDesc method(TypeElement refType, String name, TypeElement retType, List<? extends TypeElement> params) {
        return method(refType, name, methodType(retType, params));
    }

    // Copied code in jdk.compiler module throws UOE
    static MethodDesc ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseMethodDesc(s);
    }
}