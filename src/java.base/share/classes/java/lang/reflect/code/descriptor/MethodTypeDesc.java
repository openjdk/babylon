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

import java.lang.reflect.code.descriptor.impl.MethodTypeDescImpl;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.List;

/**
 * The symbolic description of a method type, comprising descriptions of zero or more parameter types and a return type.
 */
public sealed interface MethodTypeDesc permits MethodTypeDescImpl {
    MethodTypeDesc VOID = methodType(TypeDesc.VOID);

    //

    TypeDesc returnType();

    List<TypeDesc> parameters();

    // Conversions

    MethodTypeDesc erase();

    String toNominalDescriptorString();

    default java.lang.constant.MethodTypeDesc toNominalDescriptor() {
        return java.lang.constant.MethodTypeDesc.ofDescriptor(toNominalDescriptorString());
    }

    @SuppressWarnings("cast")
    default MethodType resolve(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return toNominalDescriptor().resolveConstantDesc(l);
    }

    // Factories

    static MethodTypeDesc methodType(MethodType mt) {
        return methodType(mt.returnType(), mt.parameterList());
    }

    static MethodTypeDesc methodType(Class<?> ret, Class<?>... params) {
        return methodType(ret, List.of(params));
    }

    static MethodTypeDesc methodType(Class<?> ret, List<Class<?>> params) {
        return new MethodTypeDescImpl(TypeDesc.type(ret), params.stream().map(TypeDesc::type).toList());
    }

    static MethodTypeDesc ofNominalDescriptor(java.lang.constant.MethodTypeDesc d) {
        return methodType(TypeDesc.ofNominalDescriptor(d.returnType()),
                d.parameterList().stream().map(TypeDesc::ofNominalDescriptor).toList());
    }

    static MethodTypeDesc ofNominalDescriptorString(String d) {
        return ofNominalDescriptor(java.lang.constant.MethodTypeDesc.ofDescriptor(d));
    }

    static MethodTypeDesc methodType(TypeDesc ret, TypeDesc... params) {
        return methodType(ret, List.of(params));
    }

    static MethodTypeDesc methodType(TypeDesc ret, List<TypeDesc> params) {
        return new MethodTypeDescImpl(ret, params);
    }

    // Copied code in jdk.compiler module throws UOE
    static MethodTypeDesc ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseMethodTypeDesc(s);
    }
}
