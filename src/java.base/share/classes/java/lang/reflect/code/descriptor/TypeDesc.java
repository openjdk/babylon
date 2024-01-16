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

import java.lang.constant.ClassDesc;
import java.lang.reflect.code.descriptor.impl.TypeDescImpl;
import java.lang.invoke.MethodHandles;
import java.util.List;

/**
 * The symbolic description of a type.
 */
public sealed interface TypeDesc permits TypeDescImpl {

    TypeDesc VOID = new TypeDescImpl("void");

    TypeDesc BOOLEAN = new TypeDescImpl("boolean");

    TypeDesc J_L_BOOLEAN = new TypeDescImpl("java.lang.Boolean");

    TypeDesc BOOLEAN_ARRAY = new TypeDescImpl("boolean", 1);

    TypeDesc BYTE = new TypeDescImpl("byte");

    TypeDesc J_L_BYTE = new TypeDescImpl("java.lang.Byte");

    TypeDesc BYTE_ARRAY = new TypeDescImpl("byte", 1);

    TypeDesc CHAR = new TypeDescImpl("char");

    TypeDesc J_L_CHARACTER = new TypeDescImpl("java.lang.Character");

    TypeDesc CHAR_ARRAY = new TypeDescImpl("char", 1);

    TypeDesc SHORT = new TypeDescImpl("short");

    TypeDesc J_L_SHORT = new TypeDescImpl("java.lang.Short");

    TypeDesc SHORT_ARRAY = new TypeDescImpl("short", 1);

    TypeDesc INT = new TypeDescImpl("int");

    TypeDesc J_L_INTEGER = new TypeDescImpl("java.lang.Integer");

    TypeDesc INT_ARRAY = new TypeDescImpl("int", 1);

    TypeDesc LONG = new TypeDescImpl("long");

    TypeDesc J_L_LONG = new TypeDescImpl("java.lang.Long");

    TypeDesc LONG_ARRAY = new TypeDescImpl("long", 1);

    TypeDesc FLOAT = new TypeDescImpl("float");

    TypeDesc J_L_FLOAT = new TypeDescImpl("java.lang.Float");

    TypeDesc FLOAT_ARRAY = new TypeDescImpl("float", 1);

    TypeDesc DOUBLE = new TypeDescImpl("double");

    TypeDesc J_L_DOUBLE = new TypeDescImpl("java.lang.Double");

    TypeDesc DOUBLE_ARRAY = new TypeDescImpl("double", 1);

    TypeDesc J_L_OBJECT = new TypeDescImpl("java.lang.Object");

    TypeDesc J_L_OBJECT_ARRAY = new TypeDescImpl("java.lang.Object", 1);

    TypeDesc J_L_CLASS = new TypeDescImpl("java.lang.Class");

    TypeDesc J_L_STRING = new TypeDescImpl("java.lang.String");

    //

    boolean isArray();

    int dimensions();

    TypeDesc componentType();

    TypeDesc rawType();

    boolean hasTypeArguments();

    List<TypeDesc> typeArguments();

    // Conversions

    TypeDesc toBasicType();

    String toClassName();

    String toInternalName();

    String toNominalDescriptorString();

    default ClassDesc toNominalDescriptor() {
        return ClassDesc.ofDescriptor(toNominalDescriptorString());
    }

    default Class<?> resolve(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return (Class<?>) toNominalDescriptor().resolveConstantDesc(l);
    }

    // Factories

    static TypeDesc type(Class<?> c) {
        int dims = 0;
        if (c.isArray()) {
            while (c.isArray()) {
                c = c.getComponentType();
                dims++;
            }
        }
        return new TypeDescImpl(c.getName(), dims);
    }

    static TypeDesc type(Class<?> c, Class<?>... typeArguments) {
        return type(c, List.of(typeArguments));
    }

    static TypeDesc type(Class<?> c, List<Class<?>> typeArguments) {
        int dims = 0;
        if (c.isArray()) {
            while (c.isArray()) {
                c = c.getComponentType();
                dims++;
            }
        }
        return new TypeDescImpl(c.getName(), dims, typeArguments.stream().map(TypeDesc::type).toList());
    }

    static TypeDesc ofNominalDescriptor(ClassDesc d) {
        String descriptor = d.descriptorString();
        int i = 0;
        while (descriptor.charAt(i) == '[') {
            i++;
        }
        int dims = i;

        TypeDesc td = switch (descriptor.charAt(i)) {
            case 'V' -> TypeDesc.VOID;
            case 'I' -> TypeDesc.INT;
            case 'J' -> TypeDesc.LONG;
            case 'C' -> TypeDesc.CHAR;
            case 'S' -> TypeDesc.SHORT;
            case 'B' -> TypeDesc.BYTE;
            case 'F' -> TypeDesc.FLOAT;
            case 'D' -> TypeDesc.DOUBLE;
            case 'Z' -> TypeDesc.BOOLEAN;
            case 'L' -> {
                // La.b.c.Class;
                String typeName = descriptor.substring(i + 1, descriptor.length() - 1).replace('/', '.');
                yield new TypeDescImpl(typeName, 0);
            }
            default -> throw new InternalError();
        };

        return TypeDesc.type(td, dims);
    }

    static TypeDesc ofNominalDescriptorString(String d) {
        return ofNominalDescriptor(ClassDesc.ofDescriptor(d));
    }

    static TypeDesc type(TypeDesc t, TypeDesc... typeArguments) {
        return type(t, List.of(typeArguments));
    }

    static TypeDesc type(TypeDesc t, List<TypeDesc> typeArguments) {
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        TypeDescImpl timpl = (TypeDescImpl) t;
        return new TypeDescImpl(timpl.type, timpl.dims, typeArguments);
    }

    static TypeDesc type(TypeDesc t, int dims, TypeDesc... typeArguments) {
        return type(t, dims, List.of(typeArguments));
    }

    static TypeDesc type(TypeDesc t, int dims, List<TypeDesc> typeArguments) {
        if (t.isArray()) {
            throw new IllegalArgumentException("Type descriptor must not be an array: " + t);
        }
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        TypeDescImpl timpl = (TypeDescImpl) t;
        return new TypeDescImpl(timpl.type, dims, typeArguments);
    }

    // Copied code in jdk.compiler module throws UOE
    static TypeDesc ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseTypeDesc(s);
    }
}
