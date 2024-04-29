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

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Map;

/**
 * A class type.
 */
public final class ClassType implements JavaType {
    // Fully qualified name
    private final String type;

    private final List<JavaType> typeArguments;

    ClassType(String type) {
        this(type, List.of());
    }

    ClassType(String type, List<JavaType> typeArguments) {
        switch (type) {
            case "boolean", "char", "byte", "short", "int", "long",
                    "float", "double", "void" -> throw new IllegalArgumentException();
        }
        this.type = type;
        this.typeArguments = List.copyOf(typeArguments);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        List<TypeDefinition> args = typeArguments.stream()
                .map(TypeElement::toTypeDefinition)
                .toList();

        TypeDefinition td = new TypeDefinition(type, args);
        return td;
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ClassType typeDesc = (ClassType) o;

        if (!type.equals(typeDesc.type)) return false;
        return typeArguments.equals(typeDesc.typeArguments);
    }

    @Override
    public int hashCode() {
        int result = type.hashCode();
        result = 31 * result + typeArguments.hashCode();
        return result;
    }

    @Override
    public JavaType erasure() {
        return rawType();
    }

    // Conversions

    public ClassType rawType() {
        return new ClassType(type);
    }

    public boolean hasTypeArguments() {
        return !typeArguments.isEmpty();
    }

    public List<JavaType> typeArguments() {
        return typeArguments;
    }

    @Override
    public JavaType toBasicType() {
        return JavaType.J_L_OBJECT;
    }

    public String toClassName() {
        return type;
    }

    public String toInternalName() {
        return toClassDescriptor(type);
    }

    @Override
    public String toNominalDescriptorString() {
        return toBytecodeDescriptor(type);
    }

    static String toBytecodeDescriptor(String type) {
        if (type.equals("null")) {
            type = Object.class.getName();
        }

        return "L" + type.replace('.', '/') + ";";
    }

    static String toClassDescriptor(String type) {
        return type.replace('.', '/');
    }
}
