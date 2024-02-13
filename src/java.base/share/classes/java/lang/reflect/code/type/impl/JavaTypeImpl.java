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

package java.lang.reflect.code.type.impl;

import java.lang.reflect.code.type.JavaType;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public final class JavaTypeImpl implements JavaType {
    // Fully qualified name
    public final String type;

    public final int dims;

    public final List<JavaType> typeArguments;

    public JavaTypeImpl(String type) {
        this(type, 0, List.of());
    }

    public JavaTypeImpl(String type, int dim) {
        this(type, dim, List.of());
    }

    public JavaTypeImpl(String type, int dims, List<JavaType> typeArguments) {
        this.type = type;
        this.dims = dims;
        this.typeArguments = List.copyOf(typeArguments);
    }

    @Override
    public String toString() {
        if (dims == 0 && typeArguments.isEmpty()) {
            return type;
        } else if (typeArguments.isEmpty()) {
            return type + "[]".repeat(dims);
        } else {
            String params = typeArguments.stream().map(JavaType::toString).collect(Collectors.joining(", ", "<", ">"));
            return type + params + "[]".repeat(dims);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        JavaTypeImpl typeDesc = (JavaTypeImpl) o;

        if (dims != typeDesc.dims) return false;
        if (!type.equals(typeDesc.type)) return false;
        return typeArguments.equals(typeDesc.typeArguments);
    }

    @Override
    public int hashCode() {
        int result = type.hashCode();
        result = 31 * result + dims;
        result = 31 * result + typeArguments.hashCode();
        return result;
    }

    @Override
    public boolean isArray() {
        return dims != 0;
    }

    @Override
    public int dimensions() {
        return dims;
    }

    @Override
    public JavaType componentType() {
        if (!isArray()) {
            return null;
        }

        return new JavaTypeImpl(type, dims - 1, List.of());
    }

    @Override
    public JavaTypeImpl rawType() {
        return new JavaTypeImpl(type, dims);
    }

    @Override
    public boolean hasTypeArguments() {
        return !typeArguments.isEmpty();
    }

    @Override
    public List<JavaType> typeArguments() {
        return typeArguments;
    }

    // Conversions

    @Override
    public JavaType toBasicType() {
        if (isArray()) {
            return JavaType.J_L_OBJECT;
        }

        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        if (bytecodeKind == null) {
            return JavaType.J_L_OBJECT;
        } else {
            return switch (bytecodeKind) {
                case 'V' -> JavaType.VOID;
                case 'J' -> JavaType.LONG;
                case 'F' -> JavaType.FLOAT;
                case 'D' -> JavaType.DOUBLE;
                default -> JavaType.INT;
            };
        }
    }

    @Override
    public String toClassName() {
        if (isArray()) {
            throw new IllegalStateException("Array type cannot be converted to class name: " + type);
        }

        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        if (bytecodeKind != null) {
            throw new IllegalStateException("Invalid class: " + type);
        }

        return type;
    }

    @Override
    public String toInternalName() {
        if (isArray()) {
            throw new IllegalArgumentException("Array type cannot be converted to class descriptor");
        }

        return toClassDescriptor(type);
    }

    @Override
    public String toNominalDescriptorString() {
        if (!isArray()) {
            return toBytecodeDescriptor(type);
        }

        String arraySignature = "[".repeat(dims);
        return arraySignature + toBytecodeDescriptor(type);
    }

    static String toBytecodeDescriptor(String type) {
        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        if (bytecodeKind != null) {
            return bytecodeKind.toString();
        }

        if (type.equals("null")) {
            type = Object.class.getName();
        }

        return "L" + type.replace('.', '/') + ";";
    }

    static String toClassDescriptor(String type) {
        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        if (bytecodeKind != null) {
            throw new IllegalArgumentException("Primitive type has no class descriptor");
        }

        return type.replace('.', '/');
    }

    static Map<String, Character> PRIMITIVE_TYPE_MAP;

    static {
        PRIMITIVE_TYPE_MAP = Map.of(
                "boolean", 'Z',
                "byte", 'B',
                "short", 'S',
                "char", 'C',
                "int", 'I',
                "long", 'J',
                "float", 'F',
                "double", 'D',
                "void", 'V'
        );
    }
}
