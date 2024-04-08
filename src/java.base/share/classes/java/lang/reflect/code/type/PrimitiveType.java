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

import java.util.List;
import java.util.Map;

/**
 * A primitive type.
 */
final class PrimitiveType implements JavaType {
    // Fully qualified name
    private final String type;

    PrimitiveType(String type) {
        this.type = type;
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(type, List.of());
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        PrimitiveType typeDesc = (PrimitiveType) o;

        return type.equals(typeDesc.type);
    }

    @Override
    public int hashCode() {
        return type.hashCode();
    }

    @Override
    public boolean isArray() {
        return false;
    }

    @Override
    public boolean isPrimitive() {
        return true;
    }

    @Override
    public boolean isClass() {
        return false;
    }

    @Override
    public JavaType toBasicType() {
        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        return switch (bytecodeKind) {
            case 'V' -> JavaType.VOID;
            case 'J' -> JavaType.LONG;
            case 'F' -> JavaType.FLOAT;
            case 'D' -> JavaType.DOUBLE;
            default -> JavaType.INT;
        };
    }

    @Override
    public String toNominalDescriptorString() {
        return toBytecodeDescriptor(type);
    }

    static String toBytecodeDescriptor(String type) {
        Character bytecodeKind = PRIMITIVE_TYPE_MAP.get(type);
        return bytecodeKind.toString();
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
