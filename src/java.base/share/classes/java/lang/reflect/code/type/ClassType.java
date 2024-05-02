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
import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A class type.
 */
public final class ClassType implements TypeVarRef.Owner, JavaType {
    // Fully qualified name
    private final ClassDesc type;

    private final List<JavaType> typeArguments;

    ClassType(ClassDesc type) {
        this(type, List.of());
    }

    ClassType(ClassDesc type, List<JavaType> typeArguments) {
        if (!type.isClassOrInterface()) {
            throw new IllegalArgumentException("Invalid base type: " + type);
        }
        this.type = type;
        this.typeArguments = List.copyOf(typeArguments);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        List<TypeDefinition> args = typeArguments.stream()
                .map(TypeElement::toTypeDefinition)
                .toList();

        TypeDefinition td = new TypeDefinition(toClassName(), args);
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

    /**
     * {@return the unboxed primitive type associated with this class type (if any)}
     */
    public Optional<PrimitiveType> unbox() {
        class LazyHolder {
            static final Map<ClassType, PrimitiveType> wrapperToPrimitive = Map.of(
                    J_L_BYTE, BYTE,
                    J_L_SHORT, SHORT,
                    J_L_INTEGER, INT,
                    J_L_LONG, LONG,
                    J_L_FLOAT, FLOAT,
                    J_L_DOUBLE, DOUBLE,
                    J_L_CHARACTER, CHAR,
                    J_L_BOOLEAN, BOOLEAN
            );
        }
        return Optional.ofNullable(LazyHolder.wrapperToPrimitive.get(this));
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
        String pkg = type.packageName();
        return pkg.isEmpty() ?
                type.displayName() :
                String.format("%s.%s", pkg, type.displayName());
    }

    public String toInternalName() {
        return toClassName().replace('.', '/');
    }

    @Override
    public ClassDesc toNominalDescriptor() {
        return type;
    }
}
