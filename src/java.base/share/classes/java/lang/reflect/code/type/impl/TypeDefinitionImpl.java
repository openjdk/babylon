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

import java.lang.reflect.code.type.TypeDefinition;
import java.util.List;
import java.util.stream.Collectors;

public final class TypeDefinitionImpl implements TypeDefinition {
    // Fully qualified name
    public final String type;

    public final int dims;

    public final List<TypeDefinition> typeArguments;

    public TypeDefinitionImpl(String type) {
        this(type, 0, List.of());
    }

    public TypeDefinitionImpl(String type, int dim) {
        this(type, dim, List.of());
    }

    public TypeDefinitionImpl(String type, int dims, List<TypeDefinition> typeArguments) {
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
            String params = typeArguments.stream().map(TypeDefinition::toString).collect(Collectors.joining(", ", "<", ">"));
            return type + params + "[]".repeat(dims);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        TypeDefinitionImpl typeDesc = (TypeDefinitionImpl) o;

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
    public String name() {
        return type;
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
    public TypeDefinition componentType() {
        if (!isArray()) {
            return null;
        }

        return new TypeDefinitionImpl(type, dims - 1, List.of());
    }

    @Override
    public TypeDefinitionImpl rawType() {
        return new TypeDefinitionImpl(type, dims);
    }

    @Override
    public boolean hasTypeArguments() {
        return !typeArguments.isEmpty();
    }

    @Override
    public List<TypeDefinition> typeArguments() {
        return typeArguments;
    }

}
