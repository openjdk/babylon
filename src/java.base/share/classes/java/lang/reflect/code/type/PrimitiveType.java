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
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.reflect.Type;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A primitive type.
 */
public final class PrimitiveType implements JavaType {
    // Fully qualified name
    private final ClassDesc type;

    PrimitiveType(ClassDesc type) {
        this.type = type;
    }

    @Override
    public Type resolve(Lookup lookup) throws ReflectiveOperationException {
        return type.resolveConstantDesc(lookup);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(type.displayName(), List.of());
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
    public JavaType erasure() {
        return this;
    }

    @Override
    public JavaType toBasicType() {
        return switch (type.descriptorString().charAt(0)) {
            case 'V' -> JavaType.VOID;
            case 'J' -> JavaType.LONG;
            case 'F' -> JavaType.FLOAT;
            case 'D' -> JavaType.DOUBLE;
            default -> JavaType.INT;
        };
    }

    /**
     * {@return the boxed class type associated with this primitive type (if any)}
     */
    public Optional<ClassType> box() {
        class LazyHolder {
            static final Map<PrimitiveType, ClassType> primitiveToWrapper = Map.of(
                    BYTE, J_L_BYTE,
                    SHORT, J_L_SHORT,
                    INT, J_L_INTEGER,
                    LONG, J_L_LONG,
                    FLOAT, J_L_FLOAT,
                    DOUBLE, J_L_DOUBLE,
                    CHAR, J_L_CHARACTER,
                    BOOLEAN, J_L_BOOLEAN
            );
        }
        return Optional.ofNullable(LazyHolder.primitiveToWrapper.get(this));
    };

    @Override
    public ClassDesc toNominalDescriptor() {
        return type;
    }

    /**
     * {@return {@code true} if this type is {@link JavaType#VOID}}
     */
    public boolean isVoid() {
        return toBasicType() == JavaType.VOID;
    }
}
