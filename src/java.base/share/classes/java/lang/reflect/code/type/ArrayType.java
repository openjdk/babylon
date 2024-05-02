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

/**
 * An array type.
 */
public final class ArrayType implements JavaType {
    static final String NAME = "[";

    final JavaType componentType;

    ArrayType(JavaType componentType) {
        this.componentType = componentType;
    }

    /**
     * {@return the array type's component type}
     */
    public JavaType componentType() {
        return componentType;
    }

    /**
     * {@return the dimensions associated with this array type}
     */
    public int dimensions() {
        int dims = 0;
        JavaType current = this;
        while (current instanceof ArrayType at) {
            dims++;
            current = at.componentType();
        }
        return dims;
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        int dims = 0;
        TypeElement current = this;
        while (current instanceof ArrayType at) {
            dims++;
            current = at.componentType();
        }
        return new TypeDefinition("[".repeat(dims), List.of(current.toTypeDefinition()));
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof ArrayType that &&
                componentType.equals(that.componentType);
    }

    @Override
    public int hashCode() {
        return 17 * componentType.hashCode();
    }

    @Override
    public JavaType erasure() {
        return JavaType.array(componentType.erasure());
    }

    @Override
    public JavaType toBasicType() {
        return JavaType.J_L_OBJECT;
    }

    @Override
    public ClassDesc toNominalDescriptor() {
        return componentType.toNominalDescriptor().arrayType();
    }
}
