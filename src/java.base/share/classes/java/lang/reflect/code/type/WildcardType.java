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
import java.util.List;
import java.util.Objects;

/**
 * A wildcard type.
 */
public final class WildcardType implements JavaType {

    final BoundKind kind;
    final JavaType boundType;

    WildcardType(BoundKind kind, JavaType boundType) {
        this.kind = kind;
        this.boundType = boundType;
    }

    /**
     * {@return the wildcard type's bound type}
     */
    public JavaType boundType() {
        return boundType;
    }

    /**
     * {@return the wildcard type's bound kind}
     */
    public BoundKind boundKind() {
        return kind;
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        String prefix = kind == BoundKind.EXTENDS ? "+" : "-";
        return new TypeDefinition(prefix, List.of(boundType.toTypeDefinition()));
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof WildcardType that &&
                kind.equals(that.kind) &&
                boundType.equals(that.boundType);
    }

    @Override
    public int hashCode() {
        return Objects.hash(boundType, kind);
    }

    @Override
    public JavaType erasure() {
        throw new UnsupportedOperationException("Wildcard type");
    }

    @Override
    public JavaType toBasicType() {
        throw new UnsupportedOperationException("Wildcard type");
    }

    @Override
    public ClassDesc toNominalDescriptor() {
        throw new UnsupportedOperationException("Wildcard type");
    }

    public enum BoundKind {
        EXTENDS,
        SUPER
    }
}
