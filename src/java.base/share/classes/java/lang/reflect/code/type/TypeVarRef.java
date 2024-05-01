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

/**
 * A type-variable reference.
 */
public final class TypeVarRef implements JavaType {

    final String name;
    final Owner owner;
    final JavaType bound;

    TypeVarRef(String name, Owner owner, JavaType bound) {
        this.name = name;
        this.owner = owner;
        this.bound = bound;
    }

    /**
     * {@return the type-variable name}
     */
    public String name() {
        return name;
    }

    /**
     * {@return the type-variable bound}
     */
    public JavaType bound() {
        return bound;
    }

    /**
     * {@return the owner of this type-variable}
     */
    public Owner owner() {
        return owner;
    }

    @Override
    public JavaType erasure() {
        return bound.erasure();
    }

    @Override
    public ExternalizedCodeType externalize() {
        return new ExternalizedCodeType(String.format("#%s::%s", owner, name),
                List.of(bound.externalize()));
    }

    @Override
    public String toString() {
        return externalize().toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return o instanceof TypeVarRef that &&
                name.equals(that.name) &&
                bound.equals(that.bound);
    }

    @Override
    public int hashCode() {
        return name.hashCode();
    }

    @Override
    public JavaType toBasicType() {
        throw new UnsupportedOperationException("Type var");
    }

    @Override
    public String toNominalDescriptorString() {
        throw new UnsupportedOperationException("Type var");
    }

    /**
     * The owner of a type-variable - either a class or a method.
     */
    public sealed interface Owner permits ClassType, MethodRef { }
}
