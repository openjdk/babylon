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

import java.lang.reflect.code.type.impl.TypeDefinitionImpl;
import java.util.List;

/**
 * The general symbolic description of a type.
 */
public sealed interface TypeDefinition permits TypeDefinitionImpl {

    TypeDefinition VOID = new TypeDefinitionImpl("void");

    //

    String name();

    boolean isArray();

    int dimensions();

    TypeDefinition componentType();

    TypeDefinition rawType();

    boolean hasTypeArguments();

    List<TypeDefinition> typeArguments();

    // Factories

    static TypeDefinition type(TypeDefinition t, TypeDefinition... typeArguments) {
        return type(t, List.of(typeArguments));
    }

    static TypeDefinition type(TypeDefinition t, List<TypeDefinition> typeArguments) {
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        TypeDefinitionImpl timpl = (TypeDefinitionImpl) t;
        return new TypeDefinitionImpl(timpl.type, timpl.dims, typeArguments);
    }

    static TypeDefinition type(TypeDefinition t, int dims, TypeDefinition... typeArguments) {
        return type(t, dims, List.of(typeArguments));
    }

    static TypeDefinition type(TypeDefinition t, int dims, List<TypeDefinition> typeArguments) {
        if (t.isArray()) {
            throw new IllegalArgumentException("Type descriptor must not be an array: " + t);
        }
        if (t.hasTypeArguments()) {
            throw new IllegalArgumentException("Type descriptor must not have type arguments: " + t);
        }
        TypeDefinitionImpl timpl = (TypeDefinitionImpl) t;
        return new TypeDefinitionImpl(timpl.type, dims, typeArguments);
    }

    // Copied code in jdk.compiler module throws UOE
    static TypeDefinition ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseTypeDesc(s);
    }
}
