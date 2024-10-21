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

package java.lang.reflect.code.bytecode;

import java.lang.classfile.TypeKind;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;
import java.util.List;

sealed abstract class UnresolvedType implements TypeElement {

    static Ref unresolvedRef() {
        return new Ref();
    }

    static Int unresolvedInt() {
        return new Int();
    }

    static final class Ref extends UnresolvedType {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_REF = new TypeElement.ExternalizedTypeElement("?REF", List.of());

        TypeElement resolved;

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return UNRESOLVED_REF;
        }

        @Override
        TypeElement resolved() {
            return resolved == null ? JavaType.J_L_OBJECT : resolved;
        }

        @Override
        Object convertValue(Object value) {
            return value;
        }
    }

    static final class Int extends UnresolvedType {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_INT = new TypeElement.ExternalizedTypeElement("?INT", List.of());

        TypeKind resolved;

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return UNRESOLVED_INT;
        }

        @Override
        public TypeElement resolved() {
            return switch (resolved) {
                case BOOLEAN -> JavaType.BOOLEAN;
                case BYTE -> JavaType.BYTE;
                case CHAR -> JavaType.CHAR;
                case INT -> JavaType.INT;
                case SHORT -> JavaType.SHORT;
                case null -> JavaType.INT;
                default -> throw new IllegalStateException("Unexpected " + resolved);
            };
        }

        @Override
        Object convertValue(Object value) {
            return switch (resolved) {
                case BOOLEAN -> value instanceof Number n ? n.intValue() != 0 : (Boolean)value;
                case BYTE -> toNumber(value).byteValue();
                case CHAR -> (char)toNumber(value).intValue();
                case INT -> toNumber(value).intValue();
                case SHORT -> toNumber(value).shortValue();
                case null -> toNumber(value).intValue();
                default -> throw new IllegalStateException("Unexpected " + resolved);
            };
        }

        static Number toNumber(Object value) {
            return value instanceof Boolean b ? b ? 1 : 0 : (Number)value;
        }
    }

    // Support for UnresolvedTypesTransformer

    abstract TypeElement resolved();

    abstract Object convertValue(Object value);
}
