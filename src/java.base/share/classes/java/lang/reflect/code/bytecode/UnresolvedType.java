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

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.TypeWithComponent;
import java.lang.reflect.code.type.ArrayType;
import java.lang.reflect.code.type.JavaType;
import java.util.List;

sealed interface UnresolvedType extends TypeElement {

    static Ref unresolvedRef() {
        return new Ref();
    }

    static Int unresolvedInt() {
        return new Int();
    }

    JavaType resolved();

    boolean resolveTo(TypeElement type);
    boolean resolveFrom(TypeElement type);

    static final class Ref implements UnresolvedType, TypeWithComponent {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_REF = new TypeElement.ExternalizedTypeElement("?REF", List.of());

        private JavaType resolved;

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return resolved == null ? UNRESOLVED_REF : resolved.externalize();
        }

        @Override
        public boolean resolveTo(TypeElement type) {
            if (resolved == null || resolved.equals(JavaType.J_L_OBJECT)) {
                if (type instanceof UnresolvedType utt) {
                    type = utt.resolved();
                }
                if (type != null && !type.equals(resolved)) {
                    resolved = (JavaType)type;
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean resolveFrom(TypeElement type) {
            if (resolved == null || resolved.equals(JavaType.J_L_OBJECT)) {
                if (type instanceof UnresolvedType utt) {
                    type = utt.resolved();
                }
                // Only care about arrays
                if (type instanceof ArrayType at) {
                    resolved = at;
                    return true;
                }
            }
            return false;
        }

        @Override
        public JavaType resolved() {
            return resolved;
        }

        @Override
        public TypeElement componentType() {
            return resolved == null ? new Comp(this) : ((TypeWithComponent)resolved).componentType();
        }
    }

    static final class Comp implements  UnresolvedType, TypeWithComponent {

        private final UnresolvedType array;

        Comp(UnresolvedType array) {
            this.array = array;
        }

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            var res = resolved();
            return res == null ? new TypeElement.ExternalizedTypeElement("?COMP", List.of(array.externalize())) : res.externalize();
        }

        @Override
        public boolean resolveTo(TypeElement type) {
            return false;
        }

        @Override
        public boolean resolveFrom(TypeElement type) {
            return false;
        }

        @Override
        public JavaType resolved() {
            return array.resolved() instanceof ArrayType at ? at.componentType() : null;
        }

        @Override
        public TypeElement componentType() {
            return new Comp(this);
        }
    }

    static final class Int implements  UnresolvedType {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_INT = new TypeElement.ExternalizedTypeElement("?INT", List.of());

        private int resolved = -1;

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return resolved < 0 ? UNRESOLVED_INT : resolved().externalize();
        }

        @Override
        public boolean resolveFrom(TypeElement type) {
            // Only care about booleans
            if (resolved < 4) {
                if (type instanceof UnresolvedType utt) {
                    type = utt.resolved();
                }
                if (JavaType.BOOLEAN.equals(type)) {
                    resolved = 4;
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean resolveTo(TypeElement type) {
            if (resolved < 4) {
                if (type instanceof UnresolvedType utt) {
                    type = utt.resolved();
                }
                if (type != null) {
                    int p = TYPES.indexOf(type);
                    if (p > resolved) {
                        resolved = p;
                        return true;
                    }
                }
            }
            return false;
        }

        @Override
        public JavaType resolved() {
            return resolved >=0 ? TYPES.get(resolved) : null;
        }
    }

    static Object convertValue(UnresolvedType ut, Object value) {
        return switch (TYPES.indexOf(ut.resolved())) {
            case 0 -> toNumber(value).intValue();
            case 1 -> (char)toNumber(value).intValue();
            case 2 -> toNumber(value).shortValue();
            case 3 -> toNumber(value).byteValue();
            case 4 -> value instanceof Number n ? n.intValue() != 0 : (Boolean)value;
            default -> value;
        };
    }

    static final List<JavaType> TYPES = List.of(JavaType.INT, JavaType.CHAR, JavaType.SHORT, JavaType.BYTE, JavaType.BOOLEAN);

    private static Number toNumber(Object value) {
        return switch (value) {
            case Boolean b -> b ? 1 : 0;
            case Character c -> (int)c;
            case Number n -> n;
            default -> throw new IllegalStateException("Unexpected " + value);
        };
    }
}
