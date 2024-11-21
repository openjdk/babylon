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

package jdk.incubator.code.bytecode;

import jdk.incubator.code.TypeElement;
import java.util.List;

sealed interface UnresolvedType extends TypeElement {

    static Ref unresolvedRef() {
        return new Ref();
    }

    static Int unresolvedInt() {
        return new Int();
    }

    static final class Ref implements UnresolvedType {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_REF = new TypeElement.ExternalizedTypeElement("?REF", List.of());

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return UNRESOLVED_REF;
        }
    }

    static final class Int implements  UnresolvedType {
        private static final TypeElement.ExternalizedTypeElement UNRESOLVED_INT = new TypeElement.ExternalizedTypeElement("?INT", List.of());

        @Override
        public TypeElement.ExternalizedTypeElement externalize() {
            return UNRESOLVED_INT;
        }
    }
}
