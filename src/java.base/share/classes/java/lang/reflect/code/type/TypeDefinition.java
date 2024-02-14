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

import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.stream.Collectors;

/**
 * A type in general symbolic structured form.
 * <p>
 * A type definition can be converted to an instance of a type, a {@link TypeElement type element}, using
 * a {@link TypeElementFactory}.
 */
public record TypeDefinition(String identifier, List<TypeDefinition> arguments) {

    public static final TypeDefinition VOID = new TypeDefinition("void", List.of());

    public TypeDefinition {
        arguments = List.copyOf(arguments);
    }

    @Override
    public String toString() {
        return toString(this);
    }

    static String toString(TypeDefinition t) {
        int dimensions = dimensions(t.identifier);
        if (dimensions > 0 && t.arguments.size() == 1) {
            t = t.arguments.getFirst();
        }

        if (dimensions == 0 && t.arguments.isEmpty()) {
            return t.identifier;
        } else {
            StringBuilder s = new StringBuilder();
            s.append(t.identifier);
            if (!t.arguments.isEmpty()) {
                String args = t.arguments.stream()
                        .map(Object::toString)
                        .collect(Collectors.joining(", ", "<", ">"));
                s.append(args);
            }

            if (dimensions > 0) {
                s.append("[]".repeat(dimensions));
            }

            return s.toString();
        }
    }

    static int dimensions(String identifier) {
        if (!identifier.isEmpty() && identifier.charAt(0) == '[') {
            for (int i = 1; i < identifier.length(); i++) {
                if (identifier.charAt(i) != '[') {
                    return 0;
                }
            }
            return identifier.length();
        } else {
            return 0;
        }
    }

    // Factories

    // Copied code in jdk.compiler module throws UOE
    public static TypeDefinition ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseTypeDefinition(s);
    }
}
