/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.extern.impl;

import java.lang.reflect.Array;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.java.impl.JavaTypeUtils;
import jdk.incubator.code.extern.ExternalizedOp;

public final class AttributeMapper {

    private AttributeMapper() {
    }

    public static String toString(Object value) {
        if (value == ExternalizedOp.NULL_ATTRIBUTE_VALUE) {
            return "null";
        }
        StringBuilder sb = new StringBuilder();
        toString(value, sb);
        return sb.toString();
    }

    static void toString(Object o, StringBuilder sb) {
        if (o.getClass().isArray()) {
            // note, while we can't parse back the array representation, this might be useful
            // for non-externalizable ops that want better string representation of array attribute values (e.g. ONNX)
            arrayToString(o, sb);
        } else {
            switch (o) {
                case Integer i -> sb.append(i);
                case Long l -> sb.append(l).append('L');
                case Float f -> sb.append(f).append('f');
                case Double d -> sb.append(d).append('d');
                case Character c -> sb.append('\'').append(c).append('\'');
                case Boolean b -> sb.append(b);
                case TypeElement te -> sb.append(JavaTypeUtils.flatten(te.externalize()));
                default -> {
                    // fallback to a string
                    sb.append('"');
                    quote(o.toString(), sb);
                    sb.append('"');
                }
            }
        }
    }

    static void arrayToString(Object a, StringBuilder sb) {
        boolean first = true;
        sb.append("[");
        for (int i = 0; i < Array.getLength(a); i++) {
            if (!first) {
                sb.append(", ");
            }
            toString(Array.get(a, i), sb);
            first = false;
        }
        sb.append("]");
    }

    static void quote(String s, StringBuilder sb) {
        for (int i = 0; i < s.length(); i++) {
            sb.append(quote(s.charAt(i)));
        }
    }

    /**
     * Escapes a character if it has an escape sequence or is
     * non-printable ASCII.  Leaves non-ASCII characters alone.
     */
    // Copied from com.sun.tools.javac.util.Convert
    static String quote(char ch) {
        return switch (ch) {
            case '\b' -> "\\b";
            case '\f' -> "\\f";
            case '\n' -> "\\n";
            case '\r' -> "\\r";
            case '\t' -> "\\t";
            case '\'' -> "\\'";
            case '\"' -> "\\\"";
            case '\\' -> "\\\\";
            default -> (isPrintableAscii(ch)) ? String.valueOf(ch) : String.format("\\u%04x", (int) ch);
        };
    }

    /**
     * Is a character printable ASCII?
     */
    static boolean isPrintableAscii(char ch) {
        return ch >= ' ' && ch <= '~';
    }

}
