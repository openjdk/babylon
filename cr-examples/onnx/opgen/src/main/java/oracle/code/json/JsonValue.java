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

package oracle.code.json;

/**
 * The interface that represents a JSON value. {@code JsonValue} is the type returned
 * by a {@link Json#parse(String)}. Valid subtypes are either {@code JsonString},
 * {@code JsonNumber}, {@code JsonObject}, {@code JsonArray}, {@code JsonBoolean},
 * or {@code JsonNull}.
 * <p>
 * See {@link Json#toUntyped(JsonValue)} and {@link Json#fromUntyped(Object)} for converting
 * between a {@code JsonValue} and its corresponding data type. For example,
 * {@snippet lang=java:
 *     var values = Arrays.asList("foo", true, 25);
 *     var json = Json.fromUntyped(values);
 *     Json.toUntyped(json).equals(values); // returns true
 * }
 * See {@link #toString()} for converting a {@code JsonValue}
 * to its corresponding JSON String. For example,
 * {@snippet lang=java:
 *     var values = Arrays.asList("foo", true, 25);
 *     var json = Json.fromUntyped(values);
 *     json.toString(); // returns "[\"foo\",true,25]"
 * }
 *
 * 
 */
public sealed interface JsonValue
        permits JsonString, JsonNumber, JsonObject, JsonArray, JsonBoolean, JsonNull {

    /**
     * {@return the String representation of this {@code JsonValue} that conforms
     * to the JSON syntax} The returned string do not contain any white spaces
     * or newlines to produce a compact representation.
     */
    @Override
    String toString();

    /**
     * Indicates whether the given {@code obj} is "equal to" this {@code JsonValue}.
     *
     * @implSpec The comparison is based on the original document if it was produced by
     * parsing a JSON document.
     */
    @Override
    boolean equals(Object obj);

    // TBD: do we need this override?
    /**
     * {@return the hash code value of this {@code JsonValue}}
     *
     * @implSpec The returned hash code is based on the original document if it was
     * produced by parsing a JSON document.
     */
    @Override
    int hashCode();
}
