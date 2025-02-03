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

import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;

/**
 * This class provides static methods for producing and manipulating a {@link JsonValue}.
 * <p>
 * {@link #parse(String)} and {@link #parse(char[])} produce a {@code JsonValue}
 * by parsing data adhering to the JSON syntax defined in RFC 8259.
 * <p>
 * {@link #toDisplayString(JsonValue)} is a formatter that produces a
 * representation of the JSON value suitable for display.
 * <p>
 * {@link #fromUntyped(Object)} and {@link #toUntyped(JsonValue)} provide a conversion
 * between {@code JsonValue} and an untyped object.
 *
 * <table id="mapping-table" class="striped">
 * <caption>Mapping Table</caption>
 * <thead>
 *    <tr>
 *       <th scope="col" class="TableHeadingColor">JsonValue</th>
 *       <th scope="col" class="TableHeadingColor">Untyped Object</th>
 *    </tr>
 * </thead>
 * <tbody>
     * <tr>
     *     <th>{@code List<Object>}</th>
     *     <th> {@code JsonArray}</th>
     * </tr>
     * <tr>
     *     <th>{@code Boolean}</th>
     *     <th>{@code JsonBoolean}</th>
     * </tr>
     * <tr>
     *     <th>{@code `null`}</th>
     *     <th> {@code JsonNull}</th>
     * </tr>
     * <tr>
     *     <th>{@code Number}</th>
     *     <th>{@code JsonNumber}</th>
     * </tr>
     * <tr>
     *     <th>{@code Map<String, Object>}</th>
     *     <th> {@code JsonObject}</th>
     * </tr>
     * <tr>
     *     <th>{@code String}</th>
     *     <th>{@code JsonString}</th>
     * </tr>
 * </tbody>
 * </table>
 *
 * @implSpec The reference implementation defines a {@code JsonValue} nesting
 * depth limit of 32. Attempting to construct a {@code JsonValue} that exceeds this limit
 * will throw an {@code IllegalArgumentException}.
 *
 * @spec https://datatracker.ietf.org/doc/html/rfc8259 RFC 8259: The JavaScript
 *          Object Notation (JSON) Data Interchange Format
 * 
 */
//@PreviewFeature(feature = PreviewFeature.Feature.JSON)
public final class Json {

    // Depth limit used by Parser and Generator
    static final int MAX_DEPTH = 32;

    /**
     * Parses and creates the top level {@code JsonValue} in this JSON
     * document. If the document contains any JSON Object that has
     * duplicate keys, a {@code JsonParseException} is thrown.
     *
     * @param in the input JSON document as {@code String}. Non-null.
     * @throws JsonParseException if the input JSON document does not conform
     *      to the JSON document format, a JSON object containing
     *      duplicate keys is encountered, or a nest limit is exceeded.
     * @return the top level {@code JsonValue}
     */
    public static JsonValue parse(String in) {
        Objects.requireNonNull(in);
        return JsonGenerator.createValue(JsonParser.parseRoot(
                new JsonDocumentInfo(in.toCharArray())), 0, 0);
    }

    /**
     * Parses and creates the top level {@code JsonValue} in this JSON
     * document. If the document contains any JSON Object that has
     * duplicate keys, a {@code JsonParseException} is thrown.
     *
     * @param in the input JSON document as {@code char[]}. Non-null.
     * @throws JsonParseException if the input JSON document does not conform
     *      to the JSON document format, a JSON object containing
     *      duplicate keys is encountered, or a nest limit is exceeded.
     * @return the top level {@code JsonValue}
     */
    public static JsonValue parse(char[] in) {
        Objects.requireNonNull(in);
        return JsonGenerator.createValue(JsonParser.parseRoot(
                new JsonDocumentInfo(Arrays.copyOf(in, in.length))), 0, 0);
    }

    /**
     * {@return a {@code JsonValue} corresponding to {@code src}}
     * See the {@link ##mapping-table Mapping Table} for conversion details.
     *
     * <p>If {@code src} contains a circular reference, {@code IllegalArgumentException}
     * will be thrown. For example, the following code throws an exception,
     * {@snippet lang=java:
     *     var map = new HashMap<String, Object>();
     *     map.put("foo", false);
     *     map.put("bar", map);
     *     Json.fromUntyped(map);
     * }
     *
     * @param src the data to produce the {@code JsonValue} from. May be null.
     * @throws IllegalArgumentException if {@code src} cannot be converted
     *      to {@code JsonValue}, contains a circular reference, or exceeds a nesting limit.
     * @see ##mapping-table Mapping Table
     * @see #toUntyped(JsonValue)
     */
    public static JsonValue fromUntyped(Object src) {
        if (src instanceof JsonValue jv) {
            return jv; // If root is JV, no need to check depth
        } else {
            return JsonGenerator.fromUntyped(
                    src, Collections.newSetFromMap(new IdentityHashMap<>()), 0);
        }
    }

    /**
     * {@return an {@code Object} corresponding to {@code src}}
     * See the {@link ##mapping-table Mapping Table} for conversion details.
     *
     * @param src the {@code JsonValue} to convert to untyped. Non-null.
     * @see ##mapping-table Mapping Table
     * @see #fromUntyped(Object)
     */
    public static Object toUntyped(JsonValue src) {
        Objects.requireNonNull(src);
        return ((JsonValueImpl)src).toUntyped();
    }

    /**
     * {@return the String representation of the given {@code JsonValue} that conforms
     * to the JSON syntax} As opposed to {@link JsonValue#toString()}, this method returns
     * a JSON string that is suitable for display.
     *
     * @param value the {@code JsonValue} to create the display string from. Non-null.
     */
    public static String toDisplayString(JsonValue value) {
        Objects.requireNonNull(value);
        return ((JsonValueImpl)value).toDisplayString();
    }

    // no instantiation is allowed for this class
    private Json() {}
}
