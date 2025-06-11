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

import oracle.code.json.impl.JsonParser;
import oracle.code.json.impl.Utils;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

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
 * @spec https://datatracker.ietf.org/doc/html/rfc8259 RFC 8259: The JavaScript
 *      Object Notation (JSON) Data Interchange Format
 * @since 99
 */
public final class Json {

    /**
     * Parses and creates a {@code JsonValue} from the given JSON document.
     * If parsing succeeds, it guarantees that the input document conforms to
     * the JSON syntax. If the document contains any JSON Object that has
     * duplicate names, a {@code JsonParseException} is thrown.
     * <p>
     * {@code JsonValue}s created by this method produce their String and underlying
     * value representation lazily.
     * <p>
     * {@code JsonObject}s preserve the order of their members declared in and parsed from
     * the JSON document.
     *
     * @param in the input JSON document as {@code String}. Non-null.
     * @throws JsonParseException if the input JSON document does not conform
     *      to the JSON document format or a JSON object containing
     *      duplicate names is encountered.
     * @throws NullPointerException if {@code in} is {@code null}
     * @return the parsed {@code JsonValue}
     */
    public static JsonValue parse(String in) {
        Objects.requireNonNull(in);
        return new JsonParser(in.toCharArray()).parseRoot();
    }

    /**
     * Parses and creates a {@code JsonValue} from the given JSON document.
     * If parsing succeeds, it guarantees that the input document conforms to
     * the JSON syntax. If the document contains any JSON Object that has
     * duplicate names, a {@code JsonParseException} is thrown.
     * <p>
     * {@code JsonValue}s created by this method produce their String and underlying
     * value representation lazily.
     * <p>
     * {@code JsonObject}s preserve the order of their members declared in and parsed from
     * the JSON document.
     *
     * @param in the input JSON document as {@code char[]}. Non-null.
     * @throws JsonParseException if the input JSON document does not conform
     *      to the JSON document format or a JSON object containing
     *      duplicate names is encountered.
     * @throws NullPointerException if {@code in} is {@code null}
     * @return the parsed {@code JsonValue}
     */
    public static JsonValue parse(char[] in) {
        Objects.requireNonNull(in);
        return new JsonParser(Arrays.copyOf(in, in.length)).parseRoot();
    }

    /**
     * {@return a {@code JsonValue} created from the given {@code src} object}
     * The mapping from an untyped {@code src} object to a {@code JsonValue}
     * follows the table below.
     * <table class="striped">
     * <caption>Untyped to JsonValue mapping</caption>
     * <thead>
     *    <tr>
     *       <th scope="col" class="TableHeadingColor">Untyped Object</th>
     *       <th scope="col" class="TableHeadingColor">JsonValue</th>
     *    </tr>
     * </thead>
     * <tbody>
     * <tr>
     *     <th>{@code List<Object>}</th>
     *     <th>{@code JsonArray}</th>
     * </tr>
     * <tr>
     *     <th>{@code Boolean}</th>
     *     <th>{@code JsonBoolean}</th>
     * </tr>
     * <tr>
     *     <th>{@code `null`}</th>
     *     <th>{@code JsonNull}</th>
     * </tr>
     * <tr>
     *     <th>{@code Number*}</th>
     *     <th>{@code JsonNumber}</th>
     * </tr>
     * <tr>
     *     <th>{@code Map<String, Object>}</th>
     *     <th>{@code JsonObject}</th>
     * </tr>
     * <tr>
     *     <th>{@code String}</th>
     *     <th>{@code JsonString}</th>
     * </tr>
     * </tbody>
     * </table>
     *
     * <i><sup>*</sup>The supported {@code Number} subclasses are: {@code Byte},
     * {@code Short}, {@code Integer}, {@code Long}, {@code Float},
     * {@code Double}, {@code BigInteger}, and {@code BigDecimal}.</i>
     *
     * <p>If {@code src} is an instance of {@code JsonValue}, it is returned as is.
     * If {@code src} contains a circular reference, {@code IllegalArgumentException}
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
     *      to {@code JsonValue} or contains a circular reference.
     * @see #toUntyped(JsonValue)
     */
    public static JsonValue fromUntyped(Object src) {
        return fromUntyped(src, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    static JsonValue fromUntyped(Object src, Set<Object> identitySet) {
        return switch (src) {
            // Structural: JSON object, JSON array
            case Map<?, ?> map -> {
                if (!identitySet.add(map)) {
                    throw new IllegalArgumentException("Circular reference detected");
                }
                Map<String, JsonValue> m = LinkedHashMap.newLinkedHashMap(map.size());
                for (Map.Entry<?, ?> entry : new LinkedHashMap<>(map).entrySet()) {
                    if (!(entry.getKey() instanceof String strKey)) {
                        throw new IllegalArgumentException("Key is not a String: " + entry.getKey());
                    } else {
                        var unescapedKey = Utils.unescape(
                                strKey.toCharArray(), 0, strKey.length());
                        if (m.containsKey(unescapedKey)) {
                            throw new IllegalArgumentException(
                                    "Duplicate member name: '%s'".formatted(unescapedKey));
                        } else {
                            m.put(unescapedKey, Json.fromUntyped(entry.getValue(), identitySet));
                        }
                    }
                }
                // Bypasses defensive copy in JsonObject.of(m)
                yield Utils.objectOf(m);
            }
            case List<?> list -> {
                if (!identitySet.add(list)) {
                    throw new IllegalArgumentException("Circular reference detected");
                }
                List<JsonValue> l = new ArrayList<>(list.size());
                for (Object o : list) {
                    l.add(Json.fromUntyped(o, identitySet));
                }
                // Bypasses defensive copy in JsonArray.of(l)
                yield Utils.arrayOf(l);
            }
            // JSON primitives
            case String str -> JsonString.of(str);
            case Boolean bool -> JsonBoolean.of(bool);
            case Byte b -> JsonNumber.of(b);
            case Integer i -> JsonNumber.of(i);
            case Long l -> JsonNumber.of(l);
            case Short s -> JsonNumber.of(s);
            case Float f -> JsonNumber.of(f);
            case Double d -> JsonNumber.of(d);
            case BigInteger bi -> JsonNumber.of(bi);
            case BigDecimal bd -> JsonNumber.of(bd);
            case null -> JsonNull.of();
            // JsonValue
            case JsonValue jv -> jv;
            default -> throw new IllegalArgumentException("Type not recognized.");
        };
    }

    /**
     * {@return an {@code Object} created from the given {@code src}
     * {@code JsonValue}} The mapping from a {@code JsonValue} to an
     * untyped {@code src} object follows the table below.
     * <table class="striped">
     * <caption>JsonValue to Untyped mapping</caption>
     * <thead>
     *    <tr>
     *       <th scope="col" class="TableHeadingColor">JsonValue</th>
     *       <th scope="col" class="TableHeadingColor">Untyped Object</th>
     *    </tr>
     * </thead>
     * <tbody>
     * <tr>
     *     <th>{@code JsonArray}</th>
     *     <th>{@code List<Object>}(unmodifiable)</th>
     * </tr>
     * <tr>
     *     <th>{@code JsonBoolean}</th>
     *     <th>{@code Boolean}</th>
     * </tr>
     * <tr>
     *     <th>{@code JsonNull}</th>
     *     <th>{@code `null`}</th>
     * </tr>
     * <tr>
     *     <th>{@code JsonNumber}</th>
     *     <th>{@code Number}</th>
     * </tr>
     * <tr>
     *     <th>{@code JsonObject}</th>
     *     <th>{@code Map<String, Object>}(unmodifiable)</th>
     * </tr>
     * <tr>
     *     <th>{@code JsonString}</th>
     *     <th>{@code String}</th>
     * </tr>
     * </tbody>
     * </table>
     *
     * <p>
     * A {@code JsonObject} in {@code src} is converted to a {@code Map} whose
     * entries occur in the same order as the {@code JsonObject}'s members.
     *
     * @param src the {@code JsonValue} to convert to untyped. Non-null.
     * @throws NullPointerException if {@code src} is {@code null}
     * @see #fromUntyped(Object)
     */
    public static Object toUntyped(JsonValue src) {
        Objects.requireNonNull(src);
        return switch (src) {
            case JsonObject jo -> jo.members().entrySet().stream()
                    .collect(LinkedHashMap::new, // to allow `null` value
                            (m, e) -> m.put(e.getKey(), Json.toUntyped(e.getValue())),
                            HashMap::putAll);
            case JsonArray ja -> ja.values().stream()
                    .map(Json::toUntyped)
                    .toList();
            case JsonBoolean jb -> jb.value();
            case JsonNull _ -> null;
            case JsonNumber n -> n.toNumber();
            case JsonString js -> js.value();
        };
    }

    /**
     * {@return the String representation of the given {@code JsonValue} that conforms
     * to the JSON syntax} As opposed to the compact output returned by {@link
     * JsonValue#toString()}, this method returns a JSON string that is better
     * suited for display.
     *
     * @param value the {@code JsonValue} to create the display string from. Non-null.
     * @throws NullPointerException if {@code value} is {@code null}
     * @see JsonValue#toString()
     */
    public static String toDisplayString(JsonValue value) {
        Objects.requireNonNull(value);
        return toDisplayString(value, 0 , false);
    }

    private static String toDisplayString(JsonValue jv, int indent, boolean isField) {
        return switch (jv) {
            case JsonObject jo -> toDisplayString(jo, indent, isField);
            case JsonArray ja -> toDisplayString(ja, indent, isField);
            default -> " ".repeat(isField ? 1 : indent) + jv;
        };
    }

    private static String toDisplayString(JsonObject jo, int indent, boolean isField) {
        var prefix = " ".repeat(indent);
        var s = new StringBuilder(isField ? " " : prefix);
        if (jo.members().isEmpty()) {
            s.append("{}");
        } else {
            s.append("{\n");
            jo.members().forEach((name, value) -> {
                if (value instanceof JsonValue val) {
                    s.append(prefix)
                            .append(" ".repeat(INDENT))
                            .append("\"")
                            .append(name)
                            .append("\":")
                            .append(Json.toDisplayString(val, indent + INDENT, true))
                            .append(",\n");
                } else {
                    throw new InternalError("type mismatch");
                }
            });
            s.setLength(s.length() - 2); // trim final comma
            s.append("\n").append(prefix).append("}");
        }
        return s.toString();
    }

    private static String toDisplayString(JsonArray ja, int indent, boolean isField) {
        var prefix = " ".repeat(indent);
        var s = new StringBuilder(isField ? " " : prefix);
        if (ja.values().isEmpty()) {
            s.append("[]");
        } else {
            s.append("[\n");
            for (JsonValue v: ja.values()) {
                if (v instanceof JsonValue jv) {
                    s.append(Json.toDisplayString(jv,indent + INDENT, false)).append(",\n");
                } else {
                    throw new InternalError("type mismatch");
                }
            }
            s.setLength(s.length() - 2); // trim final comma/newline
            s.append("\n").append(prefix).append("]");
        }
        return s.toString();
    }

    // default indentation for display string
    private static final int INDENT = 2;

    // no instantiation is allowed for this class
    private Json() {}
}
