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

import java.util.List;
import java.util.Map;
import java.util.Set;

// Responsible for creating "lazy" state JsonValue(s) using the tokens array
final class JsonGenerator {

    static JsonValue createValue(JsonDocumentInfo docInfo, int offset, int index) {
        offset = JsonParser.skipWhitespaces(docInfo, offset);
        return switch (docInfo.charAt(offset)) {
            case '{' -> createObject(docInfo, index);
            case '[' -> createArray(docInfo, offset, index);
            case '"' -> createString(docInfo, offset, index);
            case 't', 'f' -> createBoolean(docInfo, offset, index);
            case 'n' -> createNull(docInfo, index);
            case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
                    -> createNumber(docInfo, offset, index);
            default -> throw new InternalError();
        };
    }

    static JsonObject createObject(JsonDocumentInfo docInfo, int index) {
        return new JsonObjectImpl(docInfo, index);
    }

    static JsonArray createArray(JsonDocumentInfo docInfo, int offset, int index) {
        return new JsonArrayImpl(docInfo, offset, index);
    }

    static JsonString createString(JsonDocumentInfo docInfo, int offset, int index) {
        return new JsonStringImpl(docInfo, offset, index);
    }

    static JsonBoolean createBoolean(JsonDocumentInfo docInfo, int offset, int index) {
        return new JsonBooleanImpl(docInfo, offset, index);
    }

    static JsonNull createNull(JsonDocumentInfo docInfo, int index) {
        return new JsonNullImpl(docInfo, index);
    }

    static JsonNumber createNumber(JsonDocumentInfo docInfo, int offset, int index) {
        return new JsonNumberImpl(docInfo, offset, index);
    }

    // untypedObjs is an identity hash set that serves to identify if a circular
    // reference exists
    static JsonValue fromUntyped(Object src, Set<Object> untypedObjs, int depth) {
        return switch (src) {
            // Structural JSON: Object, Array
            case Map<?, ?> map -> {
                if (!untypedObjs.add(map)) {
                    throw new IllegalArgumentException("Circular reference detected");
                }
                if (depth + 1 > Json.MAX_DEPTH) {
                    throw new IllegalArgumentException("Max depth exceeded");
                }
                yield new JsonObjectImpl(map, untypedObjs, depth + 1);
            }
            case List<?> list-> {
                if (!untypedObjs.add(list)) {
                    throw new IllegalArgumentException("Circular reference detected");
                }
                if (depth + 1 > Json.MAX_DEPTH) {
                    throw new IllegalArgumentException("Max depth exceeded");
                }
                yield new JsonArrayImpl(list, untypedObjs, depth + 1);
            }
            // JsonPrimitives
            case String str -> new JsonStringImpl(str);
            case Boolean bool -> new JsonBooleanImpl(bool);
            case null -> JsonNull.of();
            case Float f -> JsonNumber.of(f); // promote Float to Double
            case Integer i -> new JsonNumberImpl(i); // preserve Integer via ctr
            case Double db -> JsonNumber.of(db);
            case Long lg -> JsonNumber.of(lg);
            // JsonValue
            case JsonValue jv -> {
                checkDepth(jv, depth + 1);
                yield jv;
            }
            default -> throw new IllegalArgumentException("Type not recognized.");
        };
    }

    static void checkDepth(JsonValue val, int depth) {
        if (depth > Json.MAX_DEPTH) {
            throw new IllegalArgumentException("Max depth exceeded");
        }
        switch (val) {
            case JsonObject jo -> jo.keys().forEach((_, jV) -> checkDepth(jV, depth + 1));
            case JsonArray ja -> ja.values().forEach(jV -> checkDepth(jV, depth + 1));
            default -> {} // Primitive JSON can not nest
        }
    }

    // no instantiation of this generator
    private JsonGenerator(){}
}
