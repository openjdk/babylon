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

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * JsonObject implementation class
 */
final class JsonObjectImpl implements JsonObject, JsonValueImpl {

    private final JsonDocumentInfo docInfo;
    private final int startIndex;
    private final int endIndex;
    private Map<String, JsonValue> theKeys;

    // Via of factory
    JsonObjectImpl(Map<String, ? extends JsonValue> map) {
        theKeys = Collections.unmodifiableMap(map);
        docInfo = null;
        startIndex = 0;
        endIndex = 0;
    }

    // Via untyped
    JsonObjectImpl(Map<?, ?> map, Set<Object> untypedObjs, int depth) {
        HashMap<String, JsonValue> m = HashMap.newHashMap(map.size());
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String strKey)) {
                throw new IllegalArgumentException("Key is not a String: " + entry.getKey());
            } else {
                m.put(strKey, JsonGenerator.fromUntyped(entry.getValue(), untypedObjs, depth));
            }
        }
        theKeys = Collections.unmodifiableMap(m);
        docInfo = null;
        startIndex = 0;
        endIndex = 0;
    }

    JsonObjectImpl(JsonDocumentInfo doc, int index) {
        docInfo = doc;
        startIndex = index;
        endIndex = startIndex == 0 ? docInfo.getIndexCount() - 1 // For root
                : docInfo.nextIndex(index, '{', '}');
    }

    @Override
    public Map<String, JsonValue> keys() {
        if (theKeys == null) {
            theKeys = inflate();
        }
        return theKeys;
    }

    // Inflates the JsonObject using the tokens array
    private Map<String, JsonValue> inflate() {
        var k = new HashMap<String, JsonValue>();
        var index = startIndex + 1;
        // Empty case automatically checked by index increment. {} is 2 tokens
        while (index < endIndex) {
            // Member name should be source string, not unescaped
            // Member equality is done via unescaped in JsonParser
            var key = docInfo.substring(
                    docInfo.getOffset(index) + 1, docInfo.getOffset(index + 1));
            index = index + 2;

            // Get value
            int offset = docInfo.getOffset(index) + 1;
            if (docInfo.shouldWalkToken(docInfo.charAtIndex(index + 1))) {
                index++;
            }
            var value = JsonGenerator.createValue(docInfo, offset, index);

            // Store key and value
            k.put(key, value);
            // Move to the next key
            index = ((JsonValueImpl)value).getEndIndex() + 1;
        }
        return Collections.unmodifiableMap(k);
    }

    @Override
    public int getEndIndex() {
        return endIndex + 1; // We are interested in the index after '}'
    }

    @Override
    public boolean equals(Object o) {
        return this == o ||
            o instanceof JsonObjectImpl ojoi &&
            Objects.equals(keys(), ojoi.keys());
    }

    @Override
    public int hashCode() {
        return Objects.hash(keys());
    }

    @Override
    public Map<String, Object> toUntyped() {
        return keys().entrySet().stream()
            .collect(HashMap::new, // to allow `null` value
                (m, e) -> m.put(e.getKey(), Json.toUntyped(e.getValue())),
                HashMap::putAll);
    }

    @Override
    public String toString() {
        var s = new StringBuilder("{");
        for (Map.Entry<String, JsonValue> kv: keys().entrySet()) {
            s.append("\"").append(kv.getKey()).append("\":")
             .append(kv.getValue().toString())
             .append(",");
        }
        if (!keys().isEmpty()) {
            s.setLength(s.length() - 1); // trim final comma
        }
        return s.append("}").toString();
    }

    @Override
    public String toDisplayString(int indent, boolean isField) {
        var prefix = " ".repeat(indent);
        var s = new StringBuilder(isField ? " " : prefix);
        if (keys().isEmpty()) {
            s.append("{}");
        } else {
            s.append("{\n");
            keys().entrySet().stream()
                .sorted(Map.Entry.comparingByKey(String::compareTo))
                .forEach(e -> {
                    var key = e.getKey();
                    var value = e.getValue();
                    if (value instanceof JsonValueImpl val) {
                        s.append(prefix)
                                .append(" ".repeat(INDENT))
                                .append("\"")
                                .append(key)
                                .append("\":")
                                .append(val.toDisplayString(indent + INDENT, true))
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
}
