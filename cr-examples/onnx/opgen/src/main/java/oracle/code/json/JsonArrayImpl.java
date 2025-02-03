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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * JsonArray implementation class
 */
final class JsonArrayImpl implements JsonArray, JsonValueImpl {

    private final JsonDocumentInfo docInfo;
    private final int endIndex;
    private final int startIndex;
    private final int startOffset;
    private List<JsonValue> theValues;

    // Via of factory
    JsonArrayImpl(List<? extends JsonValue> from) {
        theValues = Collections.unmodifiableList(from);
        this.endIndex = 0;
        this.startIndex = 0;
        this.startOffset = 0;
        docInfo = null;
    }

    // Via untyped
    JsonArrayImpl(List<?> from, Set<Object> untypedObjs, int depth) {
        List<JsonValue> l = new ArrayList<>(from.size());
        for (Object o : from) {
            l.add(JsonGenerator.fromUntyped(o, untypedObjs, depth));
        }
        theValues = Collections.unmodifiableList(l);
        this.endIndex = 0;
        this.startIndex = 0;
        this.startOffset = 0;
        docInfo = null;
    }

    JsonArrayImpl(JsonDocumentInfo doc, int offset, int index) {
        docInfo = doc;
        startOffset = offset;
        startIndex = index;
        endIndex = startIndex == 0 ? docInfo.getIndexCount() - 1 // For root
                : docInfo.nextIndex(index, '[', ']');
    }

    @Override
    public List<JsonValue> values() {
        if (theValues == null) {
            theValues = inflate();
        }
        return theValues;
    }

    // Inflate the JsonArray using the tokens array.
    private List<JsonValue> inflate() {
        if (docInfo.charAt(JsonParser.skipWhitespaces(docInfo, startOffset + 1)) == ']') {
            return Collections.emptyList();
        }
        var v = new ArrayList<JsonValue>();
        var index = startIndex;
        while (index < endIndex) { // start on comma or opening bracket
            // Get Val
            int offset = docInfo.getOffset(index) + 1;
            if (docInfo.shouldWalkToken(docInfo.charAtIndex(index + 1))) {
                index++;
            }
            var value = JsonGenerator.createValue(docInfo, offset, index);
            v.add(value);
            index = ((JsonValueImpl)value).getEndIndex(); // Move to comma or closing
        }
        return Collections.unmodifiableList(v);
    }

    @Override
    public int getEndIndex() {
        return endIndex + 1;  // We are always interested in the index after ']'
    }

    @Override
    public boolean equals(Object o) {
        return this == o ||
            o instanceof JsonArrayImpl ojai &&
            Objects.equals(values(), ojai.values());
    }

    @Override
    public int hashCode() {
        return Objects.hash(values());
    }

    @Override
    public List<Object> toUntyped() {
        return values().stream()
                .map(Json::toUntyped)
                .toList();
    }

    @Override
    public String toString() {
        var s = new StringBuilder("[");
        for (JsonValue v: values()) {
            s.append(v.toString()).append(",");
        }
        if (!values().isEmpty()) {
            s.setLength(s.length() - 1); // trim final comma
        }
        return s.append("]").toString();
    }

    @Override
    public String toDisplayString(int indent, boolean isField) {
        var prefix = " ".repeat(indent);
        var s = new StringBuilder(isField ? " " : prefix);
        if (values().isEmpty()) {
            s.append("[]");
        } else {
            s.append("[\n");
            for (JsonValue v: values()) {
                if (v instanceof JsonValueImpl impl) {
                    s.append(impl.toDisplayString(indent + INDENT, false)).append(",\n");
                } else {
                    throw new InternalError("type mismatch");
                }
            }
            s.setLength(s.length() - 2); // trim final comma/newline
            s.append("\n").append(prefix).append("]");
        }
        return s.toString();
    }
}
