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

package oracle.code.json.impl;

import oracle.code.json.JsonObject;
import oracle.code.json.JsonValue;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * JsonObject implementation class
 */
public final class JsonObjectImpl implements JsonObject {

    private final Map<String, JsonValue> theMembers;

    public JsonObjectImpl(Map<String, JsonValue> map) {
        theMembers = map;
    }

    @Override
    public Map<String, JsonValue> members() {
        return Collections.unmodifiableMap(theMembers);
    }

    @Override
    public String toString() {
        var s = new StringBuilder("{");
        for (Map.Entry<String, JsonValue> kv: members().entrySet()) {
            s.append("\"").append(kv.getKey()).append("\":")
             .append(kv.getValue().toString())
             .append(",");
        }
        if (!members().isEmpty()) {
            s.setLength(s.length() - 1); // trim final comma
        }
        return s.append("}").toString();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof JsonObject ojo &&
                Objects.equals(members(), ojo.members());
    }

    @Override
    public int hashCode() {
        return Objects.hash(members());
    }
}
