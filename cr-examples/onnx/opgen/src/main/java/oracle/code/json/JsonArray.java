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

import oracle.code.json.impl.JsonArrayImpl;

import java.util.ArrayList;
import java.util.List;

/**
 * The interface that represents JSON array.
 * <p>
 * A {@code JsonArray} can be produced by {@link Json#parse(String)}.
 * <p> Alternatively, {@link #of(List)} can be used to obtain a {@code JsonArray}.
 *
 * @since 99
 */
public non-sealed interface JsonArray extends JsonValue {

    /**
     * {@return an unmodifiable list of the {@code JsonValue} elements in
     * this {@code JsonArray}}
     */
    List<JsonValue> values();

    /**
     * {@return the {@code JsonArray} created from the given
     * list of {@code JsonValue}s}
     *
     * @param src the list of {@code JsonValue}s. Non-null.
     * @throws NullPointerException if {@code src} is {@code null}, or contains
     *      any values that are {@code null}
     */
    static JsonArray of(List<? extends JsonValue> src) {
        var values = new ArrayList<JsonValue>(src); // implicit null check
        if (values.contains(null)) {
            throw new NullPointerException("src contains null value(s)");
        }
        return new JsonArrayImpl(values);
    }

    /**
     * {@return {@code true} if the given object is also a {@code JsonArray}
     * and the two {@code JsonArray}s represent the same elements} Two
     * {@code JsonArray}s {@code ja1} and {@code ja2} represent the same
     * elements if {@code ja1.values().equals(ja2.values())}.
     *
     * @see #values()
     */
    @Override
    boolean equals(Object obj);

    /**
     * {@return the hash code value for this {@code JsonArray}} The hash code of a
     * {@code JsonArray} is calculated by {@code Objects.hash(JsonArray.values()}.
     * Thus, for two {@code JsonArray}s {@code ja1} and {@code ja2},
     * {@code ja1.equals(ja2)} implies that {@code ja1.hashCode() == ja2.hashCode()}
     * as required by the general contract of {@link Object#hashCode}.
     *
     * @see #values()
     */
    @Override
    int hashCode();
}
