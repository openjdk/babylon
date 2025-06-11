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

import oracle.code.json.impl.JsonBooleanImpl;

/**
 * The interface that represents JSON boolean.
 * <p>
 * A {@code JsonBoolean} can be produced by {@link Json#parse(String)}.
 * <p> Alternatively, {@link #of(boolean)} can be used to
 * obtain a {@code JsonBoolean}.
 *
 * @since 99
 */
public non-sealed interface JsonBoolean extends JsonValue {

    /**
     * {@return the {@code boolean} value represented by this
     * {@code JsonBoolean}}
     */
    boolean value();

    /**
     * {@return the {@code JsonBoolean} created from the given
     * {@code boolean}}
     *
     * @param src the given {@code boolean}.
     */
    static JsonBoolean of(boolean src) {
        return src ? JsonBooleanImpl.TRUE : JsonBooleanImpl.FALSE;
    }

    /**
     * {@return {@code true} if the given object is also a {@code JsonBoolean}
     * and the two {@code JsonBoolean}s represent the same boolean value} Two
     * {@code JsonBoolean}s {@code jb1} and {@code jb2} represent the same
     * boolean values if {@code jb1.value().equals(jb2.value())}.
     *
     * @see #value()
     */
    @Override
    boolean equals(Object obj);

    /**
     * {@return the hash code value for this {@code JsonBoolean}} The hash code value
     * of a {@code JsonBoolean} is defined to be the hash code of {@code JsonBoolean}'s
     * {@link #value()}. Thus, for two {@code JsonBooleans}s {@code jb1} and {@code jb2},
     * {@code jb1.equals(jb2)} implies that {@code jb1.hashCode() == jb2.hashCode()}
     * as required by the general contract of {@link Object#hashCode}.
     *
     * @see #value()
     */
    @Override
    int hashCode();
}
