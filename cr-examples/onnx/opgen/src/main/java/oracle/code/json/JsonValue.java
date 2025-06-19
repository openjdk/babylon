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
 * The interface that represents a JSON value.
 * <p>
 * Instances of {@code JsonValue} are immutable and thread safe.
 * <p>
 * A {@code JsonValue} can be produced by {@link Json#parse(String)} or {@link
 * Json#fromUntyped(Object)}. See {@link #toString()}  for converting a {@code
 * JsonValue} to its corresponding JSON String. For example,
 * {@snippet lang=java:
 *     List<Object> values = Arrays.asList("foo", true, 25);
 *     JsonValue json = Json.fromUntyped(values);
 *     json.toString(); // returns "[\"foo\",true,25]"
 * }
 *
 * A class implementing a non-sealed {@code JsonValue} sub-interface must adhere
 * to the following:
 * <ul>
 * <li>The class's implementations of {@code equals}, {@code hashCode},
 * and {@code toString} compute their results solely from the values
 * of the class's instance fields (and the members of the objects they
 * reference), not from the instance's identity.</li>
 * <li>The class's methods treat instances as <em>freely substitutable</em>
 * when equal, meaning that interchanging any two instances {@code x} and
 * {@code y} that are equal according to {@code equals()} produces no
 * visible change in the behavior of the class's methods.</li>
 * <li>The class performs no synchronization using an instance's monitor.</li>
 * <li>The class does not provide any instance creation mechanism that promises
 * a unique identity on each method call&mdash;in particular, any factory
 * method's contract must allow for the possibility that if two independently-produced
 * instances are equal according to {@code equals()}, they may also be
 * equal according to {@code ==}.</li>
 * </ul>
 * <p>
 * Users of {@code JsonValue} instances should ensure the following:
 * <ul>
 * <li> When two instances of {@code JsonValue} are equal (according to {@code equals()}), users
 * should not attempt to distinguish between their identities, whether directly via reference
 * equality or indirectly via an appeal to synchronization, identity hashing,
 * serialization, or any other identity-sensitive mechanism.</li>
 * <li> Synchronization on instances of {@code JsonValue} is strongly discouraged,
 * because the programmer cannot guarantee exclusive ownership of the
 * associated monitor.</li>
 * </ul>
 *
 * @since 99
 */
public sealed interface JsonValue
        permits JsonString, JsonNumber, JsonObject, JsonArray, JsonBoolean, JsonNull {

    /**
     * {@return the String representation of this {@code JsonValue} that conforms
     * to the JSON syntax} If this {@code JsonValue} is created by parsing a
     * JSON document, it preserves the text representation of the corresponding
     * JSON element, except that the returned string does not contain any white
     * spaces or newlines to produce a compact representation.
     * For a String representation suitable for display, use
     * {@link Json#toDisplayString(JsonValue)}.
     *
     * @see Json#toDisplayString(JsonValue)
     */
    String toString();
}
