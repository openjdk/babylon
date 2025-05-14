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

import oracle.code.json.impl.JsonNumberImpl;

import java.math.BigDecimal;
import java.math.BigInteger;

/**
 * The interface that represents JSON number, an arbitrary-precision
 * number represented in base 10 using decimal digits.
 * <p>
 * A {@code JsonNumber} can be produced by {@link Json#parse(String)}.
 * Alternatively, {@link #of(double)} and its overloads can be used to obtain
 * a {@code JsonNumber} from a {@code Number}.
 * When a JSON number is parsed, a {@code JsonNumber} object is created
 * as long as the parsed value adheres to the JSON number
 * <a href="https://datatracker.ietf.org/doc/html/rfc8259#section-6">
 * syntax</a>. The value of the {@code JsonNumber}
 * can be retrieved from {@link #toString()} as the string representation
 * from which the JSON number is originally parsed, with
 * {@link #toNumber()} as a {@code Number} instance, or with
 * {@link #toBigDecimal()}.
 *
 * @spec https://datatracker.ietf.org/doc/html/rfc8259#section-6 RFC 8259:
 *      The JavaScript Object Notation (JSON) Data Interchange Format - Numbers
 * @since 99
 */
public non-sealed interface JsonNumber extends JsonValue {

    /**
     * {@return the {@code Number} parsed or translated from the
     * {@link #toString string representation} of this {@code JsonNumber}}
     * <p>
     * This method operates on the string representation and depending on that
     * representation computes and returns an instance of {@code Long}, {@code BigInteger},
     * {@code Double}, or {@code BigDecimal}.
     * <p>
     * If the string representation is the decimal string representation of
     * a {@code long} value, parsable by {@link Long#parseLong(String)},
     * then that {@code long} value is returned in its boxed form as {@code Long}.
     * Otherwise, if the string representation is the decimal string representation of a
     * {@code BigInteger}, translatable by {@link BigInteger#BigInteger(String)},
     * then that {@code BigInteger} is returned.
     * Otherwise, if the string representation is the decimal string representation of
     * a {@code double} value, parsable by {@link Double#parseDouble(String)},
     * and the {@code double} value is not {@link Double#isInfinite() infinite}, then that
     * {@code double} value is returned in its boxed form as {@code Double}.
     * Otherwise, and in all other cases, the string representation is the decimal string
     * representation of a {@code BigDecimal}, translatable by
     * {@link BigDecimal#BigDecimal(String)}, and that {@code BigDecimal} is
     * returned.
     * <p>
     * The computation may not preserve all information in the string representation.
     * In all of the above cases one or more leading zero digits are not preserved.
     * In the third case, returning {@code Double}, decimal to binary conversion may lose
     * decimal precision, and will not preserve one or more trailing zero digits in the fraction
     * part.
     *
     * @apiNote
     * Pattern matching can be used to match against {@code Long},
     * {@code Double}, {@code BigInteger}, or {@code BigDecimal} reference
     * types. For example:
     * {@snippet lang=java:
     * switch(jsonNumber.toNumber()) {
     *     case Long l -> { ... }
     *     case Double d -> { ... }
     *     case BigInteger bi -> { ... }
     *     case BigDecimal bd -> { ... }
     *     default -> { } // should not happen
     * }
     *}
     * @throws NumberFormatException if the {@code Number} cannot be parsed or translated from the string representation
     * @see #toBigDecimal()
     * @see #toString()
     */
    Number toNumber();

    /**
     * {@return the {@code BigDecimal} translated from the
     * {@link #toString string representation} of this {@code JsonNumber}}
     * <p>
     * The string representation is the decimal string representation of a
     * {@code BigDecimal}, translatable by {@link BigDecimal#BigDecimal(String)},
     * and that {@code BigDecimal} is returned.
     * <p>
     * The translation may not preserve all information in the string representation.
     * The sign is not preserved for the decimal string representation {@code -0.0}. One or more
     * leading zero digits are not preserved.
     *
     * @throws NumberFormatException if the {@code BigDecimal} cannot be translated from the string representation
     */
    BigDecimal toBigDecimal();

    /**
     * Creates a JSON number whose string representation is the
     * decimal string representation of the given {@code double} value,
     * produced by applying the value to {@link Double#toString(double)}.
     *
     * @param num the given {@code double} value.
     * @return a JSON number created from a {@code double} value
     * @throws IllegalArgumentException if the given {@code double} value
     * is {@link Double#isNaN() NaN} or is {@link Double#isInfinite() infinite}.
     */
    static JsonNumber of(double num) {
        // non-integral types
        return new JsonNumberImpl(num);
    }

    /**
     * Creates a JSON number whose string representation is the
     * decimal string representation of the given {@code long} value,
     * produced by applying the value to {@link Long#toString(long)}.
     *
     * @param num the given {@code long} value.
     * @return a JSON number created from a {@code long} value
     */
    static JsonNumber of(long num) {
        // integral types
        return new JsonNumberImpl(num);
    }

    /**
     * Creates a JSON number whose string representation is the
     * string representation of the given {@code BigInteger} value.
     *
     * @param num the given {@code BigInteger} value.
     * @return a JSON number created from a {@code BigInteger} value
     */
    static JsonNumber of(BigInteger num) {
        return new JsonNumberImpl(num);
    }

    /**
     * Creates a JSON number whose string representation is the
     * string representation of the given {@code BigDecimal} value.
     *
     * @param num the given {@code BigDecimal} value.
     * @return a JSON number created from a {@code BigDecimal} value
     */
    static JsonNumber of(BigDecimal num) {
        return new JsonNumberImpl(num);
    }

    /**
     * {@return the decimal string representation of this {@code JsonNumber}}
     *
     * If this {@code JsonNumber} is created by parsing a JSON number in a JSON document,
     * it preserves the string representation in the document, regardless of its
     * precision or range. For example, a JSON number like
     * {@code 3.141592653589793238462643383279} in the JSON document will be
     * returned exactly as it appears.
     * If this {@code JsonNumber} is created via one of the factory methods,
     * such as {@link JsonNumber#of(double)}, then the string representation is
     * specified by the factory method.
     */
    @Override
    String toString();

    /**
     * {@return true if the given {@code obj} is equal to this {@code JsonNumber}}
     * The comparison is based on the string representation of this {@code JsonNumber},
     * ignoring the case.
     *
     * @see #toString()
     */
    @Override
    boolean equals(Object obj);

    /**
     * {@return the hash code value of this {@code JsonNumber}} The returned hash code
     * is calculated based on the string representation of this {@code JsonNumber},
     * ignoring the case.
     *
     * @see #toString()
     */
    @Override
    int hashCode();
}
