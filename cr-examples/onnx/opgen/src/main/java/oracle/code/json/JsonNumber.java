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
 * The interface that represents JSON number.
 * <p>
 * A {@code JsonNumber} can be produced by {@link Json#parse(String)}.
 * Alternatively, {@link #of(double)} and its overload can be used to obtain
 * a {@code JsonNumber} from a {@code Number}.
 * When a JSON number is parsed, a {@code JsonNumber} object is created
 * regardless of its precision or magnitude as long as the syntax is valid.
 * The parsed string representation is retrieved from {@link #toString()}.
 *
 * 
 */
public sealed interface JsonNumber extends JsonValue permits JsonNumberImpl {

    /**
     * {@return the {@code Number} value represented by this
     * {@code JsonNumber}}
     *
     * @implNote The returned value's type is {@code Double} for floating point
     * numbers. For integer numbers, it is either {@code Integer}, {@code Long},
     * or {@code Double}. The return value is derived from the respective
     * {@code Number} subclass {@code valueOf(String)} methods, where the {@code String}
     * corresponds to the {@link #toString()} of this {@code JsonNumber}.
     *
     * @throws NumberFormatException if the string representation of this
     *          {@code JsonNumber} cannot be converted to a {@code Number}.
     * @see Double##decimalToBinaryConversion Decimal &harr; Binary Conversion Issues
     */
    Number value();

    /**
     * {@return the {@code JsonNumber} created from the given
     * {@code double}}
     *
     * @implNote If the given {@code double} is equivalent to {@code +/-infinity}
     * or {@code NaN}, this method will throw an {@code IllegalArgumentException}.
     *
     * @param num the given {@code double}.
     * @throws IllegalArgumentException if the given {@code num} is out
     *          of the accepted range.
     */
    static JsonNumber of(double num) {
        // non-integral types
        if (Double.isNaN(num) || Double.isInfinite(num)) {
            throw new IllegalArgumentException("Not a valid JSON number");
        }
        return new JsonNumberImpl(num);
    }

    /**
     * {@return the {@code JsonNumber} created from the given
     * {@code long}}
     *
     * @param num the given {@code long}.
     */
    static JsonNumber of(long num) {
        // integral types
        return new JsonNumberImpl(num);
    }
}
