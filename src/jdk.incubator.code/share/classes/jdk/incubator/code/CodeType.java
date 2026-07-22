/*
 * Copyright (c) 2024, 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code;

import jdk.incubator.code.extern.ExternalizedCodeType;

/**
 * A code type that classifies values.
 * <p>
 * A {@link Value value}, one of {@link Block.Parameter} or {@link Op.Result}, has a {@code CodeType} classifying that
 * value. A {@link Body} has a {@code CodeType}, the {@link Body#yieldType yield type}, classifying values yielded from
 * the body.
 * <p>
 * The {@code equals} method should be used to compare code types.
 *
 * <h2>Code type implementation requirements</h2>
 * <p>
 * Instances of a concrete code type class must be immutable.
 * <p>
 * A concrete code type class must satisfy the following requirements:
 * <ul>
 * <li>
 * implement {@link #equals(Object)}, {@link #hashCode()}, and {@link #toString()} so that their results are solely
 * computed from code type state, and not from an instance's identity;
 * <li>
 * treat equal instances as freely substitutable, so that interchanging two instances that are equal according to
 * {@code equals} produces no visible change in the behavior of the code type's methods;
 * <li>
 * ensure equal instances have equal {@link #externalize() externalized content};
 * <li>
 * provide no creation mechanism that promises unique identity for created instances;
 * <li>
 * copy mutable constructor arguments that define code type state, ensuring they are all fixed when construction
 * completes; and
 * <li>
 * return unmodifiable views or immutable values from accessors that expose code type state.
 * </ul>
 * <p>
 * A concrete code type class may additionally:
 * <ul>
 * <li>
 * override {@link #externalize()} to define an external form; and
 * <li>
 * provide accessors for code type state.
 * </ul>
 *
 * @apiNote
 * Code types enable reasoning statically about a code model, approximating run time behavior.
 * <p>
 * A code type might model a Java primitive type such as {@link jdk.incubator.code.dialect.java.JavaType#INT int},
 * a specific Java class such as {@link jdk.incubator.code.dialect.java.JavaType#J_L_STRING String}, or more generally a
 * {@link jdk.incubator.code.dialect.core.FunctionType function type} or a
 * {@link jdk.incubator.code.dialect.core.TupleType tuple type}.
 *
 * @see Value
 * @see Block.Parameter
 * @see Op.Result
 * @see Body#yieldType()
 */
public non-sealed interface CodeType extends CodeItem {
    // @@@ Common useful methods generally associated with properties of a type
    // e.g., arguments, is an array etc. (dimensions)

    /**
     * Externalizes this code type's content.
     *
     * @implSpec
     * This implementation returns an externalized code type whose identifier is the result of invoking
     * {@link #toString()} and whose argument list is empty.
     *
     * @return the code type's content.
     */
    default ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of(toString());
    }

    /**
     * Return a string representation of this code type.
     * <p>
     * An implementing class should avoid implementing this method by returning the result of
     * {@code externalize().toString()} unless that class overrides the {@code externalize} method with its own
     * implementation.
     */
    @Override
    String toString();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
