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
 * A {@link Value value}, one of {@link Block.Parameter} or {@link Op.Result}, has a
 * {@code CodeType} classifying that value.
 * A {@link Body} has a {@code CodeType}, the {@link Body#yieldType yield type},
 * classifying values yielded from the body.
 * <p>
 * The {@code equals} method should be used to compare code types.
 *
 * @apiNote
 * Code types enable reasoning statically about a code model, approximating
 * run time behavior.
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
     * @return the code type's content.
     */
    ExternalizedCodeType externalize();

    /**
     * Return a string representation of this code type.
     */
    @Override
    String toString();

    @Override
    boolean equals(Object o);

    @Override
    int hashCode();
}
