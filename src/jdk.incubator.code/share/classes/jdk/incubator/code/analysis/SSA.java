/*
 * Copyright (c) 2024, 2025, Oracle and/or its affiliates. All rights reserved.
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

package jdk.incubator.code.analysis;

import jdk.incubator.code.*;

/**
 * Functionality to transform a code model into pure SSA form, replacing operations that declare variables and
 * access them with the use of values they depend on or additional block parameters.
 */
public final class SSA {
    private SSA() {
    }

    /**
     * Applies an SSA transformation to an operation with bodies, replacing operations that declare variables and
     * access them with the use of values they depend on or additional block parameters.
     * <p>
     * The operation should first be in lowered form before applying this transformation.
     * <p>
     * Note: this implementation does not currently work correctly when a variable is stored to within an exception
     * region and read from outside as a result of catching an exception. In such cases a complete transformation may be
     * not possible and such variables will need to be retained.
     *
     * @param nestedOp the operation with bodies
     * @return the transformed operation
     * @param <T> the operation type
     */
    public static <T extends Op & Op.Nested> T transform(T nestedOp) {
        // @@@ property is used to test both impls
        if (!"cytron".equalsIgnoreCase(System.getProperty("babylon.ssa"))) {
            return SSABraun.transform(nestedOp);
        } else {
            return SSACytron.transform(nestedOp);
        }
    }
}
