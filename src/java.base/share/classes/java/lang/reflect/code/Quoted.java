/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

package java.lang.reflect.code;

import java.util.ArrayList;
import java.util.Map;

/**
 * The quoted form of an operation.
 * <p>
 * The quoted form is utilized when the code model of some code is to be obtained rather than obtaining the result of
 * executing that code. For example passing the of a lambda expression in quoted form rather than the expression being
 * targeted to a functional interface from which it can be invoked.
 */
public final class Quoted {
    private final Op op;
    private final Map<Value, Object> capturedValues;

    /**
     * Constructs the quoted form of a given operation.
     *
     * @param op the invokable operation.
     */
    public Quoted(Op op) {
        this(op, Map.of());
    }

    /**
     * Constructs the quoted form of a given operation.
     * <p>
     * The captured values key set must have the same elements and same encounter order as
     * operation's captured values, specifically the following expression should evaluate to true:
     * {@snippet lang=java :
     * op.capturedValues().equals(new ArrayList<>(capturedValues.keySet()));
     * }
     * The captured values key set is not required to preserve the encounter
     * order of the operation's captured values.
     *
     * @param op             the operation.
     * @param capturedValues the captured values referred to by the operation
     * @see Op#capturedValues()
     */
    public Quoted(Op op, Map<Value, Object> capturedValues) {
        // @@@ This check is potentially expensive, remove or keep as assert?
        // @@@ Or make Quoted an interface, with a module private implementation?
        assert op.capturedValues().equals(new ArrayList<>(capturedValues.keySet()));
        this.op = op;
        this.capturedValues = Map.copyOf(capturedValues);
    }

    /**
     * Returns the operation.
     *
     * @return the operation.
     */
    public Op op() {
        return op;
    }

    /**
     * Returns the captured values.
     * <p>
     * The captured values key set has the same elements and same encounter order as
     * operation's captured values, specifically the following expression should evaluate to true:
     * {@snippet lang=java :
     * op().capturedValues().equals(new ArrayList<>(capturedValues().keySet()));
     * }
     *
     * @return the captured values, as an unmodifiable map.
     */
    public Map<Value, Object> capturedValues() {
        return capturedValues;
    }
}
