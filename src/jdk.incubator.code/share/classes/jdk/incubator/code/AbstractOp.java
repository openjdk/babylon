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

import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.extern.OpWriter;

import java.util.*;

/**
 * The abstract implementation of an operation. All concrete operations extend this class.
 *
 * <h2>Operation implementation requirements</h2>
 * <p>
 * A concrete operation class must satisfy the following requirements:
 * <ul>
 * <li>
 * implement {@link #resultType()} to return the result type of operation instances;
 * <li>
 * implement {@link #transform(CodeContext, CodeTransformer)} to return a newly constructed, unplaced copy whose
 * concrete class is the concrete operation class;
 * <li>
 * call an appropriate {@code AbstractOp} superclass constructor from each concrete operation constructor. Constructors
 * for new operations pass the operation's operands to {@link #AbstractOp(List)}. Constructors for transformed copies
 * can pass the input operation and code context to {@link #AbstractOp(AbstractOp, CodeContext)};
 * <li>
 * override {@link #bodies()} if instances may have bodies. If the operation class implements {@link Op.Nested}, then
 * {@code bodies()} must return one or more bodies;
 * <li>
 * override {@link #successors()} if instances may have successors. If the operation class implements
 * {@link Op.BlockTerminating}, then {@code successors()} must return one or more successors;
 * <li>
 * copy mutable constructor arguments that define successors, bodies, and operation-specific state, ensuring they are
 * all fixed when construction completes; and
 * <li>
 * return unmodifiable views or immutable values from accessors that expose successors, bodies, and operation-specific
 * state.
 * </ul>
 * <p>
 * A concrete operation class may additionally:
 * <ul>
 * <li>
 * override {@link #externalizeOpName()} and {@link #externalize()} to define an external form;
 * <li>
 * implement {@link Op.Lowerable} to define a lowering; and
 * <li>
 * provide operation-specific accessors for operation-specific state.
 * </ul>
 */
public non-sealed abstract class AbstractOp extends InternalAbstractOp {
    /**
     * Constructs an operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     */
    protected AbstractOp(List<? extends Value> operands) {
        super(operands);
    }

    /**
     * Constructs an operation with operands mapped from, and location copied from, the given operation.
     * <p>
     * The constructed operation's operands are the values mapped, in order, from the given operation's operands using
     * the given code context. The new operation's location is the given operation's location, if any.
     *
     * @param that the operation
     * @param cc   the code context
     * @throws IllegalArgumentException if an operation's operand has no context mapping
     * @throws IllegalArgumentException if a mapped value's declaring block is built.
     */
    protected AbstractOp(AbstractOp that, CodeContext cc) {
        super(that, cc);
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    @Override
    public final List<Block.Reference> successors() {
        return List.of();
    }
}
