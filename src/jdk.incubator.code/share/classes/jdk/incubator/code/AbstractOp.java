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

import java.util.*;

/**
 * The abstract implementation of a non-terminating operation. All concrete non-terminating operations extend this
 * class and implement {@link Op}.
 *
 * <h2>Operation implementation requirements</h2>
 * <p>
 * A concrete non-terminating operation class must satisfy the following requirements:
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
 * copy mutable constructor arguments that define successors, bodies, and operation-specific state, ensuring they are
 * all fixed when construction completes; and
 * <li>
 * return unmodifiable views or immutable values from accessors that expose successors, bodies, and operation-specific
 * state.
 * </ul>
 * <p>
 * A concrete non-terminating operation class may additionally:
 * <ul>
 * <li>
 * implement {@link jdk.incubator.code.extern.ExternalizedOp.Externalizable} to define an external form;
 * <li>
 * implement {@link Op.Lowerable} to define a lowering; and
 * <li>
 * provide operation-specific accessors for operation-specific state.
 * </ul>
 */
public non-sealed abstract class AbstractOp extends InternalAbstractOp {
    /**
     * Constructs a non-terminating operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     */
    protected AbstractOp(List<? extends Value> operands) {
        super(operands);
    }

    /**
     * Constructs a non-terminating with operands mapped from, and location copied from, the given operation.
     * <p>
     * The constructed operation's operands are the values mapped, in order, from the given operation's operands using
     * the given code context. The constructed operation's location is the given operation's location, if any.
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

    /**
     * The abstract implementation of a terminating operation. All concrete terminating operations extend this
     * class and implement {@link Terminating}.
     *
     * <h2>Operation implementation requirements</h2>
     * <p>
     * A concrete terminating operation class must satisfy the implementation requirements of a concrete non-terminating
     * operation specified by {@link AbstractOp} in addition to the following requirements:
     * <ul>
     * <li>
     * override {@link #successors()} if instances may have successors;
     * </ul>
     * <p>
     * A concrete terminating operation class may additionally:
     * <ul>
     * <li>
     * implement {@link jdk.incubator.code.extern.ExternalizedOp.Externalizable} to define an external form;
     * <li>
     * implement {@link Lowerable} to define a lowering; and
     * <li>
     * provide operation-specific accessors for operation-specific state.
     * </ul>
     */
    public non-sealed abstract static class Terminating extends InternalAbstractOp
            implements Op.Terminating {

        final List<Block.Reference> successors;

        /**
         * Constructs a terminating operation with a list of operands and list of successors
         *
         * @param operands the list of operands, a copy of the list is performed if required.
         * @param successors the list of successors, a copy of the list is performed if required.
         * @throws IllegalArgumentException if an operand's declaring block is built.
         * @throws IllegalArgumentException if a successor's referencing block is built or successor's block argument's
         * declaring block is built.
         */
        protected Terminating(List<? extends Value> operands, List<Block.Reference> successors) {
            super(operands);

            // @@@ Check unbuilt blocks/arguments
            this.successors = List.copyOf(successors);
        }

        /**
         * Constructs a terminating operation with a list of operands and an empty list of successors
         *
         * @param operands the list of operands, a copy of the list is performed if required.
         * @throws IllegalArgumentException if an operand's declaring block is built.
         */
        protected Terminating(List<? extends Value> operands) {
            this(operands, List.of());
        }

        /**
         * Constructs a terminating operation with operands and successors mapped from, and location copied from, the given operation.
         * <p>
         * The constructed operation's operands are the values mapped, in order, from the given operation's operands using
         * the given code context. The constructed operation's successors are the successors mapped, in order, from the
         * given operation's successors using the given code context and applying
         * {@link CodeContext#getReferenceOrCreate(Block.Reference)} to each successor. The constructed operation's location
         * is the given operation's location, if any.
         *
         * @param that the operation
         * @param cc   the code context
         * @throws IllegalArgumentException if an operation's operand has no context mapping
         * @throws IllegalArgumentException if a mapped value's declaring block is built.
         * @throws IllegalArgumentException if a mapped successor's referencing block is built or mapped successor's block
         * argument's declaring block is built.
         */
        protected Terminating(AbstractOp.Terminating that, CodeContext cc) {
            super(that, cc);

            this.successors = that.successors().stream().map(s -> cc.getReferenceOrCreate(s)).toList();
        }

        @Override
        public final List<Block.Reference> successors() {
            return successors;
        }
    }
}
