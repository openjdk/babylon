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

sealed abstract class InternalAbstractOp implements Op permits AbstractOp, AbstractTerminatingOp {

    // Set when op is placed in a block or as a root operation, otherwise null when unplaced
    // @@@ stable value?
    Result result;

    // null if not specified
    // @@@ stable value?
    Location location;

    final List<Value> operands;

    /**
     * Constructs an operation with a list of operands.
     *
     * @param operands the list of operands, a copy of the list is performed if required.
     * @throws IllegalArgumentException if an operand's declaring block is built.
     */
    protected InternalAbstractOp(List<? extends Value> operands) {
        for (Value operand : operands) {
            if (operand.isBuilt()) {
                throw new IllegalArgumentException("Operand's declaring block is built: " + operand);
            }
        }
        this.operands = List.copyOf(operands);
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
    protected InternalAbstractOp(InternalAbstractOp that, CodeContext cc) {
        this(cc.getValues(that.operands));
        this.location = that.location;
    }

    @Override
    public final void setLocation(Location l) {
        // @@@ Fail if location != null?
        if (isRoot() || (result != null && result.block.isBuilt())) {
            throw new IllegalStateException("Built operation");
        }

        location = l;
    }

    @Override
    public final Location location() {
        return location;
    }

    @Override
    public final Block parent() {
        if (isRoot() || result == null) {
            return null;
        }

        if (!result.block.isBuilt()) {
            throw new IllegalStateException("Parent block is unobservable");
        }

        return result.block;
    }

    @Override
    public final List<Body> children() {
        return bodies();
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty list.
     */
    @Override
    public List<Body> bodies() {
        return List.of();
    }

    @Override
    public final Result result() {
        return result == Result.ROOT_RESULT ? null : result;
    }

    @Override
    public final List<Value> operands() {
        return operands;
    }

    @Override
    public final FunctionType opSignature() {
        List<CodeType> operandTypes = operands.stream().map(Value::type).toList();
        return CoreType.functionType(resultType(), operandTypes);
    }

    @Override
    public final List<Value> capturedValues() {
        Set<Value> cvs = new LinkedHashSet<>();

        Deque<Body> bodyStack = new ArrayDeque<>();
        for (Body childBody : bodies()) {
            Body.capturedValues(cvs, bodyStack, childBody);
        }
        return new ArrayList<>(cvs);
    }

    @Override
    public final void buildAsRoot() {
        if (result == Result.ROOT_RESULT) {
            return;
        }
        if (!bodies().stream().allMatch(Body::isIsolated)) {
            throw new IllegalStateException("One of the operation bodies is not isolated");
        }
        if (!operands().isEmpty()) {
            throw new IllegalStateException("Operation has operands");
        }
        if (!successors().isEmpty()) {
            throw new IllegalStateException("Operation has successors");
        }
        if (result != null) {
            throw new IllegalStateException("Operation is placed in a block");
        }
        result = Result.ROOT_RESULT;
    }

    @Override
    public final boolean isRoot() {
        return result == Result.ROOT_RESULT;
    }

    @Override
    public final boolean isPlacedInBlock() {
        return !isRoot() && result != null;
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns the result of the expression {@code this.getClass().getName()}.
     */
    @Override
    public String externalizeOpName() {
        return this.getClass().getName();
    }

    /**
     * {@inheritDoc}
     * @implSpec this implementation returns an unmodifiable empty map.
     */
    @Override
    public Map<String, Object> externalize() {
        return Map.of();
    }

    @Override
    public final String toText() {
        return OpWriter.toText(this);
    }
}
