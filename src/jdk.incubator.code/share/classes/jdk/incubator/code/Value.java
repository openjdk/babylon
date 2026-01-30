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
 * A value, that is the result of an operation or a block parameter.
 * @sealedGraph
 */
public sealed abstract class Value implements CodeItem
        permits Block.Parameter, Op.Result {
    final Block block;
    final TypeElement type;
    // @@@ In topological order?
    //     Can the representation be more efficient e.g. an array?
    final SequencedSet<Op.Result> uses;

    Value(Block block, TypeElement type) {
        this.block = block;
        this.type = type;
        this.uses = new LinkedHashSet<>();
    }

    /**
     * Returns this value's declaring block.
     * <p>If the value is an operation result, then the declaring block is the operation's parent block.
     * If the value is a block parameter then the declaring block is the block declaring the parameter.
     *
     * @return the value's declaring block.
     * @throws IllegalStateException if an unbuilt block is encountered.
     */
    public Block declaringBlock() {
        if (!isBound()) {
            throw new IllegalStateException("Declaring block is not built");
        }
        return block;
    }

    /**
     * Returns this value's declaring code element.
     * <p>If the value is an operation result, then the declaring code element is the operation.
     * If the value is a block parameter then the declaring code element is this value's declaring block.
     *
     * @return the value's declaring code element.
     * @throws IllegalStateException if an unbuilt block is encountered.
     */
    public CodeElement<?, ?> declaringElement() {
        return switch (this) {
            case Block.Parameter _ -> block;
            case Op.Result r -> r.op();
        };
    }

    /**
     * Returns the type of the value.
     *
     * @return the type of the value.
     */
    public TypeElement type() {
        return type;
    }

    /**
     * Returns this value as an operation result.
     *
     * @return the value as an operation result.
     * @throws IllegalStateException if the value is not an instance of an operation result.
     */
    public Op.Result result() {
        if (this instanceof Op.Result r) {
            return r;
        }
        throw new IllegalStateException("Value is not an instance of operation result");
    }

    /**
     * Returns this value as a block parameter.
     *
     * @return the value as a block parameter.
     * @throws IllegalStateException if the value is not an instance of a block parameter.
     */
    public Block.Parameter parameter() {
        if (this instanceof Block.Parameter p) {
            return p;
        }
        throw new IllegalStateException("Value is not an instance of block parameter");
    }

    /**
     * Returns the values this value directly depends on.
     * <p>
     * An operation result depends on the set of values whose members are the operation's operands and block arguments
     * of the operation's successors.
     * A block parameter does not depend on any values, and therefore this method returns an empty sequenced set.
     *
     * @return the values this value directly depends on, as an unmodifiable sequenced set. For an operation result the
     * operation's operands will occur first and then block arguments of each successor.
     */
    public abstract SequencedSet<Value> dependsOn();

    /**
     * Returns the uses of this value, specifically each operation result of an operation where this value is used as
     * an operand or as an argument of a block reference that is a successor.
     *
     * @return the uses of this value, as an unmodifiable sequenced set. The encouncter order is unspecified
     * and determined by the order in which operations are built into blocks.
     * @throws IllegalStateException if an unbuilt block is encountered.
     */
    public SequencedSet<Op.Result> uses() {
        if (!isBound()) {
            throw new IllegalStateException("Users are are not built");
        }

        return Collections.unmodifiableSequencedSet(uses);
    }

    /**
     * Returns {@code true} if this value is dominated by the given value {@code dom}.
     * <p>
     * If {@code v} and {@code dom} are in not declared in the same block then, domination is the result of
     * if the declaring block of {@code v} is dominated by the declaring block of {@code dom}.
     * <p>
     * Otherwise, if {@code v} and {@code dom} are declared in the same block then (in order):
     * <ul>
     * <li>if {@code dom} is a block parameter, then {@code v} is dominated by {@code dom}.
     * <li>if {@code v} is a block parameter, then {@code v} is <b>not</b> dominated by {@code dom}.
     * <li>otherwise, both {@code v} and {@code dom} are operation results, then {@code v} is dominated by {@code dom}
     * if {@code v} is the same as {@code dom} or {@code v} occurs after {@code dom} in the declaring block.
     * </ul>
     *
     * @param dom the dominating value
     * @return {@code true} if this value is dominated by the given value {@code dom}.
     * @throws IllegalStateException if an unbuilt block is encountered.
     */
    public boolean isDominatedBy(Value dom) {
        if (this == dom) {
            return true;
        }

        if (declaringBlock() != dom.declaringBlock()) {
            return declaringBlock().isDominatedBy(dom.declaringBlock());
        }

        // Any value is dominated by a block parameter
        if (dom instanceof Block.Parameter) {
            return true;
        } else if (this instanceof Block.Parameter) {
            return false;
        } else {
            assert this instanceof Op.Result &&
                    dom instanceof Op.Result;
            List<Op> ops = declaringBlock().ops();
            return ops.indexOf(((Op.Result) this).op()) >= ops.indexOf(((Op.Result) dom).op());
        }
    }

    /**
     * Compares two values by comparing their declaring elements.
     *
     * @apiNote
     * This method behaves is if it returns the result of the following expression but may be implemented more
     * efficiently.
     * {@snippet :
     * Comparator.comparing(Value::declaringElement, CodeElement::compare).compare(a, b)
     * }
     * @param a the first value to compare
     * @param b the second value to compare
     * @return the value {@code 0} if {@code a == b}; {@code -1} if {@code a} is less than {@code b}; and {@code -1}
     * if {@code a} is greater than {@code b}.
     * @throws IllegalArgumentException if {@code a} and {@code b} are not present in the same code model
     * @throws IllegalStateException if an unbuilt block is encountered.
     * @see CodeElement#compare
     */
    public static int compare(Value a, Value b) {
        return CodeElement.compare(a.declaringElement(), b.declaringElement());
    }

    boolean isBound() {
        return block.isBound();
    }
}
