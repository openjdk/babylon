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

import java.lang.reflect.code.descriptor.TypeDesc;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A value, that is the result of an operation or a block parameter.
 */
public abstract sealed class Value implements Comparable<Value>, CodeItem
        permits Block.Parameter, Op.Result {
    final Block block;
    final TypeDesc type;
    // @@@ In topological order?
    //     Can the representation be more efficient e.g. an array?
    final Set<Op.Result> uses;

    Value(Block block, TypeDesc type) {
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
     * @throws IllegalStateException if the declaring block is partially built
     */
    public Block declaringBlock() {
        if (!isBound()) {
            throw new IllegalStateException("Declaring block is partially constructed");
        }
        return block;
    }

    /**
     * Returns the type of the value.
     *
     * @return the type of the value.
     */
    public TypeDesc type() {
        return type;
    }

    /**
     * Returns the values this value directly depends on.
     * <p>
     * An operation result depends on the set of values whose members are the operation's operands and block arguments
     * of the operation's successors.
     * A block parameter does not depend on any values.
     *
     * @return the values this value directly depends on, as an unmodifiable set.
     */
    public abstract Set<Value> dependsOn();

    /**
     * Returns the uses of this value, specifically each operation result of an operation where this value is used as
     * an operand or as an argument of a block reference that is a successor.
     *
     * @return the uses of this value, as an unmodifiable set.
     * @throws IllegalStateException if the declaring block is partially built
     */
    public Set<Op.Result> uses() {
        if (!isBound()) {
            throw new IllegalStateException("Users are partially constructed");
        }

        return Collections.unmodifiableSet(uses);
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
     * @throws IllegalStateException if the declaring block is partially built
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


    @Override
    public int compareTo(Value o) {
        return compare(this, o);
    }

    // @@@
    public static int compare(Value v1, Value v2) {
        if (v1 == v2) return 0;

        Block b1 = v1.declaringBlock();
        Block b2 = v2.declaringBlock();
        if (b1 == b2) {
            if (v1 instanceof Op.Result or1 && v2 instanceof Op.Result or2) {
                List<Op> ops = b1.ops();
                return Integer.compare(ops.indexOf(or1.op()), ops.indexOf(or2.op()));
            } else if (v1 instanceof Op.Result) {
                // v2 instanceof BlockParameter
                return 1;
            } else if (v2 instanceof Op.Result) {
                // v1 instanceof BlockParameter
                return -1;
            } else { // v1 && v2 instanceof BlockParameter
                assert v1 instanceof Block.Parameter && v2 instanceof Block.Parameter;
                List<Block.Parameter> args = b1.parameters();
                return Integer.compare(args.indexOf(v1), args.indexOf(v2));
            }
        }

        Body r1 = b1.parentBody();
        Body r2 = b2.parentBody();
        if (r1 == r2) {
            // @@@ order should be defined by CFG and dominator relations
            List<Block> bs = r1.blocks();
            return Integer.compare(bs.indexOf(b1), bs.indexOf(b2));
        }

        Op o1 = r1.parentOp();
        Op o2 = r2.parentOp();
        if (o1 == o2) {
            List<Body> rs = o1.bodies();
            return Integer.compare(rs.indexOf(r1), rs.indexOf(r2));
        }

        return compare(o1.result(), o2.result());
    }

    boolean isBound() {
        return block.isBound();
    }
}
