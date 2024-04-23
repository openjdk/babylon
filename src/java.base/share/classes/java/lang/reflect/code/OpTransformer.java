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

import java.util.Objects;
import java.util.function.BiFunction;

/**
 * An operation transformer.
 */
@FunctionalInterface
public interface OpTransformer extends BiFunction<Block.Builder, Op, Block.Builder> {
    /**
     * A copying transformer that applies the operation to the block builder, and returning the block builder.
     */
    OpTransformer COPYING_TRANSFORMER = (block, op) -> {
        block.op(op);
        return block;
    };

    /**
     * A transformer that performs no action on the block builder.
     */
    OpTransformer NOOP_TRANSFORMER = (block, op) -> block;

    /**
     * A transformer that removes location information from operations.
     */
    OpTransformer REMOVE_LOCATION = (block, op) -> {
        Op.Result r = block.op(op);
        r.op().setLocation(Location.NO_LOCATION);
        return block;
    };

    /**
     * Transforms a given operation to zero or more other operations appended to the
     * given block builder. Returns a block builder to be used for appending further operations, such
     * as subsequent operations from the same block as the given operation.
     *
     * @param block the block builder.
     * @param op    the operation to transform.
     * @return      the block builder to append to for subsequent operations to transform that have same parent block.
     */
    Block.Builder apply(Block.Builder block, Op op);

    /**
     * Transforms a given block to zero or more operations appended to the given block builder.
     *
     * @implSpec
     * The default implementation iterates through each operation of the block to transform
     * and {@link #apply(Block.Builder, Op) applies} a block builder and the operation to this
     * transformer.
     * On first iteration the block builder that is applied is block builder passed as an argument
     * to this method.
     * On second and subsequent iterations the block builder that is applied is the resulting
     * block builder of the prior iteration.
     *
     * @param block the block builder
     * @param b     the block to transform
     * @throws NullPointerException if a resulting block builder is null
     */
    default void apply(Block.Builder block, Block b) {
        for (Op op : b.ops()) {
            block = apply(block, op);
            // @@@ See andThen composition
            Objects.requireNonNull(block);
        }
    }

    default OpTransformer compose(OpTransformer before) {
        return before.andThen(this);
    }

    default OpTransformer andThen(OpTransformer after) {
        if (after == NOOP_TRANSFORMER) {
            return this;
        } else if (this == NOOP_TRANSFORMER) {
            return after;
        } else {
            return (bb, o) -> {
                Block.Builder nbb = apply(bb, o);
                if (nbb != null) {
                    return after.apply(nbb, o);
                } else {
                    // @@@ This does not currently occur
                    return null;
                }
            };
        }
    }
}
