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

package jdk.incubator.code;

import java.util.List;
import java.util.Objects;

/**
 * An operation transformer.
 */
@FunctionalInterface
// @@@ Change to OpTransform or CodeTransform
public interface OpTransformer {

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
     * A transformer that drops location information from operations.
     */
    OpTransformer DROP_LOCATION_TRANSFORMER = (block, op) -> {
        Op.Result r = block.op(op);
        r.op().setLocation(Location.NO_LOCATION);
        return block;
    };

    /**
     * A transformer that lowers operations that are {@link Op.Lowerable lowerable},
     * and copies other operations.
     */
    OpTransformer LOWERING_TRANSFORMER = (block, op) -> {
        if (op instanceof Op.Lowerable lop) {
            return lop.lower(block);
        } else {
            block.op(op);
            return block;
        }
    };

    /**
     * Transforms a body starting from a block builder.
     *
     * @implSpec
     * The default implementation {@link #acceptBlock(Block.Builder, Block) accepts} a block builder
     * and a block for each block of the body, in order, using this operation transformer.
     * The following sequence of actions is performed:
     * <ol>
     * <li>
     * the body's entry block is mapped to the block builder, and the (input) block parameters of the
     * body's entry block are mapped to the (output) values, using the builder's context.
     * <li>for each (input) block in the body (except the entry block) an (output) block builder is created
     * from the builder with the same parameter types as the (input) block, in order.
     * The (input) block is mapped to the (output) builder, and the (input) block parameters are mapped to the
     * (output) block parameters, using the builder's context.
     * <li>
     * for each (input) block in the body (in order) the (input) block is transformed
     * by {@link #acceptBlock(Block.Builder, Block) accepting} the mapped (output) builder and
     * (input) block, using this operation transformer.
     * </ol>
     *
     * @param builder the block builder
     * @param body the body to transform
     * @param values the values to map to the body's entry block parameters
     */
    default void acceptBody(Block.Builder builder, Body body, List<? extends Value> values) {
        CopyContext cc = builder.context();

        // Map blocks up front, for forward referencing successors
        for (Block block : body.blocks()) {
            if (block.isEntryBlock()) {
                cc.mapBlock(block, builder);
                cc.mapValues(block.parameters(), values);
            } else {
                Block.Builder blockBuilder = builder.block(block.parameterTypes());
                cc.mapBlock(block, blockBuilder);
                cc.mapValues(block.parameters(), blockBuilder.parameters());
            }
        }

        // Transform blocks
        for (Block b : body.blocks()) {
            acceptBlock(cc.getBlock(b), b);
        }
    }

    /**
     * Transforms a block starting from a block builder.
     *
     * @implSpec
     * The default implementation {@link #acceptOp(Block.Builder, Op) accepts} a block builder
     * and an operation for each operation of the block, in order, using this operation transformer.
     * On first iteration the block builder that is applied is block builder passed as an argument
     * to this method.
     * On second and subsequent iterations the block builder that is applied is the resulting
     * block builder of the prior iteration.
     *
     * @param builder the block builder
     * @param block   the block to transform
     * @throws NullPointerException if a resulting block builder is null
     */
    default void acceptBlock(Block.Builder builder, Block block) {
        for (Op op : block.ops()) {
            builder = acceptOp(builder, op);
            // @@@ See andThen composition
            Objects.requireNonNull(builder);
        }
    }

    /**
     * Transforms an operation to zero or more operations, appending those operations to a
     * block builder. Returns a block builder to be used for transforming further operations, such
     * as subsequent operations from the same block as the given operation.
     *
     * @param block the block builder.
     * @param op    the operation to transform.
     * @return      the block builder to append to for subsequent operations.
     */
    Block.Builder acceptOp(Block.Builder block, Op op);

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
                Block.Builder nbb = acceptOp(bb, o);
                if (nbb != null) {
                    return after.acceptOp(nbb, o);
                } else {
                    // @@@ This does not currently occur
                    return null;
                }
            };
        }
    }
}
