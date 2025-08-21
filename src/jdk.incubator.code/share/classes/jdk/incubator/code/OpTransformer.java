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
import java.util.function.BiFunction;

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
            return lop.lower(block, null);
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
     */
    default void acceptBlock(Block.Builder builder, Block block) {
        for (Op op : block.ops()) {
            builder = acceptOp(builder, op);
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

    /**
     * Returns a composed code transform that transforms an operation that first applies
     * a block builder and operation to the function {@code f}, and then applies
     * the resulting block builder and the same operation to {@link OpTransformer#acceptOp acceptOp}
     * of the code transformer {@code after}.
     * <p>
     * If the code transformer {@code after} is {@code null} then it is not applied.
     *
     * @param after the code transformer to apply after
     * @param f the operation transformer function to apply before
     * @return the composed code transformer
     */
    static OpTransformer compose(OpTransformer after, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return after == null
                ? f::apply
                : (block, op) -> after.acceptOp(f.apply(block, op), op);
    }

    /**
     * Returns a composed code transformer that first applies a block builder and operation to
     * {@link OpTransformer#acceptOp acceptOp} of the code transformer {@code after},
     * and then applies resulting block builder and the same operation to the function {@code f}.
     * <p>
     * If the code transformer {@code before} is {@code null} then it is as if a code transformer
     * is applied that does nothing except return the block builder it was given.
     *
     * @param before the code transformer to apply before
     * @param f the operation transformer function to apply after
     * @return the composed code transformer
     */
    static OpTransformer andThen(OpTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return before == null
                ? f::apply
                : (block, op) -> f.apply(before.acceptOp(block, op), op);
    }

    /**
     * Returns a composed code transformer that composes with an operation transformer function adapted to lower
     * operations.
     * <p>
     * This method behaves as if it returns the result of the following expression:
     * {@snippet lang=java :
     * andThen(before, lowering(before, f));
     * }
     *
     * @param before the code transformer to apply before
     * @param f the operation transformer function to apply after
     * @return the composed code transformer
     */
    static OpTransformer andThenLowering(OpTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return andThen(before, lowering(before, f));
    }

    /**
     * Returns an adapted operation transformer function that adapts an operation transformer function
     * {@code f} to also transform lowerable operations.
     * <p>
     * The adapted operation transformer function first applies a block builder and operation
     * to the operation transformer function {@code f}.
     * If the result is not {@code null} then the result is returned.
     * Otherwise, if the operation is a lowerable operation then the result of applying the
     * block builder and code transformer {@code before} to {@link jdk.incubator.code.Op.Lowerable#lower lower}
     * of the lowerable operation is returned.
     * Otherwise, the operation is copied by applying it to {@link Block.Builder#op op} of the block builder,
     * and the block builder is returned.
     *
     * @param before the code transformer to apply for lowering
     * @param f the operation transformer function to apply after
     * @return the adapted operation transformer function
     */
    static BiFunction<Block.Builder, Op, Block.Builder> lowering(OpTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return (block, op) -> {
            Block.Builder b = f.apply(block, op);
            if (b == null) {
                if (op instanceof Op.Lowerable lop) {
                    block = lop.lower(block, before);
                } else {
                    block.op(op);
                }
            } else {
                block = b;
            }
            return block;
        };
    }
}
