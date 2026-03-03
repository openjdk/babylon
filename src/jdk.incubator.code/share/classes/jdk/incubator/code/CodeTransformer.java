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
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * A code transformer.
 */
@FunctionalInterface
public interface CodeTransformer {

    /**
     * A simplified transformer for only transforming operations.
     */
    @FunctionalInterface
    interface OpTransformer {
        /**
         * Transforms an operation to zero or more operations.
         *
         * @param op       the operation to transform
         * @param operands the operands of the operation mapped to
         *                 values in the transformed code model. If
         *                 there is no mapping of an operand then it is
         *                 mapped to {@code null}.
         * @param builder  the function to apply zero or more operations
         *                 into the transformed code model
         */
        void acceptOp(Function<Op, Op.Result> builder, Op op, List<Value> operands);
    }

    /**
     * Creates a code transformer that transforms operations using the
     * given operation transformer.
     *
     * @param opTransformer the operation transformer.
     * @return the code transformer that transforms operations.
     */
    static CodeTransformer opTransformer(OpTransformer opTransformer) {
        final class CodeTransformerOfOps implements CodeTransformer, Function<Op, Op.Result> {
            Block.Builder builder;
            Op op;

            @Override
            public Block.Builder acceptOp(Block.Builder builder, Op op) {
                this.builder = builder;
                this.op = op;
                // Use null if there is no mapping in the output
                // This can happen if an operation is removed by not applying
                // it to the builder
                List<Value> operands = op.operands().stream()
                        .map(v -> builder.context().getValueOrDefault(v, null)).toList();
                opTransformer.acceptOp(this, op, operands);
                return builder;
            }

            @Override
            public Op.Result apply(Op op) {
                Op.Result result = builder.op(op);
                builder.context().mapValue(this.op.result(), result);
                return result;
            }
        }
        return new CodeTransformerOfOps();
    }

    /**
     * A copying transformer that applies the operation to the block builder, and returning the block builder.
     */
    CodeTransformer COPYING_TRANSFORMER = (builder, op) -> {
        builder.op(op);
        return builder;
    };

    /**
     * A transformer that drops location information from operations.
     */
    CodeTransformer DROP_LOCATION_TRANSFORMER = (builder, op) -> {
        Op.Result r = builder.op(op);
        r.op().setLocation(Op.Location.NO_LOCATION);
        return builder;
    };

    /**
     * A transformer that lowers operations that are {@link Op.Lowerable lowerable},
     * and copies other operations.
     */
    CodeTransformer LOWERING_TRANSFORMER = (builder, op) -> {
        if (op instanceof Op.Lowerable lop) {
            return lop.lower(builder, null);
        } else {
            builder.op(op);
            return builder;
        }
    };

    /**
     * Transforms a body starting from a block builder.
     *
     * @implSpec
     * The default implementation {@link #acceptBlock(Block.Builder, Block) accepts} a block builder
     * and a block for each block of the body, in order, using this code transformer.
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
     * (input) block, using this code transformer.
     * </ol>
     *
     * @param builder the block builder
     * @param body the body to transform
     * @param values the values to map to the body's entry block parameters
     */
    default void acceptBody(Block.Builder builder, Body body, List<? extends Value> values) {
        CodeContext cc = builder.context();

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
     * and an operation for each operation of the block, in order, using this code transformer.
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
     * @param builder the block builder.
     * @param op      the operation to transform.
     * @return the block builder to append to for subsequent operations.
     */
    Block.Builder acceptOp(Block.Builder builder, Op op);

    /**
     * Returns a composed code transform that transforms an operation that first applies
     * a block builder and operation to the function {@code f}, and then applies
     * the resulting block builder and the same operation to {@link CodeTransformer#acceptOp acceptOp}
     * of the code transformer {@code after}.
     * <p>
     * If the code transformer {@code after} is {@code null} then it is as if a code transformer
     * is applied that does nothing except return the block builder it was given.
     *
     * @param after the code transformer to apply after
     * @param f the operation transformer function to apply before
     * @return the composed code transformer
     */
    static CodeTransformer compose(CodeTransformer after, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return after == null
                ? f::apply
                : (builder, op) -> after.acceptOp(f.apply(builder, op), op);
    }

    /**
     * Returns a composed code transformer that first applies a block builder and operation to
     * {@link CodeTransformer#acceptOp acceptOp} of the code transformer {@code before},
     * and then applies resulting block builder and the same operation to the function {@code f}.
     * <p>
     * If the code transformer {@code before} is {@code null} then it is as if a code transformer
     * is applied that does nothing except return the block builder it was given.
     *
     * @param before the code transformer to apply before
     * @param f the operation transformer function to apply after
     * @return the composed code transformer
     */
    static CodeTransformer andThen(CodeTransformer before, BiFunction<Block.Builder, Op, Block.Builder> f) {
        return before == null
                ? f::apply
                : (builder, op) -> f.apply(before.acceptOp(builder, op), op);
    }
}
