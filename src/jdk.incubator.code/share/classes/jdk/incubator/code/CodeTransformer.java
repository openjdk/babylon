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
import java.util.function.Function;

/**
 * A code transformer.
 * <p>
 * Code transformer implementations are not required to be thread-safe. Code transformations operate on block builders
 * and code contexts that are not thread-safe.
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
        return (builder, inputOp) -> {
            // Allocate op builder function capturing builder and inputOp
            // This is simpler and safer that using fields holding the builder and inputOp
            // and protecting use against reentry of calls to builder.op for an attached
            // operation that is transformed and contains bodies
            // @@@ If performance is an issue consider changing
            Function<Op, Op.Result> opBuilder = outputOp -> {
                Op.Result result = builder.op(outputOp);
                builder.context().mapValue(inputOp.result(), result);
                return result;
            };

            List<Value> outputOperands = inputOp.operands().stream()
                    .map(v -> builder.context().queryValue(v).orElse(null)).toList();

            opTransformer.acceptOp(opBuilder, inputOp, outputOperands);
            return builder;
        };
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
     * The default implementation {@link #acceptBlock(Block.Builder, Block) accepts} a block builder and a block for
     * each block of the body, in order, using this code transformer. The following sequence of actions is performed:
     * <ol>
     * <li>
     * the body's entry block is mapped to the block builder, and a prefix of the body's entry block parameters is
     * mapped, in order, to the given entry values, using this builder's context. Any remaining entry block parameters
     * are not mapped;
     * <li>
     * for each input block in the body, except the entry block, an output block builder is created from the builder
     * with the same parameter types as the input block, in order. The input block is mapped to the output builder, and
     * the input block parameters are mapped to the output block parameters, using the builder's context;
     * <li>
     * for each input block in the body, in order, the input block is transformed by
     * {@link #acceptBlock(Block.Builder, Block) accepting} the mapped output builder and input block, using this code
     * transformer.
     * </ol>
     *
     * @param builder the block builder
     * @param body the body to transform
     * @param entryValues the output entry values to map, in order, from a prefix of the input body's entry block
     *                    parameters
     * @throws IllegalArgumentException if there are more output entry values than entry block parameters
     */
    default void acceptBody(Block.Builder builder, Body body, List<? extends Value> entryValues) {
        CodeContext cc = builder.context();

        // Map blocks up front, for forward referencing successors
        for (Block block : body.blocks()) {
            if (block.isEntryBlock()) {
                cc.mapBlock(block, builder);
                cc.mapValuePrefix(block.parameters(), entryValues);
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
     * Transforms one input operation into the output model.
     * <p>
     * Implementations of this method may emit zero or more output operations and output blocks into the output model by
     * using the given block builder, which is the current output block builder for this transformation, or any other
     * block builder created for this transformation. Such a block builder can be used to:
     * <ul>
     * <li>
     * emit an output operation, by using the block builder to {@link Block.Builder#op(Op) append} the operation; and
     * <li>
     * emit an output block, by using the builder to {@link Block.Builder#block(List) create} a block builder for that
     * output block and appending operations to the created block builder.
     * </ul>
     * <p>
     * Implementations can choose to drop the input operation by not emitting any output operation, copy it by
     * appending it, replace it by appending a different output operation, or expand it by appending multiple
     * output operations and creating additional block builders.
     * <p>
     * A block builder will perform <a href="Block.Builder.html#transform-on-append"><i>transform-on-append</i></a> when
     * appending the input operation, or any other attached or root operation. More specifically:
     * <ul>
     * <li>
     * If the block builder's code transformer is the same as this code transformer, as is the case for the current
     * output block builder, this code transformer may be recursively invoked for any descendant input operations.
     * <li>
     * If the appended operation has a result, the block builder's code context is used to <i>implicitly</i> establish
     * a mapping between that result and the result of the emitted output operation, if no such mapping already exists.
     * </ul>
     * <p>
     * If an implementation drops, replaces, or expands the input operation, or creates output blocks that correspond to
     * input blocks, it is responsible for <i>explicitly</i> establishing mappings between input and output values,
     * blocks, and block references that are required for the transformation of subsequent input bodies, blocks, and
     * operations. Mappings are established using a block builder's code context for this transformation. For example,
     * replacing an input operation whose result is used by a subsequent input operation requires that a mapping be
     * explicitly established between the input and output operation results; otherwise an exception may be thrown if
     * transformation later appends the subsequent operation.
     * <p>
     * An implementation returns the continuation builder to use, as the current output block builder, for subsequent
     * input operations from the same input block. The returned builder may be the current output block builder, or
     * another builder created for this transformation.
     *
     * @apiNote
     * A code transformer that copies each input operation, and as a result copies the input code model, can be
     * implemented as follows:
     * {@snippet lang = "java":
     * CodeTransformer copyingTransformer = (builder, inputOp) -> {
     *     builder.op(inputOp);
     *     return builder;
     * };
     * }
     * For convenience {@code CodeTransformer} provides such an implementation,
     * {@link CodeTransformer#COPYING_TRANSFORMER}.
     *
     * @param builder the current output block builder.
     * @param op      the input operation to transform.
     * @return the continuation builder to use to transform subsequent input operations from the same input block
     * @see Block.Builder#op(Op)
     */
    Block.Builder acceptOp(Block.Builder builder, Op op);
}
