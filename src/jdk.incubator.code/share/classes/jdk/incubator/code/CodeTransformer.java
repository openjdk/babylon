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

import java.util.List;
import java.util.function.Function;

/**
 * A code transformer.
 * <p>
 * A code transformer transforms an input code model into an output code model. It traverses the input code model and
 * builds the output code model by accepting input bodies, blocks, and operations and emitting output blocks and
 * operations using block builders.
 * <p>
 * During transformation, a block builder may serve as the current output block builder when accepting bodies, blocks,
 * and operations. A transformation emits an output operation by appending it using an output block builder.
 * A transformation emits an output block by creating a block builder for that block and appending operations using that
 * block builder. Emission is commonly performed by implementations of {@link #acceptOp(Block.Builder, Op)}, and less
 * commonly so by implementations that override {@link #acceptBlock(Block.Builder, Block)} and
 * {@link #acceptBody(Block.Builder, Body, List)}.
 * <p>
 * By default, traversal transforms an input body by transforming each input block of the body, in order, and transforms
 * an input block by transforming each input operation of the block, in order. The single abstract method
 * {@link #acceptOp(Block.Builder, Op)} is the primitive transformation step. Implementations of that method may emit
 * output blocks and operations. Appending a placed operation using an output block builder may recursively invoke the
 * code transformer for descendant code elements.
 * <p>
 * A transformation uses the {@link CodeContext} of an output block builder to record correspondence between input
 * code items and outputs. Some mappings are established implicitly: by the default traversal of an input body,
 * for mappings between input blocks and output block builders and for mappings between input block parameters and
 * output values; and by appending placed operations, for mappings between appended operation results and output
 * operation results. By default, block reference mappings are never established implicitly.
 * <p>
 * Transformations that drop, replace, or expand input code elements are responsible for explicitly establishing any
 * mappings required by the transformation of subsequent code elements.
 * <p>
 * Code transformer implementations are not required to be thread-safe. Code transformations operate on block builders
 * and code contexts that are not thread-safe.
 *
 * @see Block.Builder#block(List)
 * @see Block.Builder#op(Op)
 * @see CodeContext
 */
@FunctionalInterface
public interface CodeTransformer {

    /**
     * A simplified transformer for transforming one input operation into an output code model.
     */
    @FunctionalInterface
    interface OpTransformer {
        /**
         * Transforms one input operation into the output code model.
         * <p>
         * Implementations of this method may emit zero or more output operations into the output model by applying the
         * given operation-building function to operations.
         * <p>
         * Implementations can choose to drop the input operation by not applying the function, copy it by applying the
         * function to the input operation, replace it by applying the function to a different output operation, or
         * expand it by applying the function to multiple output operations.
         * <p>
         * Each application of the operation-building function implicitly maps the input operation's result to the
         * result returned by the function, which is the result of the emitted output operation. A later application
         * replaces the mapping established by any prior application. If the function is not applied, no mapping is
         * established for the input operation's result.
         * <p>
         * The given operands list contains, in order, the output values currently mapped from the input operation's
         * operands. It has the same number of values as the input operation's operands, but if an input operand
         * has no mapped output value, the corresponding output value is {@code null}.
         * <p>
         * The operation-building function encapsulates the current output block builder for this transformation step.
         * Applying the function {@link Block.Builder#op(Op) appends} an operation using the current output block
         * builder, which will perform <a href="Block.Builder.html#transform-on-append"><i>transform-on-append</i></a>
         * when appending the input operation, or any other placed operation.
         *
         * @param builder  the operation-building function
         * @param op       the input operation to transform
         * @param operands the mapped output values for the input operation's operands
         */
        void acceptOp(Function<Op, Op.Result> builder, Op op, List<Value> operands);
    }

    /**
     * Creates a code transformer that transforms operations using the given operation transformer.
     * <p>
     * This method is intended for simplified transformations that only emit output operations using the current output
     * block builder. Transformations that need to emit output blocks, explicitly establish mappings, customize body or
     * block traversal, or return a different continuation builder should directly implement {@code CodeTransformer}.
     * <p>
     * The created code transformer uses the default {@link #acceptBody(Block.Builder, Body, List)} and
     * {@link #acceptBlock(Block.Builder, Block)} traversal. Its {@link #acceptOp(Block.Builder, Op)} implementation
     * invokes the given operation transformer with an operation-building function for the current output block builder,
     * the input operation, and output values currently mapped from the input operation's operands.
     * <p>
     * Applying the operation-building function appends the operation using the current output block builder and maps
     * the input operation's result to the result of the appended operation, as specified by
     * {@link OpTransformer#acceptOp(Function, Op, List)}.
     * <p>
     * After the operation transformer returns, the created code transformer returns the current output block builder as
     * the continuation builder for the next input operation.
     *
     * @param opTransformer the operation transformer.
     * @return the code transformer that transforms operations.
     */
    static CodeTransformer opTransformer(OpTransformer opTransformer) {
        return (builder, inputOp) -> {
            // Allocate operation-building function capturing builder and inputOp
            // This is simpler and safer that using fields holding the builder and inputOp
            // and protecting use against reentry of calls to builder.op for a placed
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
     * A copying transformer that appends the operation using the block builder, and returns the block builder.
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
     * Transforms one input body into the output model.
     *
     * @implSpec
     * The default implementation first establishes mappings, and then transforms each input block of the body, in
     * order, using this code transformer.
     * <p>
     * The default implementation first <i>implicitly</i> establishes block mappings and value mappings for block
     * parameters using the given block builder's code context, as follows:
     * <ul>
     * <li>
     * The entry block of the input body is mapped to the given block builder, which is the current output block builder
     * for the entry block. A prefix of the entry block's parameters is mapped, in order, to the given entry values.
     * Any remaining entry block parameters are unmapped.
     * <li>
     * For each non-entry block of the input body, an output block builder is created from the given block builder with
     * the same sequence of parameter types as the input block. The input block is mapped to that output block builder,
     * and the input block parameters are mapped exactly to the output block builder's parameters.
     * </ul>
     * Then, for each input block, in order, the default implementation invokes
     * {@link #acceptBlock(Block.Builder, Block)} with the output block builder mapped from the input block and the
     * input block.
     *
     * @param builder the current output block builder for the input body's entry block
     * @param body the input body to transform
     * @param entryValues the output entry values to map, in order, from a prefix of the input body's entry block
     *                    parameters
     * @throws IllegalArgumentException if there are more output entry values than entry block parameters
     * @see #acceptBlock(Block.Builder, Block)
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
     * Transforms one input block into the output model.
     *
     * @implSpec
     * The default implementation transforms each input operation of the given block, in order, using this code
     * transformer.
     * <p>
     * The given block builder is the current output block builder for the first input operation. For each input
     * operation, the default implementation invokes {@link #acceptOp(Block.Builder, Op)} with the current output block
     * builder and the input operation. The continuation builder returned by that invocation becomes the current output
     * block builder for the next input operation from the same input block.
     *
     * @param builder the current output block builder
     * @param block   the input block to transform
     * @see #acceptOp(Block.Builder, Op)
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
     * output block and appending operations using the created block builder.
     * </ul>
     * <p>
     * Implementations can choose to drop the input operation by not emitting any output operation, copy it by
     * appending it, replace it by appending a different output operation, or expand it by appending multiple
     * output operations and creating additional block builders.
     * <p>
     * A block builder will perform <a href="Block.Builder.html#transform-on-append"><i>transform-on-append</i></a> when
     * appending the input operation, or any other placed operation. More specifically:
     * <ul>
     * <li>
     * if the block builder's code transformer is the same as this code transformer, as is the case for the current
     * output block builder, this code transformer may be recursively invoked for any descendant input operations.
     * <li>
     * if the appended operation has a result, the block builder's code context is used to <i>implicitly</i> establish
     * a mapping between that result and the result of the emitted output operation, if no such mapping already exists.
     * </ul>
     * <p>
     * If an implementation drops, replaces, or expands the input operation, or creates output blocks that correspond to
     * input blocks, it is responsible for <i>explicitly</i> establishing mappings between input and output values,
     * blocks and block references, that are required for the transformation of subsequent input bodies, blocks, and
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
     * The call to {@code builder.op(inputOp)} performs transform-on-append. If the input operation has descendant code
     * elements, this code transformer is recursively invoked to transform those elements. The result is that all input
     * code elements are copied into the output model.
     * <p>
     * For convenience {@code CodeTransformer} provides such an implementation,
     * {@link CodeTransformer#COPYING_TRANSFORMER}.
     *
     * @param builder the current output block builder.
     * @param op      the input operation to transform.
     * @return the continuation builder to use to transform subsequent input operations from the same input block
     * @see Block.Builder#op(Op)
     * @see CodeTransformer#COPYING_TRANSFORMER
     */
    Block.Builder acceptOp(Block.Builder builder, Op op);
}
