/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.dialect.core;

import jdk.incubator.code.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static jdk.incubator.code.dialect.core.CoreOp.branch;

/**
 * Functionality for inlining code models.
 */
public final class Inliner {

    private Inliner() {
    }

    /**
     * Inlines the invokable operation into the given block builder and returns a block builder from which to
     * continue building. The invokable operation must contain at least one return operation and each return operation
     * must be an inlinable return operation, an operation whose {@link Op#ancestorOp() nearest ancestor} operation is
     * the same as the invokable operation. Otherwise, an exception is thrown.
     * <p>
     * This method {@link Block.Builder#transformBody(Body, List, CodeTransformer) transforms} the body of the invokable
     * operation with the given arguments and a code transformer that replaces inlinable return operations.
     * <p>
     * The code transformer copies all operations except the inlinable return operations.
     * <p>
     * The transformer creates a return block builder from the code transformer's block builder. If the invokable
     * operation returns a value, the return block builder has one block parameter, representing the return value, whose
     * type is the same as the invokable operation's return type. Otherwise, the return block has no block parameter.
     * <p>
     * The transformer replaces each return operation with a branch operation whose successor is a block reference to
     * the return block builder. If the return operation has an operand, since the invokable operation returns a value,
     * then the successor has an argument. The successor argument is the value mapped to the return operation's operand
     * in the code transformer's block builder's code context.
     * <p>
     * After transformation this method returns the return block builder and the return value, if any, is represented by
     * the return block builder's block parameter.
     * @apiNote
     * An invokable operation containing non-inlinable return operations may be
     * {@link CodeTransformer#LOWERING_TRANSFORMER lowered} into one that contains only inlinable return operations,
     * and therefore the lowered invokable operation can be inlined.
     *
     * @param inBlock     the block builder
     * @param invokableOp the invokable operation
     * @param args        the arguments to map, in order, from a prefix of the invokable operation's parameters
     * @param <O>         The invokable type
     * @return the block builder to continue building from, which has the same code context and code transformer as the
     * given block builder
     * @throws IllegalArgumentException if the invokable operation has no inlinable return operations
     * @throws IllegalArgumentException if the invocation operation has one or more non-inlinable return operations
     * @see CodeTransformer#LOWERING_TRANSFORMER
     */
    public static <O extends Op & Op.Invokable>
    Block.Builder inline(Block.Builder inBlock, O invokableOp, List<? extends Value> args) {
        // Find the nearest ancestor op for each return operation targeting this invokable operation
        Set<Op> collect = invokableOp.elements()
                .filter(e -> e instanceof CoreOp.ReturnOp rop
                        && getNearestInvokeableAncestorOp(rop) == invokableOp)
                .map(CodeElement::ancestorOp)
                .collect(Collectors.toSet());
        if (!collect.contains(invokableOp)) {
            throw new IllegalArgumentException("The invokable operation has no inlinable return operations");
        }
        if (collect.size() > 1) {
            throw new IllegalArgumentException("The invokable operation has one or more non-inlinable return operations");
        }

        Map<Body, Block.Builder> returnBlocks = new HashMap<>(1);
        inBlock.transformBody(invokableOp.body(), args, (block, op) -> {
            if (op instanceof CoreOp.ReturnOp rop && op.ancestorOp() == invokableOp) {
                // Compute the return block
                Block.Builder returnBlock = returnBlocks.computeIfAbsent(rop.ancestorBody(), _ -> {
                    List<CodeType> param = rop.returnValue() != null
                            ? List.of(rop.returnValue().type())
                            : List.of();
                    return block.block(param);
                });

                // Replace return op with branch to return block, with given return value
                block.add(branch(returnBlock.reference(block.context().getValues(rop.operands()))));

                return block;
            }

            block.add(op);
            return block;
        });


        Block.Builder builder = returnBlocks.get(invokableOp.body());
        assert builder != null;
        return builder.withContextAndTransformer(inBlock.context(), inBlock.transformer());
    }

    /**
     * Inlines the invokable operation into the given block builder, applying given consumer for continuation of
     * inlining. The invokable operation must contain at least one inlinable return operation, an operation that targets
     * given the invokable operation. Otherwise, an exception is thrown.
     * <p>
     * This method {@link Block.Builder#transformBody(Body, List, CodeTransformer) transforms} the body of the invokable
     * operation with the given arguments and a code transformer that replaces inlinable return operations by applying
     * a return block builder to the given consumer.
     * <p>
     * The code transformer copies all operations except inlinable return operations. When an inlinable return operation
     * is encountered, then on first encounter of its nearest ancestor body a return block builder is created and used
     * for this return operation and encounters of subsequent return operations with the same ancestor body.
     * <p>
     * The transformer creates a return block builder from the code transformer's block builder. If the invokable
     * operation returns a value, the return block builder has one block parameter, representing the return value, whose
     * type is the same as the invokable operation's return type. Otherwise, the return block has no block parameter.
     * <p>
     * The transformer replaces each return operation with a branch operation whose successor is a block reference to
     * the return block builder, and then applies the return block builder to the given consumer. If the return
     * operation has an operand, since the invokable operation returns a value, then the successor has an argument. The
     * successor argument is the value mapped to the return operation's operand in the code transformer's block
     * builder's code context.
     * @apiNote
     * An invokable operation containing inlinable return operations that cannot be inlined by continuation may be
     * {@link CodeTransformer#LOWERING_TRANSFORMER lowered} and the lowered invokable operation can be
     * {@link #inline(Block.Builder, Op, List) inlined} without continuation.
     * @see #inline(Block.Builder, Op, List)
     *
     * @param inBlock        the block builder
     * @param invokableOp    the invokable operation
     * @param args           the arguments to map, in order, from a prefix of the invokable operation's parameters
     * @param inlineConsumer the consumer applied for continuation of inlining
     * @param <O>            The invokable type
     * @throws IllegalArgumentException if the invocation operation has no inlineable return operations
     */
    public static <O extends Op & Op.Invokable>
    void inlineWithContinuation(Block.Builder inBlock, O invokableOp, List<? extends Value> args,
                                Consumer<Block.Builder> inlineConsumer) {
        // Count the number of return opertion's targeting the invokable operation
        long nInlinableReturnOps = invokableOp.elements()
                .filter(e -> e instanceof CoreOp.ReturnOp rop
                        && getNearestInvokeableAncestorOp(rop) == invokableOp)
                .count();
        if (nInlinableReturnOps == 0) {
            throw new IllegalArgumentException("The invocation operation has no inlineable return operations");
        }

        Map<Body, Block.Builder> returnBlocks = new HashMap<>();
        inBlock.transformBody(invokableOp.body(), args, (block, op) -> {
            // If the return operation is associated with the invokable operation
            if (op instanceof CoreOp.ReturnOp rop && getNearestInvokeableAncestorOp(op) == invokableOp) {
                // Compute the return block
                Block.Builder returnBlock = returnBlocks.computeIfAbsent(rop.ancestorBody(), _ -> {
                    List<CodeType> param = rop.returnValue() != null
                            ? List.of(rop.returnValue().type())
                            : List.of();
                    Block.Builder rb = block.block(param);
                    // Continue the return
                    inlineConsumer.accept(rb);

                    return rb;
                });

                // Replace return op with branch to return block, with given return value
                block.add(branch(returnBlock.reference(block.context().getValues(rop.operands()))));

                return block;
            }

            block.add(op);
            return block;
        });
    }

    private static Op getNearestInvokeableAncestorOp(Op op) {
        do {
            op = op.ancestorOp();
        } while (!(op instanceof Op.Invokable));
        return op;
    }
}
