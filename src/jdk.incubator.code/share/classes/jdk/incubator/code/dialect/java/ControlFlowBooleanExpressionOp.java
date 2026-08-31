/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.dialect.java;

import jdk.incubator.code.*;
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.*;
import java.util.function.BiFunction;

import static jdk.incubator.code.Op.Lowerable.loweringTransformer;

/**
 * An operation characteristic indicating that an operation can model a boolean expression whose evaluation contains
 * control flow. Such operations can lower themselves using a supplied continuation for its boolean result to produce
 * simpler control flow graphs.
 * <p>
 * For example, consider the following code, in which a boolean expression is used by a {@code while} statement:
 * {@snippet lang = java:
 * while (a && (b || c)) {
 *     ...
 * }
 * }
 * The result of the {@code while} loop's boolean expression determines whether the loop body is executed or the loop
 * finishes. If {@code a} is {@code true} and {@code b} or {@code c} is {@code true} then the loop body is executed.
 * Conversely, if {@code a} is {@code false} or {@code b} and {@code c} is {@code false} then the loop finishes.
 * <p>
 * The code model for this code contains a while operation, modeling the {@code while} statement. When the while
 * operation lowers itself it creates a boolean result continuation containing block references to the blocks associated
 * with start of executing the loop body and the loop finishing, referred to respectively as true and false branch
 * references.
 * That boolean result continuation is used when lowering operation's predicate body, which models the boolean
 * expression, the conditional-and operation modeling the conditional-and expression. The continuation is passed along
 * when lowering the sub-expressions, and when needed the continuation is operated on to continue with a boolean result
 * or to obtain block references for a statically known boolean result. Consequently, the lowering of the boolean
 * expression will directly branch to the while operation's continuation's true and false branch references. This is far
 * more preferable than creating localized control flow behavior that joins to blocks whose boolean block parameter
 * represents an intermediate boolean result; a result that is then used by a conditional branch operations to continue
 * towards the blocks of the lowered while operation.
 */
interface ControlFlowBooleanExpressionOp extends Op.Lowerable {

    ContextStackArg<BooleanResultContinuation> BOOLEAN_CONTINUATION_ARG = new ContextStackArg<>();

    static void lowerBooleanBody(
            Block.Builder startBlock,
            Body body,
            List<? extends Value> entryValues,
            BooleanResultContinuation continuation,
            BiFunction<Block.Builder, Op, Block.Builder> inherited) {
        BooleanExpressionSuffix suffix = findBooleanExpressionSuffix(body);
        CodeTransformer codeTransformer;
        if (suffix != null) {
            boolean[] isSuffixProcessed = new boolean[1];
            codeTransformer = loweringTransformer(inherited, (block, op) -> {
                if (op == suffix.expression) {
                    // Process the boolean expression
                    isSuffixProcessed[0] = true;

                    ConditionalBranchContinuation expressionContinuation =
                            continuation.forExpression(block, suffix.negatesExpression);
                    // Pass expression continuation as additional implicit argument
                    BOOLEAN_CONTINUATION_ARG.push(block.context(), expressionContinuation);
                    return suffix.expression.lower(block, inherited);
                } else if (!isSuffixProcessed[0]) {
                    // Process any operation in the prefix
                    return null;
                } else {
                    // Ignore any operation in the suffix after ControlFlowBooleanExpressionOp
                    return block;
                }
            });
        } else {
            codeTransformer = loweringTransformer(inherited, (block, op) -> {
                if (op instanceof CoreOp.YieldOp yield) {
                    Value booleanResult = block.context().getValue(yield.yieldValue());
                    continuation.continueWith(block, booleanResult);
                    return block;
                } else {
                    return null;
                }
            });
        }
        startBlock.transformBody(body, entryValues, codeTransformer);
    }

    /**
     * Represents the continuation of a boolean result.
     * <p>
     * If the lowering of a boolean expression operation needs to continue with expression's boolean result value,
     * then it can invoke {@link #continueWith(Block.Builder, Value) continueWith}. Otherwise, if lowering statically
     * knows the value of the expression and requires a block reference targeting a continuing block corresponding to
     * that known value , then it can invoke {@link #referenceFor(Block.Builder, boolean) referenceFor}.
     */
    sealed interface BooleanResultContinuation
            permits ConditionalBranchContinuation, BranchWithArgumentContinuation {
        /**
         * Continues with the result of a boolean expression.
         *
         * @param block the block to add a block terminating operation
         * @param result the boolean result.
         */
        void continueWith(Block.Builder block, Value result);

        /**
         * Creates a block reference to branch to continue the result of the boolean expression when the result is
         * statically known.
         *
         * @param block the block to add any necessary operations
         * @param result the static boolean result
         * @return the block reference to continue the result
         */
        Block.Reference referenceFor(Block.Builder block, boolean result);

        /**
         * Creates a conditional branch continuation from this continuation to be used as the continuation of a boolean
         * expression.
         *
         * @param negatesExpression true if the result of the expression is negated
         * @return the conditional branch continuation
         */
        default ConditionalBranchContinuation forExpression(Block.Builder block, boolean negatesExpression) {
            Block.Reference trueRef = referenceFor(block, !negatesExpression);
            Block.Reference falseRef = referenceFor(block, negatesExpression);
            return new ConditionalBranchContinuation(trueRef, falseRef);
        }
    }

    /**
     * Represents a boolean result continuation as block references to blocks corresponding to continuing the
     * {@code true} result and the {@code false} result.
     *
     * @param trueRef the block reference to a block corresponding to continuing the {@code true} result
     * @param falseRef the block reference to a block corresponding to continuing the {@code false} result
     */
    record ConditionalBranchContinuation(
            Block.Reference trueRef,
            Block.Reference falseRef) implements BooleanResultContinuation {
        @Override
        public void continueWith(Block.Builder block, Value result) {
            // result will be present in a model being built, so only its operation structure can be queried
            if (result instanceof Op.Result opResult
                    && opResult.op() instanceof CoreOp.ConstantOp constant
                    && constant.value() instanceof Boolean booleanValue) {
                block.add(CoreOp.branch(
                        booleanValue ? trueRef : falseRef));
            } else {
                block.add(CoreOp.conditionalBranch(result, trueRef, falseRef));
            }
        }

        @Override
        public Block.Reference referenceFor(Block.Builder block, boolean result) {
            return result ? trueRef : falseRef;
        }

        @Override
        public ConditionalBranchContinuation forExpression(Block.Builder block, boolean negatesExpression) {
            return !negatesExpression
                    ? this
                    : new ConditionalBranchContinuation(falseRef, trueRef);
        }
    }

    /**
     * Represents a boolean result continuation as a result block with a boolean parameter, whose value corresponds to
     * continuing the {@code true} result and the {@code false} result.
     *
     * @param resultBlock the result block with a boolean parameter corresponding to continuing the {@code true}
     *                    and {@code false} result
     */
    record BranchWithArgumentContinuation(Block.Builder resultBlock) implements BooleanResultContinuation {
        @Override
        public void continueWith(Block.Builder block, Value result) {
            block.add(CoreOp.branch(resultBlock.reference(result)));
        }

        @Override
        public Block.Reference referenceFor(Block.Builder block, boolean result) {
            return resultBlock.reference(block.add(CoreOp.constant(JavaType.BOOLEAN, result)));
        }
    }

    record BooleanExpressionSuffix(ControlFlowBooleanExpressionOp expression, boolean negatesExpression) {
    }

    private static BooleanExpressionSuffix findBooleanExpressionSuffix(Body body) {
        if (body.blocks().size() != 1) {
            return null;
        }

        Block block = body.entryBlock();

        // Find suffix of
        //  ControlFlowBooleanExpressionOp
        //  NotOp *
        //  CoreOp.YieldOp

        Op.Terminating yop = block.terminatingOp();
        if (!(yop instanceof CoreOp.YieldOp)) {
            return null;
        }

        boolean negatesExpression = false;
        Op next = yop;
        Op expression = null;
        List<Op> ops = block.ops();
        for (int i = ops.size() - 2; i >= 0; i--) {
            Op op = ops.get(i);

            if (next.operands().isEmpty() || next.operands().getFirst() != op.result()) {
                return null;
            } else if (op instanceof JavaOp.NotOp) {
                negatesExpression = !negatesExpression;
            } else if (isSupportedBooleanExpressionOp(op)) {
                expression = op;
                break;
            } else {
                break;
            }

            next = op;
        }

        return expression != null
                ? new BooleanExpressionSuffix((ControlFlowBooleanExpressionOp) expression, negatesExpression)
                : null;
    }

    private static boolean isSupportedBooleanExpressionOp(Op op) {
        return op instanceof ControlFlowBooleanExpressionOp && op.resultType().equals(JavaType.BOOLEAN);
    }
}
