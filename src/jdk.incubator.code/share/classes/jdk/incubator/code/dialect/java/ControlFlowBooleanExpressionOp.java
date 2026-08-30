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
 * simpler control flow graphs
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
                    assert op instanceof ControlFlowBooleanExpressionOp;
                    isSuffixProcessed[0] = true;

                    ConditionalBranchContinuation expressionContinuation = suffix.continuationForExpression(block, continuation);
                    // Push expression continuation as implicit argument to lower method
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
     * Represents the continuation of a boolean result
     */
    sealed interface BooleanResultContinuation {
        /**
         * @param block the block to add a block terminating operation
         * @param result the boolean result.
         */
        void continueWith(Block.Builder block, Value result);

        /**
         * @param block the block to add any necessary operations
         * @param result the static boolean result
         * @return
         */
        Block.Reference referenceFor(Block.Builder block, boolean result);
    }

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
    }

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
        ConditionalBranchContinuation continuationForExpression(Block.Builder block, BooleanResultContinuation continuation) {
            Block.Reference trueRef = continuation.referenceFor(block, !negatesExpression);
            Block.Reference falseRef = continuation.referenceFor(block, negatesExpression);
            return new ConditionalBranchContinuation(trueRef, falseRef);
        }
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
