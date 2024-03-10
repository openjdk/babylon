/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import java.lang.reflect.code.Block;
import java.lang.reflect.code.CopyContext;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.type.MethodRef;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static java.lang.reflect.code.op.CoreOps.*;
import static java.lang.reflect.code.type.JavaType.DOUBLE;

public final class ForwardDifferentiation {
    // The function to differentiate
    final FuncOp fcm;
    // The independent variable
    final Block.Parameter ind;
    // The active set for the independent variable
    final Set<Value> activeSet;
    // The map of input value to it's (output) differentiated value
    final Map<Value, Value> diffValueMapping;

    // The constant value 0.0d
    // Declared in the (output) function's entry block
    Value zero;

    private ForwardDifferentiation(FuncOp fcm, Block.Parameter ind) {
        int indI = fcm.body().entryBlock().parameters().indexOf(ind);
        if (indI == -1) {
            throw new IllegalArgumentException("Independent argument not defined by function");
        }
        this.fcm = fcm;
        this.ind = ind;

        // Calculate the active set of dependent values for the independent value
        this.activeSet = ActiveSet.activeSet(fcm, ind);
        // A mapping of input values to their (output) differentiated values
        this.diffValueMapping = new HashMap<>();
    }

    public static FuncOp partialDiff(FuncOp fcm, Block.Parameter ind) {
        return new ForwardDifferentiation(fcm, ind).partialDiff();
    }

    FuncOp partialDiff() {
        int indI = fcm.body().entryBlock().parameters().indexOf(ind);

        AtomicBoolean first = new AtomicBoolean(true);
        FuncOp dfcm = fcm.transform(STR."d\{fcm.funcName()}_darg\{indI}",
                (block, op) -> {
                    if (first.getAndSet(false)) {
                        // Initialize
                        processBlocks(block);
                    }

                    // If the result of the operation is in the active set,
                    // then differentiate it, otherwise copy it
                    if (activeSet.contains(op.result())) {
                        Value dor = diffOp(block, op);
                        // Map the input result to its (output) differentiated result
                        // so that it can be used when differentiating subsequent operations
                        diffValueMapping.put(op.result(), dor);
                    } else {
                        block.apply(op);
                    }
                    return block;
                });

        return dfcm;
    }

    void processBlocks(Block.Builder block) {
        // Declare constants at start
        zero = block.op(constant(ind.type(), 0.0d));
        // The differential of ind is 1
        Value one = block.op(constant(ind.type(), 1.0d));
        diffValueMapping.put(ind, one);

        // Append differential block parameters to blocks
        for (Value v : activeSet) {
            if (v instanceof Block.Parameter ba) {
                if (ba != ind) {
                    // Get the output block builder for the input (declaring) block
                    Block.Builder b = block.context().getBlock(ba.declaringBlock());
                    // Add a new block parameter for differential parameter
                    Block.Parameter dba = b.parameter(ba.type());
                    // Place in mapping
                    diffValueMapping.put(ba, dba);
                }
            }
        }
    }


    static final JavaType J_L_MATH = JavaType.type(Math.class);
    static final FunctionType D_D = FunctionType.functionType(DOUBLE, DOUBLE);
    static final MethodRef J_L_MATH_SIN = MethodRef.method(J_L_MATH, "sin", D_D);
    static final MethodRef J_L_MATH_COS = MethodRef.method(J_L_MATH, "cos", D_D);

    Value diffOp(Block.Builder block, Op op) {
        // Switch on the op, using pattern matching
        return switch (op) {
            case CoreOps.NegOp _ -> {
                // Copy input operation
                block.op(op);

                // -diff(expr)
                Value a = op.operands().get(0);
                Value da = diffValueMapping.getOrDefault(a, zero);
                yield block.op(neg(da));
            }
            case CoreOps.AddOp _ -> {
                // Copy input operation
                block.op(op);

                // diff(l) + diff(r)
                Value lhs = op.operands().get(0);
                Value rhs = op.operands().get(1);
                Value dlhs = diffValueMapping.getOrDefault(lhs, zero);
                Value drhs = diffValueMapping.getOrDefault(rhs, zero);
                yield block.op(add(dlhs, drhs));
            }
            case CoreOps.MulOp _ -> {
                // Copy input operation
                block.op(op);

                // Product rule
                // diff(l) * r + l * diff(r)
                Value lhs = op.operands().get(0);
                Value rhs = op.operands().get(1);
                Value dlhs = diffValueMapping.getOrDefault(lhs, zero);
                Value drhs = diffValueMapping.getOrDefault(rhs, zero);
                Value outputLhs = block.context().getValue(lhs);
                Value outputRhs = block.context().getValue(rhs);
                yield block.op(add(
                        block.op(mul(dlhs, outputRhs)),
                        block.op(mul(outputLhs, drhs))));
            }
            case CoreOps.ConstantOp _ -> {
                // Copy input operation
                block.op(op);
                // Differential of constant is zero
                yield zero;
            }
            case CoreOps.InvokeOp c -> {
                MethodRef md = c.invokeDescriptor();
                String operationName = null;
                if (md.refType().equals(J_L_MATH)) {
                    operationName = md.name();
                }
                // Differentiate sin(x)
                if ("sin".equals(operationName)) {
                    // Copy input operation
                    block.op(op);

                    // Chain rule
                    // cos(expr) * diff(expr)
                    Value a = op.operands().get(0);
                    Value da = diffValueMapping.getOrDefault(a, zero);
                    Value outputA = block.context().getValue(a);
                    Op.Result cosx = block.op(invoke(J_L_MATH_COS, outputA));
                    yield block.op(mul(cosx, da));
                } else {
                    throw new UnsupportedOperationException("Operation not supported: " + op.opName());
                }
            }
            case CoreOps.ReturnOp _ -> {
                // Replace with return of differentiated value
                Value a = op.operands().get(0);
                Value da = diffValueMapping.getOrDefault(a, zero);
                yield block.op(_return(da));
            }
            case Op.BlockTerminating _ -> {
                // Update with differentiated block arguments
                op.successors().forEach(s -> adaptSuccessor(block.context(), s));
                yield block.op(op);
            }
            default -> throw new UnsupportedOperationException("Operation not supported: " + op.opName());
        };
    }

    void adaptSuccessor(CopyContext cc, Block.Reference from) {
        List<Value> as = from.arguments().stream()
                .filter(activeSet::contains)
                .toList();
        if (!as.isEmpty()) {
            // Get the successor arguments
            List<Value> outputArgs = cc.getValues(from.arguments());
            // Append the differential arguments, if any
            for (Value a : as) {
                Value da = diffValueMapping.get(a);
                outputArgs.add(da);
            }

            // Map successor with appended arguments
            Block.Reference to = cc.getBlock(from.targetBlock()).successor(outputArgs);
            cc.mapSuccessor(from, to);
        }
    }

}
