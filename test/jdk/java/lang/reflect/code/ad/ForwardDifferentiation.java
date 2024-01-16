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
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.op.CoreOps.FuncOp;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.descriptor.MethodDesc;
import java.lang.reflect.code.descriptor.MethodTypeDesc;
import java.lang.reflect.code.descriptor.TypeDesc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import static java.lang.reflect.code.op.CoreOps._return;
import static java.lang.reflect.code.op.CoreOps.add;
import static java.lang.reflect.code.op.CoreOps.invoke;
import static java.lang.reflect.code.op.CoreOps.constant;
import static java.lang.reflect.code.op.CoreOps.mul;
import static java.lang.reflect.code.op.CoreOps.neg;
import static java.lang.reflect.code.descriptor.TypeDesc.DOUBLE;

public final class ForwardDifferentiation {
    final FuncOp f;
    final Block.Parameter ind;
    final TypeDesc indT;

    final Set<Value> activeSet;
    final Map<Value, Value> diffValueMapping;

    Value zero;

    ForwardDifferentiation(FuncOp f, Block.Parameter ind) {
        this.f = f;

        Block fb = f.body().entryBlock();
        int indI = fb.parameters().indexOf(ind);
        if (indI == -1) {
            throw new IllegalArgumentException("Independent argument not defined by function");
        }
        this.ind = ind;
        this.indT = ind.type();

        // Calculate the active set of dependent values for the independent value
        this.activeSet = ActiveSet.activeSet(f, ind);
        this.diffValueMapping = new HashMap<>();
    }

    public static FuncOp diff(FuncOp f, Block.Parameter ind) {
        return new ForwardDifferentiation(f, ind).diff();
    }

    FuncOp diff() {
        Block fb = f.body().entryBlock();
        int indI = fb.parameters().indexOf(ind);
        if (indI == -1) {
            throw new IllegalArgumentException("Independent argument not defined by function");
        }

        AtomicBoolean first = new AtomicBoolean(true);
        FuncOp cf = f.transform("d" + f.funcName() + "_darg" + indI,
                (block, op) -> {
                    if (first.getAndSet(false)) {
                        processBlocks(block);
                    }

                    if (activeSet.contains(op.result())) {
                        Value dor = diffOp(block, op);
                        diffValueMapping.put(op.result(), dor);
                    } else {
                        block.apply(op);
                    }
                    return block;
                });

        return cf;
    }

    void processBlocks(Block.Builder block) {
        // Define constants at start
        zero = block.op(constant(ind.type(), 0.0d));
        Value one = block.op(constant(ind.type(), 1.0d));
        diffValueMapping.put(ind, one);

        // Append differential block arguments to blocks
        for (Value v : activeSet) {
            if (v instanceof Block.Parameter ba) {
                if (ba != ind) {
                    Block.Builder b = block.context().getBlock(ba.declaringBlock());
                    Block.Parameter dba = b.parameter(ba.type());
                    diffValueMapping.put(ba, dba);
                }
            }
        }
    }


    static final TypeDesc J_L_MATH = TypeDesc.type(Math.class);
    static final MethodTypeDesc D_D = MethodTypeDesc.methodType(DOUBLE, DOUBLE);
    static final MethodDesc J_L_MATH_SIN = MethodDesc.method(J_L_MATH, "sin", D_D);
    static final MethodDesc J_L_MATH_COS = MethodDesc.method(J_L_MATH, "cos", D_D);

    Value diffOp(Block.Builder block, Op op) {
        return switch (op) {
            case CoreOps.NegOp _ -> {
                block.op(op);

                // -diff(expr)
                Value a = op.operands().get(0);
                Value da = diffValueMapping.getOrDefault(a, zero);

                yield block.op(neg(da));
            }
            case CoreOps.AddOp _ -> {
                block.op(op);

                // diff(l) + diff(r)
                Value lhs = op.operands().get(0);
                Value rhs = op.operands().get(1);
                Value dlhs = diffValueMapping.getOrDefault(lhs, zero);
                Value drhs = diffValueMapping.getOrDefault(rhs, zero);

                yield block.op(add(dlhs, drhs));
            }
            case CoreOps.MulOp _ -> {
                block.op(op);

                // diff(l) * r + l * diff(r)
                Value lhs = op.operands().get(0);
                Value rhs = op.operands().get(1);
                Value dlhs = diffValueMapping.getOrDefault(lhs, zero);
                Value drhs = diffValueMapping.getOrDefault(rhs, zero);

                Op.Result x1 = block.op(mul(dlhs, block.context().getValue(rhs)));
                Op.Result x2 = block.op(mul(block.context().getValue(lhs), drhs));
                yield block.op(add(x1, x2));
            }
            case CoreOps.ConstantOp _ -> {
                block.op(op);
                yield zero;
            }
            case CoreOps.InvokeOp c -> {
                MethodDesc md = c.invokeDescriptor();
                String operationName = null;
                if (md.refType().equals(J_L_MATH)) {
                    operationName = md.name();
                }
                if ("sin".equals(operationName)) {
                    block.op(op);

                    // cos(expr) * diff(expr)
                    Value a = op.operands().get(0);
                    Value da = diffValueMapping.getOrDefault(a, zero);

                    Op.Result cosx = block.op(invoke(J_L_MATH_COS, block.context().getValue(a)));
                    yield block.op(mul(cosx, da));
                } else {
                    throw new UnsupportedOperationException("Operation not supported: " + op.opName());
                }
            }
            case CoreOps.ReturnOp _ -> {
                // Replace
                Value a = op.operands().get(0);
                Value da = diffValueMapping.getOrDefault(a, zero);

                yield block.op(_return(da));
            }
            case Op.BlockTerminating _ -> {
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
            List<Value> args = cc.getValues(from.arguments());
            // Append the differential value arguments, if any
            for (Value a : as) {
                Value da = diffValueMapping.get(a);
                args.add(da);
            }

            // Map successor with appended arguments
            Block.Reference to = cc.getBlock(from.targetBlock()).successor(args);
            cc.mapSuccessor(from, to);
        }
    }

}
