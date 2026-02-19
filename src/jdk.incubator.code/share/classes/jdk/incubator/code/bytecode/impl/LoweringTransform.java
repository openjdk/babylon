/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.bytecode.impl;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.internal.BranchTarget;

import static jdk.incubator.code.dialect.core.CoreOp.YieldOp;
import static jdk.incubator.code.dialect.core.CoreOp.branch;
import jdk.incubator.code.dialect.java.JavaType;
import static jdk.incubator.code.dialect.java.JavaType.*;

/**
 * Lowering transformer generates models supported by {@code BytecodeGenerator}.
 * Constant-labeled switch statements and switch expressions are lowered to
 * {@code ConstantLabelSwitchOp} with evaluated labels.
 */
public final class LoweringTransform {

    private static Block.Builder lowerToConstantLabelSwitchOp(Block.Builder block, CodeTransformer transformer,
                                                              JavaOp.JavaSwitchOp swOp, LabelsAndTargets labelsAndTargets) {
        List<Block> targets = labelsAndTargets.targets();
        List<Block.Builder> blocks = new ArrayList<>();
        for (int i = 0; i < targets.size(); i++) {
            Block.Builder bb = block.block();
            blocks.add(bb);
        }

        Block.Builder exit;
        if (targets.isEmpty()) {
            exit = block;
        } else {
            if (swOp.resultType() != VOID) {
                exit = block.block(swOp.resultType());
            } else {
                exit = block.block();
            }
            if (swOp instanceof JavaOp.SwitchExpressionOp) {
                exit.context().mapValue(swOp.result(), exit.parameters().get(0));
            }
        }

        BranchTarget.setBranchTarget(block.context(), swOp, exit, null);
        // map statement body to nextExprBlock
        // this mapping will be used for lowering SwitchFallThroughOp
        for (int i = 0; i < targets.size() - 1; i++) {
            BranchTarget.setBranchTarget(block.context(), targets.get(i).parent(), null, blocks.get(i + 1));
        }

        for (int i = 0; i < targets.size(); i++) {
            Block.Builder curr = blocks.get(i);
            curr.body(targets.get(i).parent(), blocks.get(i).parameters(), (b, op) -> switch (op) {
                case YieldOp _ when swOp instanceof JavaOp.SwitchStatementOp -> {
                    b.op(branch(exit.successor()));
                    yield b;
                }
                case YieldOp yop when swOp instanceof JavaOp.SwitchExpressionOp -> {
                    b.op(branch(exit.successor(b.context().getValue(yop.yieldValue()))));
                    yield b;
                }
                default -> transformer.acceptOp(b, op);
            });
        }

        Value selector = block.context().getValue(swOp.operands().get(0));
        if (isIntegralReferenceType(selector.type())) {
            // unbox selector
            if (selector.type().equals(J_L_CHARACTER)) {
                selector = block.op(JavaOp.invoke(MethodRef.method(selector.type(), "charValue", JavaType.CHAR), selector));
            } else {
                selector = block.op(JavaOp.invoke(MethodRef.method(selector.type(), "intValue", JavaType.INT), selector));
            }
        }
        var labels = labelsAndTargets.labels();
        if (!labels.contains(null)) {
            // implicit default to exit
            labels.add(null);
            blocks.add(exit);
        }
        block.op(new ConstantLabelSwitchOp(selector, labels, blocks.stream().map(Block.Builder::successor).toList()));
        return exit;
    }

    public static CodeTransformer getInstance(MethodHandles.Lookup lookup) {
        return (block, op) -> switch (op) {
            case JavaOp.JavaSwitchOp swOp -> {
                Optional<LabelsAndTargets> opt = isCaseConstantSwitchWithIntegralSelector(swOp, lookup);
                if (opt.isPresent()) {
                    yield lowerToConstantLabelSwitchOp(block, block.transformer(), swOp, opt.get());
                }
                yield swOp.lower(block, null);
            }
            case Op.Lowerable lop -> lop.lower(block, null);
            default -> {
                block.op(op);
                yield block;
            }
        };
    }

    record LabelsAndTargets(List<Integer> labels, List<Block> targets) {}

    public static Optional<LabelsAndTargets> isCaseConstantSwitchWithIntegralSelector(JavaOp.JavaSwitchOp swOp, MethodHandles.Lookup lookup) {
        //@@@ we only check for case constant switch that has integral type
        // so labels in model / source code will be identical to the one we will find in bytecode
        if (!isIntegralType(swOp.operands().get(0).type())) {
            return Optional.empty();
        }
        var labels = new ArrayList<Integer>();
        var targets = new ArrayList<Block>();
        for (int i = 0; i < swOp.bodies().size(); i += 2) {
            Body label = swOp.bodies().get(i);
            List<Integer> ls = isCaseConstantLabel(lookup, label);
            if (ls.isEmpty()) {
                return Optional.empty();
            }
            labels.addAll(ls);
            targets.addAll(Collections.nCopies(ls.size(), swOp.bodies().get(i + 1).entryBlock()));
        }
        return Optional.of(new LabelsAndTargets(labels, targets));
    }

    private static List<Integer> isCaseConstantLabel(MethodHandles.Lookup l, Body label) {
        List<Integer> empty = new ArrayList<>();
        if (label.blocks().size() != 1 || !(label.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop) ||
                !(yop.yieldValue() instanceof Op.Result r)) {
            return empty;
        }
        List<Integer> labels = new ArrayList<>();
        // we can yield a list
        MethodRef objectsEquals = MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class);
        switch (r.op()) {
            case JavaOp.EqOp eqOp -> {
                Optional<Object> v = JavaOp.JavaExpression.evaluate(l, eqOp.operands().get(1));
                v.ifPresent(o -> labels.add(toInteger(o)));
            }
            case JavaOp.InvokeOp ie when ie.invokeDescriptor().equals(objectsEquals) -> {
                Value toEvaluate;
                if (ie.operands().getLast() instanceof Op.Result opr && opr.op() instanceof JavaOp.InvokeOp ib
                        && isBoxingMethod(ib.invokeDescriptor())) {
                    // workaround the modeling of switch that has a selector of type Box and a case constant of type primitive
                    // we skip the boxing operation that's contained in the model
                    // invoking a boxing method is not a valid operation in a constant expr
                    toEvaluate = ib.operands().getFirst();
                } else {
                    toEvaluate = ie.operands().getLast();
                }
                Optional<Object> v = JavaOp.JavaExpression.evaluate(l, toEvaluate);
                v.ifPresent(o -> labels.add(toInteger(o)));
            }
            case JavaOp.ConditionalOrOp cor -> {
                for (Body corb : cor.bodies()) {
                    List<Integer> corbl = isCaseConstantLabel(l, corb);
                    if (corbl.isEmpty()) {
                        return empty;
                    }
                    labels.addAll(corbl);
                }
            }
            case CoreOp.ConstantOp cop when cop.value() instanceof Boolean b && b -> {
                labels.add(null);
            }
            default -> {
            }
        }
        return labels;
    }

    private static boolean isBoxingMethod(MethodRef mr) {
        return List.of(J_L_BYTE, J_L_CHARACTER, J_L_SHORT, J_L_INTEGER, J_L_LONG, J_L_FLOAT, J_L_DOUBLE).contains(mr.refType())
                && mr.name().equals("valueOf");
    }

    private static boolean isIntegralType(TypeElement te) {
        return isIntegralPrimitiveType(te) || isIntegralReferenceType(te);
    }

    private static boolean isIntegralPrimitiveType(TypeElement te) {
        return List.of(BYTE, SHORT, CHAR, INT).contains(te);
    }

    private static boolean isIntegralReferenceType(TypeElement te) {
        return List.of(J_L_BYTE, J_L_SHORT, J_L_CHARACTER, J_L_INTEGER).contains(te);
    }

    private static Integer toInteger(Object o) {
        return switch (o) {
            case Byte b -> Integer.valueOf(b);
            case Short s -> Integer.valueOf(s);
            case Character c -> Integer.valueOf(c);
            case Integer i -> i;
            default -> throw new IllegalStateException(); // @@@ not going to happen
        };
    }
}
