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
import java.lang.reflect.AccessFlag;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import jdk.incubator.code.Block;
import jdk.incubator.code.Body;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.dialect.java.PrimitiveType;
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
            BranchTarget.setBranchTarget(block.context(), targets.get(i).parent(), null, blocks.get(i+1));
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
        if (ConstantLabelSwitchChecker.isIntegralReferenceType(selector.type())) {
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
                    case JavaOp.JavaSwitchOp swOp when new ConstantLabelSwitchChecker(swOp, lookup).isCaseConstantSwitch() -> {
                        LabelsAndTargets labelsAndTargets = getLabelsAndTargets(lookup, swOp);
                        yield lowerToConstantLabelSwitchOp(block, block.transformer(), swOp, labelsAndTargets);
                    }
                    case Op.Lowerable lop -> lop.lower(block, null);
                    default -> {
                        block.op(op);
                        yield block;
                    }
                };
    }

    public static final class ConstantLabelSwitchChecker {
        private final MethodHandles.Lookup lookup;
        private JavaOp.JavaSwitchOp swOp;

        public ConstantLabelSwitchChecker(JavaOp.JavaSwitchOp swOp, MethodHandles.Lookup lookup) {
            this.swOp = swOp;
            this.lookup = lookup;
        }

        private static boolean isFinalVar(CoreOp.VarOp varOp) {
            return varOp.initOperand() != null && varOp.result().uses().stream().noneMatch(u -> u.op() instanceof CoreOp.VarAccessOp.VarStoreOp);
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

        private boolean isConstantExpr(Value v) {
            if (!(v instanceof Op.Result opr)) {
                return false;
            }
            return switch (opr.op()) {
                case CoreOp.ConstantOp cop ->
                        cop.resultType() instanceof PrimitiveType || cop.resultType().equals(J_L_STRING);
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->
                        isFinalVar(varLoadOp.varOp()) && isConstantExpr(varLoadOp.varOp().initOperand());
                case JavaOp.ConvOp convOp ->
                        (convOp.resultType() instanceof PrimitiveType || convOp.resultType().equals(J_L_STRING)) &&
                                isConstantExpr(convOp.operands().get(0));
                case JavaOp.InvokeOp invokeOp ->
                        isBoxingMethod(invokeOp.invokeDescriptor()) && isConstantExpr(invokeOp.operands().get(0));
                case JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp -> {
                    Field field;
                    try {
                        field = fieldLoadOp.fieldDescriptor().resolveToField(lookup);
                    } catch (ReflectiveOperationException e) {
                        throw new RuntimeException(e);
                    }
                    yield field.isEnumConstant() || field.accessFlags().containsAll(Set.of(AccessFlag.STATIC, AccessFlag.FINAL));
                }
                case JavaOp.UnaryOp unaryOp -> isConstantExpr(unaryOp.operands().get(0));
                case JavaOp.BinaryOp binaryOp -> binaryOp.operands().stream().allMatch(this::isConstantExpr);
                case JavaOp.BinaryTestOp binaryTestOp ->
                        binaryTestOp.operands().stream().allMatch(this::isConstantExpr);
                case JavaOp.ConditionalExpressionOp cexpr -> // bodies must yield constant expressions
                        isConstantExpr(((YieldOp) cexpr.bodies().get(0).entryBlock().terminatingOp()).yieldValue()) &&
                                isConstantExpr(((YieldOp) cexpr.bodies().get(1).entryBlock().terminatingOp()).yieldValue()) &&
                                isConstantExpr(((YieldOp) cexpr.bodies().get(2).entryBlock().terminatingOp()).yieldValue());

                case JavaOp.ConditionalAndOp cand ->
                        isConstantExpr(((YieldOp) cand.bodies().get(0).entryBlock().terminatingOp()).yieldValue()) &&
                                isConstantExpr(((YieldOp) cand.bodies().get(1).entryBlock().terminatingOp()).yieldValue());
                case JavaOp.ConditionalOrOp cor ->
                    // we can have a method isBodyYieldConstantExpr(Body)
                        isConstantExpr(((YieldOp) cor.bodies().get(0).entryBlock().terminatingOp()).yieldValue()) &&
                                isConstantExpr(((YieldOp) cor.bodies().get(1).entryBlock().terminatingOp()).yieldValue());
                default -> false;
            };
        }

        private boolean isCaseConstantLabel(Body label) {
            if (label.blocks().size() != 1 || !(label.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop) ||
                    !(yop.yieldValue() instanceof Op.Result r)) {
                return false;
            }

            // EqOp for primitives, method invocation for Strings and Reference Types
            return switch (r.op()) {
                case JavaOp.EqOp eqOp -> isConstantExpr(eqOp.operands().get(1));
                case JavaOp.InvokeOp invokeOp when !invokeOp.invokeDescriptor().equals(OBJECTS_EQUALS) -> false;
                case JavaOp.InvokeOp invokeOp -> {
                    // case null
                    if (invokeOp.operands().get(1) instanceof Op.Result opr && opr.op() instanceof CoreOp.ConstantOp cop && cop.value() == null) {
                        yield false;
                    }
                    yield  isConstantExpr(invokeOp.operands().get(1));
                }
                case JavaOp.ConditionalOrOp cor -> cor.bodies().stream().allMatch(b -> isCaseConstantLabel(b));
                default -> r.op() instanceof CoreOp.ConstantOp cop && cop.resultType().equals(BOOLEAN);
            };
        }

        public boolean isCaseConstantSwitch() {
            if (!isIntegralType(swOp.operands().get(0).type())) {
                return false;
            }
            for (int i = 0; i < swOp.bodies().size(); i+=2) {
                Body label = swOp.bodies().get(i);
                if (!isCaseConstantLabel(label)) {
                    return false;
                }
            }
            return true;
        }
    }

    record LabelsAndTargets(List<Integer> labels, List<Block> targets) {}

    static LabelsAndTargets getLabelsAndTargets(MethodHandles.Lookup lookup, JavaOp.JavaSwitchOp swOp) {
        var labels = new ArrayList<Integer>();
        var targets = new ArrayList<Block>();
        for (int i = 0; i < swOp.bodies().size() - 1; i += 2) {
            List<Integer> ls = getLabels(lookup, swOp.bodies().get(i));
            labels.addAll(ls);
            // getLabels returns list with null, for case default
            targets.addAll(Collections.nCopies(ls.size(), swOp.bodies().get(i + 1).entryBlock()));
        }
        return new LabelsAndTargets(labels, targets);
    }

    static final MethodRef OBJECTS_EQUALS = MethodRef.method(Objects.class, "equals", boolean.class, Object.class, Object.class);

    static List<Integer> getLabels(MethodHandles.Lookup lookup, Body body) {
        if (body.blocks().size() != 1 || !(body.entryBlock().terminatingOp() instanceof CoreOp.YieldOp yop) ||
                !(yop.yieldValue() instanceof Op.Result opr)) {
            throw new IllegalStateException("Body of a java switch fails the expected structure");
        }
        var labels = new ArrayList<Integer>();
        switch (opr.op()) {
            case JavaOp.EqOp eqOp -> labels.add(extractConstantLabel(lookup, body, eqOp));
            case JavaOp.InvokeOp invokeOp when invokeOp.invokeDescriptor().equals(OBJECTS_EQUALS) ->
                    labels.add(extractConstantLabel(lookup, body, invokeOp));
            case JavaOp.ConditionalOrOp cor -> {
                for (Body corbody : cor.bodies()) {
                    labels.addAll(getLabels(lookup, corbody));
                }
            }
            case CoreOp.ConstantOp constantOp ->  // default label
                    labels.add(null);
            case null, default -> throw new IllegalStateException();
        }
        return labels;
    }

    static Integer extractConstantLabel(MethodHandles.Lookup lookup, Body body, Op whenToStop) {
        Op lastOp = body.entryBlock().ops().get(body.entryBlock().ops().indexOf(whenToStop) - 1);
        CoreOp.FuncOp funcOp = CoreOp.func("f", CoreType.functionType(lastOp.result().type())).body(block -> {
            // in case we refer to constant variables in the label
            for (Value capturedValue : body.capturedValues()) {
                if (!(capturedValue instanceof Op.Result r) || !(r.op() instanceof CoreOp.VarOp vop)) {
                    continue;
                }
                Op cop = ((Op.Result) vop.initOperand()).op();
                if (cop instanceof JavaOp.ConvOp) {
                    // converted constant
                    block.op(((Op.Result)cop.operands().getFirst()).op());
                }
                block.op(cop);
                block.op(vop);
            }
            Op.Result last = null;
            for (Op op : body.entryBlock().ops()) {
                if (op.equals(whenToStop)) {
                    break;
                }
                last = block.op(op);
            }
            block.op(CoreOp.return_(last));
        });
        Object res;
        try {
            // @@@ workaround until JDK-8376974 Evaluate constant expressions
            // Exposed evaluation of constant expressions should simplify this process.
            // Then we can expose the computation of constant case values on the switch ops.
            res = BytecodeGenerator.generate(lookup, funcOp.transform(CodeTransformer.LOWERING_TRANSFORMER)).invoke();
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
        return switch (res) {
            case Byte b -> Integer.valueOf(b);
            case Short s -> Integer.valueOf(s);
            case Character c -> Integer.valueOf(c);
            case Integer i -> i;
            default -> throw new IllegalStateException(); // @@@ not going to happen
        };
    }
}
