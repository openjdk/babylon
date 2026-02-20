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
package hat.phases;

import hat.callgraph.KernelCallGraph;
import hat.dialect.BinaryOpEnum;
import hat.dialect.HATF16Op;
import hat.types.BF16;
import hat.types.F16;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.util.Regex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Gatherer;
import java.util.stream.Stream;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public record HATFP16Phase(KernelCallGraph kernelCallGraph) implements HATPhase {

    public static ReducedFloatType categorizeReducedFloat(Invoke invoke) {
        if (invoke.refIs(F16.class)) {
            return ReducedFloatType.HalfFloat.of();
        } else if (invoke.refIs(BF16.class)) {
            return ReducedFloatType.BFloat16.of();
        }
        return null;
    }

    public static ReducedFloatType categorizeReducedFloatFromResult(Invoke invoke) {
        if (invoke.resultTypeIs(F16.class)) {
            return ReducedFloatType.HalfFloat.of();
        } else if (invoke.resultTypeIs(BF16.class)) {
            return ReducedFloatType.BFloat16.of();
        }
        return null;
    }

    public static void createF16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        blockBuilder.context().mapValue(varOp.result(),
                blockBuilder.op(copyLocation(varOp,
                                new HATF16Op.HATF16VarOp(
                                        varOp.varName(),
                                        reducedFloatType, varOp.resultType(),
                                        blockBuilder.context().getValues(varOp.operands()))
                        )
                )
        );
    }

    private void createF16ConvOP(Invoke invoke, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        blockBuilder.context().mapValue(invoke.op().result(),
                blockBuilder.op(copyLocation(invoke.op(), new HATF16Op.HATF16ConvOp(
                                JavaType.VOID,
                                reducedFloatType,
                                blockBuilder.context().getValues(invoke.op().operands()))
                        )
                )
        );
    }

    private void createF16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp, Block.Builder blockBuilder) {
        blockBuilder.context().mapValue(varLoadOp.result(),
                blockBuilder.op(copyLocation(varLoadOp,
                                new HATF16Op.HATF16VarLoadOp(
                                        HATPhaseUtils.findVarNameOrNull(varLoadOp),
                                        varLoadOp.varType(),
                                        blockBuilder.context().getValues(varLoadOp.operands()))
                        )
                )
        );
    }

    private void createFloatFromF16(Invoke invoke, Block.Builder blockBuilder, ReducedFloatType reducedFloatType) {
        blockBuilder.context().mapValue(invoke.op().result(),
                blockBuilder.op(copyLocation(invoke.op(), new HATF16Op.HATF16ToFloatConvOp(
                                JavaType.FLOAT,
                                reducedFloatType,
                                HATPhaseUtils.isF16Local(invoke.op().operands().getFirst()),
                                invoke.opFromFirstOperandOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                                        && varLoadOp.resultType().equals(JavaType.FLOAT),
                                blockBuilder.context().getValues(invoke.op().operands()))
                        )
                )
        );
    }

    private void createF16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, BinaryOpEnum binaryOpEnum, ReducedFloatType reducedFloatType) {
        List<Value> operands = invokeOp.operands();
        TypeElement typeElement = invokeOp.resultType();
        List<Value> outputOperands = blockBuilder.context().getValues(operands);
        HATF16Op.HATF16BinaryOp binaryOp = switch (binaryOpEnum) {
            case ADD -> new HATF16Op.HATF16BinaryOp.HATF16AddOp(typeElement, reducedFloatType, outputOperands);
            case SUB -> new HATF16Op.HATF16BinaryOp.HATF16SubOp(typeElement, reducedFloatType, outputOperands);
            case MUL -> new HATF16Op.HATF16BinaryOp.HATF16MulOp(typeElement, reducedFloatType, outputOperands);
            case DIV -> new HATF16Op.HATF16BinaryOp.HATF16DivOp(typeElement, reducedFloatType, outputOperands);
        };
        blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp, binaryOp)));
    }

    private CoreOp.FuncOp dialectifyF16Ops(CoreOp.FuncOp funcOp, BinaryOpEnum binaryOpEnum) {
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup(), funcOp)
                .filter(invoke -> HATPhaseUtils.is16BitFloat(invoke, Regex.of(binaryOpEnum.name().toLowerCase())) && !invoke.returnsVoid())
                .forEach(invoke -> {
                    ReducedFloatType category = categorizeReducedFloat(invoke);
                    reducedFloatsType.put(invoke.op(), category);
                    if (invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp varOp) {
                        reducedFloatsType.put(varOp, category);
                    }
                });

        return Trxfmr.of(this, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16BinaryOp(invokeOp, blockBuilder, binaryOpEnum, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16Stores(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
        Invoke.stream(lookup(), funcOp)
                .filter(invoke -> HATPhaseUtils.is16BitFloat(invoke, Regex.of("value"))
                        && invoke.returns16BitValue())
                .forEach(invoke -> {
                    if (invoke.opFromFirstOperandOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                            && varLoadOp.operands().getFirst() instanceof Op.Result firstOperandsOpResult
                            && firstOperandsOpResult.op() instanceof HATF16Op.HATF16VarOp) {
                        nodesInvolved.addAll(Set.of(invoke.op(), varLoadOp));
                    }
                });

        return Trxfmr.of(this, funcOp).transform(ce -> nodesInvolved.contains(ce), (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.context().getValue(invokeOp.operands().getFirst()));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                createF16VarLoadOp(varLoadOp, blockBuilder);
            }
            return blockBuilder;
        }).funcOp();
    }


    private CoreOp.FuncOp dialectifyF16Init(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup(), funcOp)
                .filter(invoke -> !invoke.returnsVoid()
                        && HATPhaseUtils.is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)"))
                        && invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp
                )
                .forEach(invoke -> {
                    ReducedFloatType reducedFloatType = categorizeReducedFloat(invoke);
                    reducedFloatsType.put(invoke.opFromOnlyUseOrNull(), reducedFloatType);
                    reducedFloatsType.put(invoke.op(), reducedFloatType);
                });

        return Trxfmr.of(this, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16ConvOP(invoke(lookup(), invokeOp), blockBuilder, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16ToFloat(CoreOp.FuncOp funcOp) {
        Map<JavaOp.InvokeOp, ReducedFloatType> reducedFloatsType = new HashMap<>();
        funcOp.elements()
                .filter(ce -> ce instanceof JavaOp.InvokeOp)
                .map(ce -> invoke(lookup(), ce))
                .filter(invoke -> invoke instanceof Invoke.Static)
                .map(invoke -> (Invoke.Static) invoke)
                .filter(invoke ->
                        invoke.nameMatchesRegex("(f16ToFloat|bfloat162float)")
                                && invoke.returnsFloat())
                .forEach(invoke ->
                        reducedFloatsType.put(invoke.op(), categorizeReducedFloat(invoke))
                );


        return Trxfmr.of(this, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp && invoke(lookup(), invokeOp) instanceof Invoke invoke) {
                createFloatFromF16(invoke, blockBuilder, reducedFloatsType.get(invoke.op()));
            }
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        for (BinaryOpEnum binaryOpEnum : BinaryOpEnum.values()) {
            // F16 Operations
            funcOp = dialectifyF16Ops(funcOp, binaryOpEnum);
        }
        // Init analysis before the store
        funcOp = dialectifyF16Init(funcOp);
        funcOp = dialectifyF16ToFloat(funcOp);
        // Store analysis
        funcOp = dialectifyF16Stores(funcOp);
        return funcOp;
    }
}
