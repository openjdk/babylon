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

import hat.dialect.BinaryOpEnum;
import hat.dialect.HATF16Op;
import hat.types.S16ImplOfF16;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public record HATFP16Phase() implements HATPhase {
    private static boolean is16BitFloat(OpHelper.Invoke invoke, Regex methodName) {
        return invoke.refIs(S16ImplOfF16.class) && invoke.nameMatchesRegex(methodName);
    }


    // recursive
    private static String findVarNameOrNull(Value v) {
        return  (v instanceof Op.Result r) ? switch (r.op()){
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp-> findVarNameOrNull(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> hatf16VarOp.varName();
            default -> null;
        }:null;
    }

    // recursive
    private static String findVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVarNameOrNull(varLoadOp.operands().getFirst());
    }

    //recursive
    private static boolean isF16Local(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isF16Local(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> true;
            default -> false;
        };
    }

    //recursive
    private static boolean isF16Local(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isF16Local(varLoadOp.operands().getFirst());
    }



    public static void createF16VarOp(CoreOp.VarOp varOp, Block.Builder blockBuilder, Class<?> reducedFloatType) {
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

    private void createF16ConvOP(Invoke invoke, Block.Builder blockBuilder, Class<?> reducedFloatType) {
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
                                        findVarNameOrNull(varLoadOp),
                                        varLoadOp.varType(),
                                        blockBuilder.context().getValues(varLoadOp.operands()))
                        )
                )
        );
    }

    private void createFloatFromF16(Invoke invoke, Block.Builder blockBuilder, Class<?> reducedFloatType) {
        blockBuilder.context().mapValue(invoke.op().result(),
                blockBuilder.op(copyLocation(invoke.op(), new HATF16Op.HATF16ToFloatConvOp(
                                JavaType.FLOAT,
                                reducedFloatType,
                                isF16Local(invoke.op().operands().getFirst()),
                                invoke.opFromFirstOperandOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                                        && varLoadOp.resultType().equals(JavaType.FLOAT),
                                blockBuilder.context().getValues(invoke.op().operands()))
                        )
                )
        );
    }

    private void createF16BinaryOp(JavaOp.InvokeOp invokeOp, Block.Builder blockBuilder, BinaryOpEnum binaryOpEnum, Class<?> reducedFloatType) {
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

    private CoreOp.FuncOp dialectifyF16Ops(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp, BinaryOpEnum binaryOpEnum) {
        Map<Op, Class<? extends S16ImplOfF16>> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup, funcOp)
                .filter(invoke -> is16BitFloat(invoke, Regex.of(binaryOpEnum.name().toLowerCase())) && !invoke.returnsVoid())
                .forEach(invoke -> {
                    if (S16ImplOfF16.typeElementToFloatClassOrNull(invoke,(ClassType)invoke.refType()) instanceof Class<? extends S16ImplOfF16> category) {
                        reducedFloatsType.put(invoke.op(), category);
                        if (invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp varOp) {
                            reducedFloatsType.put(varOp, category);
                        }
                    }else{
                        throw new RuntimeException("no reduced float type");
                    }
                });

        return Trxfmr.of(lookup, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16BinaryOp(invokeOp, blockBuilder, binaryOpEnum, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16Stores(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
        Invoke.stream(lookup, funcOp)
                .filter(invoke -> is16BitFloat(invoke, Regex.of("value"))
                        && invoke.returns16BitValue())
                .forEach(invoke -> {
                    if (invoke.opFromFirstOperandOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                            && varLoadOp.operands().getFirst() instanceof Op.Result firstOperandsOpResult
                            && firstOperandsOpResult.op() instanceof HATF16Op.HATF16VarOp) {
                        nodesInvolved.addAll(Set.of(invoke.op(), varLoadOp));
                    }
                });

        return Trxfmr.of(lookup, funcOp).transform(ce -> nodesInvolved.contains(ce), (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.context().getValue(invokeOp.operands().getFirst()));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                createF16VarLoadOp(varLoadOp, blockBuilder);
            }
            return blockBuilder;
        }).funcOp();
    }


    private CoreOp.FuncOp dialectifyF16Init(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Class<? extends S16ImplOfF16>> reducedFloatsType = new HashMap<>();

        Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid()
                        && is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)"))
                        && invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp
                )
                .forEach(invoke -> {
                    if ( S16ImplOfF16.typeElementToFloatClassOrNull(invoke,(ClassType)invoke.refType())instanceof Class<? extends S16ImplOfF16> reducedFloatType) {
                        reducedFloatsType.put(invoke.opFromOnlyUseOrNull(), reducedFloatType);
                        reducedFloatsType.put(invoke.op(), reducedFloatType);
                    }else {
                        throw new RuntimeException("No reduced float type");
                    }
                });

        return Trxfmr.of(lookup, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                createF16ConvOP(invoke(lookup, invokeOp), blockBuilder, reducedFloatsType.get(invokeOp));
            } else if (op instanceof CoreOp.VarOp varOp) {
                createF16VarOp(varOp, blockBuilder, reducedFloatsType.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyF16ToFloat(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<JavaOp.InvokeOp, Class<? extends S16ImplOfF16>> reducedFloatsType = new HashMap<>();
        funcOp.elements()
                .filter(ce -> ce instanceof JavaOp.InvokeOp)
                .map(ce -> invoke(lookup, ce))
                .filter(invoke -> invoke instanceof Invoke.Static)
                .map(invoke -> (Invoke.Static) invoke)
                .filter(invoke -> invoke.nameMatchesRegex("(f16ToFloat|bfloat162float)") && invoke.returnsFloat())
                .forEach(invoke -> {
                            if (S16ImplOfF16.typeElementToFloatClassOrNull(invoke,(ClassType)invoke.refType()) instanceof Class<? extends S16ImplOfF16>  reducedFloatType) {
                                reducedFloatsType.put(invoke.op(), reducedFloatType);
                            }else{
                                throw new RuntimeException("No reduced float type");
                            }
                        }
                );


        return Trxfmr.of(lookup, funcOp).transform(reducedFloatsType::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp && invoke(lookup, invokeOp) instanceof Invoke invoke) {
                createFloatFromF16(invoke, blockBuilder, reducedFloatsType.get(invoke.op()));
            }
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        for (BinaryOpEnum binaryOpEnum : BinaryOpEnum.values()) {
            // F16 Operations
            funcOp = dialectifyF16Ops(lookup,funcOp, binaryOpEnum);
        }
        // Init analysis before the store
        funcOp = dialectifyF16Init(lookup,funcOp);
        funcOp = dialectifyF16ToFloat(lookup,funcOp);
        // Store analysis
        funcOp = dialectifyF16Stores(lookup,funcOp);
        return funcOp;
    }
}
