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
package hat.phases;

import hat.callgraph.KernelCallGraph;
import hat.dialect.HATMathLibOp;
import hat.dialect.ReducedFloatType;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public record HATMathLibPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    private void transformHATMathWithVarOp(Block.Builder blockBuilder, Op op, Map<Op, ReducedFloatType> setTypeMap) {
        switch (op) {
            case JavaOp.InvokeOp invokeOp -> {
                // Invoke Op is replaced with a HATMathLibOp
                List<Value> operands = blockBuilder.context().getValues(invokeOp.operands());

                // For each operand, obtain if it is a reference from global memory or device memory:
                List<Boolean> referenceList = IntStream.range(0, operands.size())
                        .mapToObj(i -> HATPhaseUtils.isArrayReference(lookup(), invokeOp.operands().get(i)))
                        .collect(Collectors.toList());

                HATMathLibOp hatMathLibOp = new HATMathLibOp(
                        invokeOp.resultType(),
                        invokeOp.invokeDescriptor().name(),  // intrinsic name
                        setTypeMap.get(invokeOp),
                        referenceList,
                        operands);

                Op.Result hatMathLibOpResult = blockBuilder.op(hatMathLibOp);
                blockBuilder.context().mapValue(invokeOp.result(), hatMathLibOpResult);
            }
            case CoreOp.VarOp varOp -> {
                if (setTypeMap.get(varOp) == null) {
                    // this means that the varOp is not a special type
                    // then we insert the varOp into the new tree
                    blockBuilder.op(varOp);
                } else {
                    // Add the special type as a VarOp
                    HATFP16Phase.createF16VarOp(varOp, blockBuilder, setTypeMap.get(varOp));
                }
            }
            default -> blockBuilder.op(op);
        }
    }

    private CoreOp.FuncOp transformHATMathWithVarOp(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> setTypeMap = new HashMap<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(invoke -> !invoke.returnsVoid() && HATPhaseUtils.isMathLib(invoke))
                .forEach(invoke ->
                        invoke.op().result().uses().stream()
                                .filter(result -> (result.op() instanceof CoreOp.VarOp) || (result.op() instanceof CoreOp.VarAccessOp.VarStoreOp))
                                .findFirst()
                                .ifPresent(result -> {
                                    ReducedFloatType reducedFloatType =  HATFP16Phase.categorizeReducedFloatFromResult(invoke.op());
                                    setTypeMap.put(result.op(), reducedFloatType);
                                    setTypeMap.put(invoke.op(), reducedFloatType);
                                }));

        return Trxfmr.of(this, funcOp).transform(setTypeMap::containsKey, (blockBuilder, op) -> {
            transformHATMathWithVarOp(blockBuilder, op, setTypeMap);
            return blockBuilder;
        }).funcOp();
    }

    // At this point, we already have the main HATMathOp in place, which stores the operation either in a
    // VarOp or VarStoreOp. This phase is used to detect when a HATMathOp uses as operands (VarLoadOp)
    // variables that come from other HATMathOp. In this way, we will allow composition of HATMath functions.
    private CoreOp.FuncOp transformConcatenationHATMathOps(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> setTypeMap = new HashMap<>();
        funcOp.elements()
                .filter(HATMathLibOp.class::isInstance)
                .forEach( opElement -> {
                    Op.Result result = ((HATMathLibOp) opElement).result();
                    HATMathLibOp hatMathLibOp = (HATMathLibOp) result.op();
                    List<Value> operands = hatMathLibOp.operands();
                    // for each of the operands, check where they are coming from
                    for (Value value: operands) {
                        OpHelper.Invoke.stream(lookup(), value.result().op())
                                .filter(invoke -> !invoke.returnsVoid() && HATPhaseUtils.isMathLib(invoke))
                                .forEach(invoke -> {
                                    ReducedFloatType reducedFloatType =  HATFP16Phase.categorizeReducedFloatFromResult(invoke.op());
                                    setTypeMap.put(invoke.op(), reducedFloatType);
                                });
                    }
                });

        return Trxfmr.of(this, funcOp).transform(setTypeMap::containsKey, (blockBuilder, op) -> {
            transformHATMathWithVarOp(blockBuilder, op, setTypeMap);
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        funcOp = transformHATMathWithVarOp(funcOp);
        funcOp = transformConcatenationHATMathOps(funcOp);
        return funcOp;
    }
}
