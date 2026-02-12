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

public record HATMathLibPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    private void transformHATMathWithVarOp(Block.Builder blockBuilder, Op op, Map<Op, ReducedFloatType> setTypeMap) {
        switch (op) {
            case JavaOp.InvokeOp invokeOp -> {
                // Invoke Op is replaced with a HATMathLibOp
                List<Value> operands = blockBuilder.context().getValues(invokeOp.operands());

                HATMathLibOp hatMathLibOp = new HATMathLibOp(
                        invokeOp.resultType(),               // result type from the math operation
                        invokeOp.invokeDescriptor().name(),  // intrinsic name
                        operands);                           // list of operands for the new Op

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
                    // If we add HATMath ops for other special types in HAT (e.g., Vectors),
                    // we may need to also add the new <X>HATVarOps here as well.
                }
            }
            default -> blockBuilder.op(op);
        }
    }

    private Map<Op, ReducedFloatType> analyseIRForInvokeHATMath(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> setTypeMap = new HashMap<>();
        OpHelper.Invoke.stream(lookup(), funcOp)
                .filter(invoke -> !invoke.returnsVoid() && HATPhaseUtils.isMathLib(invoke))
                .map(invoke -> {
                    // Inspection of all MathLib invokes.
                    // If operands depend on other Invoke of the same library, then we add them
                    // in the hash map to transform these invoke nodes with the corresponding HATMathLibOp.
                    List<Value> operands = invoke.op().operands();

                    // for each of the operands, check whether the operand is of type invoke with a HATMath operation.
                    // This detects composition of math operations.
                    operands.forEach((Value value) -> OpHelper.Invoke.stream(lookup(), value.result().op())
                            .filter(invokeInner -> !invokeInner.returnsVoid() && HATPhaseUtils.isMathLib(invokeInner))
                            .forEach(invokeInner -> {
                                ReducedFloatType reducedFloatType = HATFP16Phase.categorizeReducedFloatFromResult(invokeInner);
                                setTypeMap.put(invokeInner.op(), reducedFloatType);
                            }));
                    return invoke;
                }).forEach(invoke ->
                        // This detects a HATMathLib is stored either in a VarOp or a VarStoreOp
                        invoke.op().result().uses().stream()
                                .filter(result -> (result.op() instanceof CoreOp.VarOp) || (result.op() instanceof CoreOp.VarAccessOp.VarStoreOp))
                                .findFirst()
                                .ifPresent(result -> {
                                    // Special attention to HATTypes. This is some metadata associated with the invoke
                                    // to pass to the cogen to do further processing (e.g., build a new type or typecast).
                                    // An alternative is to insert a `stub`, or a code snippet that insert new nodes in the
                                    // IR
                                    ReducedFloatType reducedFloatType =  HATFP16Phase.categorizeReducedFloatFromResult(invoke);
                                    setTypeMap.put(result.op(), reducedFloatType);
                                    setTypeMap.put(invoke.op(), reducedFloatType);
                                }));
        return setTypeMap;
    }

    private CoreOp.FuncOp transformHATMathWithVarOp(CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> setTypeMap = analyseIRForInvokeHATMath(funcOp);
        return Trxfmr.of(this, funcOp).transform(setTypeMap::containsKey, (blockBuilder, op) -> {
            transformHATMathWithVarOp(blockBuilder, op, setTypeMap);
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        return transformHATMathWithVarOp(funcOp);
    }
}
