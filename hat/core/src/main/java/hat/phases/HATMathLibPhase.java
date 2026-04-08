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

import hat.HATMath;
import hat.types.ReducedFloatType;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public record HATMathLibPhase() implements HATPhase {
    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, ReducedFloatType> setTypeMap = new HashMap<>();
        OpHelper.Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid() && invoke.returnsClassType() && invoke.refIs(HATMath.class))
                .forEach(invoke ->
                        // This detects a HATMathLib is stored either in a VarOp or a VarStoreOp
                        invoke.op().result().uses().stream()

                                .filter(result -> (result.op() instanceof CoreOp.VarOp) || (result.op() instanceof CoreOp.VarAccessOp.VarStoreOp))
                                .findFirst()
                                .ifPresent(result -> {
                                    // Special attention to HATTypes. This is some metadata associated with the invoke
                                    // to pass to the cogen to do further processing (e.g., build a new type or typecast).
                                    // An alternative is to insert a `stub`, or a code snippet that insert new nodes in the
                                    // IR
                                    if (ReducedFloatType.typeElementToReducedFloatTypeOrNull(invoke,(ClassType)invoke.returnType()) instanceof ReducedFloatType reducedFloatType) {
                                        setTypeMap.put(result.op(), reducedFloatType);
                                        setTypeMap.put(invoke.op(), reducedFloatType);
                                    }
                                }));
        return Trxfmr.of(lookup, funcOp).transform(setTypeMap::containsKey, (blockBuilder, op) -> {
            if (Objects.requireNonNull(op) instanceof CoreOp.VarOp varOp) {
                if (setTypeMap.get(varOp) == null) {
                    // this varOp is not a special type we insert the varOp into the new tree
                    blockBuilder.op(varOp);
                } else {
                    // Add the special type as a VarOp
                    HATFP16Phase.createF16VarOp(varOp, blockBuilder, setTypeMap.get(varOp));
                    // If we add HATMath ops for other special types in HAT (e.g., Vectors),
                    // we may need to also add the new <X>HATVarOps here as well.
                }
            } else {
                blockBuilder.op(op);
            }
            return blockBuilder;
        }).funcOp();
    }
}
