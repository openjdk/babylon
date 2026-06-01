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
import hat.types.S16ImplOfF16;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.Set;

import static optkl.OpHelper.Invoke;

public record HATFP16Phase() implements HATPhase {

    public static final byte FIRST_OP = 0x01;
    public static final byte LAST_OP = 0x10;

    private static boolean is16BitFloat(OpHelper.Invoke invoke, Regex methodName) {
        return invoke.refIs(S16ImplOfF16.class) && invoke.nameMatchesRegex(methodName);
    }

    public static void copyVarOpWithUpdateVarTable(String functionName, CoreOp.VarOp varOp, Block.Builder blockBuilder, VarTable varTable) {
        Op.Result op = blockBuilder.add(varOp);
        varTable.addIfNeededOrThrow(functionName, op.op(), VarTable.HATOpAttribute.NARROW);
    }

    private CoreOp.FuncOp processBinaryOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, BinaryOpEnum binaryOpEnum, VarTable varTable) {
        Set<Op> reducedFloatsType = new HashSet<>();

        Invoke.stream(lookup, funcOp)
                .filter(invoke -> is16BitFloat(invoke, Regex.of(binaryOpEnum.name().toLowerCase())) && !invoke.returnsVoid())
                .forEach(invoke -> {
                    if (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) != null) {
                        if (invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp varOp) {
                            reducedFloatsType.add(varOp);
                        }
                    } else {
                        throw new RuntimeException("no reduced float type");
                    }
                });

        return Trxfmr.of(lookup, funcOp).transform(reducedFloatsType::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                copyVarOpWithUpdateVarTable(funcOp.funcName(), varOp, blockBuilder, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private CoreOp.FuncOp processInitOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> nodesToProcess = new HashSet<>();

        Invoke.stream(lookup, funcOp)
                .filter(invoke -> !invoke.returnsVoid()
                        && is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)"))
                        && invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp)
                .forEach(invoke -> {
                    if (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) != null) {
                        Op.Result first = invoke.op().result().uses().getFirst();
                        if (first.declaringElement() instanceof CoreOp.VarOp varOp) {
                            nodesToProcess.add(varOp);
                        }
                    } else {
                        throw new RuntimeException("No reduced float type");
                    }
                });

        return Trxfmr.of(lookup, funcOp).transform(nodesToProcess::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                copyVarOpWithUpdateVarTable(funcOp.funcName(), varOp, blockBuilder, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        for (BinaryOpEnum binaryOpEnum : BinaryOpEnum.values()) {
            // F16 BinarybOperations
            funcOp = processBinaryOps(lookup, funcOp, binaryOpEnum, varTable); // pending
        }
        // Init analysis before the store
        funcOp = processInitOps(lookup, funcOp, varTable);   // done
        return funcOp;
    }
}
