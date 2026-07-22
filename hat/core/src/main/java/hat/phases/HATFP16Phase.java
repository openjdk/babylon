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

import hat.types.S16ImplOfF16;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.Set;

import static optkl.OpHelper.Invoke;

public record HATFP16Phase() implements HATPhase {

    public static final byte FIRST_OP = 0x01;
    public static final byte LAST_OP = 0x10;

    public static void copyVarOpWithUpdateVarTable(String functionName, CoreOp.VarOp varOp, Block.Builder blockBuilder, VarTable varTable) {
        Op.Result op = blockBuilder.add(varOp);
        varTable.addIfNeededOrThrow(functionName, op.op(), VarTable.HATOpAttribute.NARROW);
    }

    private CoreOp.FuncOp processBinaryOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> reducedFloatsType = new HashSet<>();
        Invoke.stream(lookup, funcOp)
                .filter(invoke -> invoke.refIs(S16ImplOfF16.class)
                        && (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) != null)
                        && !invoke.returnsVoid() && !invoke.returns(float.class))
                .forEach(invoke -> {
                        if (invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp varOp) {
                            reducedFloatsType.add(varOp);
                        }
                });

        return Trxfmr.of(lookup, funcOp).transform(reducedFloatsType::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                copyVarOpWithUpdateVarTable(funcOp.funcName(), varOp, blockBuilder, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        return processBinaryOps(lookup, funcOp, varTable);
    }
}
