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

import optkl.IfaceValue.Vector;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.Set;

import static optkl.OpHelper.Invoke;

public final class HATVectorPhase implements HATPhase {

    private String functionName;

    private void varOpVector(Block.Builder blockBuilder, CoreOp.VarOp varOp, VarTable varTable) {
        Op.Result result = blockBuilder.add(varOp);
        varTable.addIfNeededOrThrow(functionName, result.op(), VarTable.HATOpAttribute.VECTOR);
    }

    private CoreOp.FuncOp transformVectorVar(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> vectorShapeMap = new HashSet<>();
        OpHelper.Named.Variable.stream(lookup, funcOp).forEach(variable -> {
            if (variable.firstOperandAsInvoke() instanceof Invoke invoke && invoke.returns(Vector.class)) {
                vectorShapeMap.add(variable.op());
            }
        });

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                varOpVector(blockBuilder, varOp, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        this.functionName = funcOp.funcName();
        funcOp = transformVectorVar(lookup, funcOp, varTable);
        return funcOp;
    }
}
