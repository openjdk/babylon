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

import hat.TileContext;
import hat.dialect.HATTileOp;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public record HATTileIDsPhase() implements HATPhase {

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<CodeElement<?, ?>> opsToRemove = new HashSet<>();
        return Trxfmr.of(lookup, funcOp).transform(c -> {
                    if (c.op() instanceof JavaOp.InvokeOp invokeOp && (OpHelper.Invoke.invoke(lookup, invokeOp)).refIs(TileContext.class) && invokeOp.invokeReference().name().equals("bid")) {
                        // Add HAT Tile Op
                        List<Value> operands = invokeOp.operands();
                        if (operands.size() != 1) {
                            throw new IllegalStateException("[Error] Expected one argument for the TileContext#bid method");
                        }

                        // We need to remove the varLoad in arg0
                        Value varLoadValue = operands.getFirst();
                        opsToRemove.add(varLoadValue.asResult().op());


                        Value dimensionValue = operands.getLast();
                        int dimension = HATPhaseUtils.findValueIntExpression(dimensionValue);

                        // add constant to be removed
                        opsToRemove.add(dimensionValue.asResult().op());

                        // Replace with the TileID Op
                        c.replace(HATTileOp.create(invokeOp.invokeReference().name(), dimension));
                    }
                }, varTable)
                .remap(opsToRemove)
                .remove(opsToRemove::contains, varTable)
                .funcOp();
    }
}
