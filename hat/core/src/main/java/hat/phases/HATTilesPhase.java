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

import hat.buffer.TensorF32;
import hat.codetypes.PtrType;
import hat.dialect.ArithMathOps;
import hat.dialect.TileOps;
import hat.types.Tile;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public record HATTilesPhase() implements HATPhase {

    private CoreOp.FuncOp appendAlignment(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        Set<Op> opsToProcess = new HashSet<>();

        // Also check the tile dialect was introduced
        final boolean isTileUsed = funcOp.elements().anyMatch(element -> element instanceof TileOps.TOp || element instanceof ArithMathOps.ArithMathOp);
        if (isTileUsed) {
            // We need to transform the code tree to insert a Invoke for the alignment associated with a new var
            // to carry for all loads/store operations after the alignment.
            List<Block.Parameter> parameters = funcOp.body().entryBlock().parameters();
            Op firstOp = funcOp.bodies().getFirst().blocks().getFirst().firstOp();
            // analyze each parameter
            List<CoreOp.VarOp> tileArgs = new ArrayList<>();
            for (Block.Parameter p : parameters) {
                Op.Result paramUsage = p.uses().getFirst();
                if (paramUsage.declaringElement() instanceof CoreOp.VarOp varOp) {
                    var s = varOp.resultType().valueType();
                    if (s instanceof PtrType || s.toString().equals(TensorF32.class.getCanonicalName())) {
                        tileArgs.add(varOp);
                        opsToProcess.add(varOp);
                    }
                    firstOp = varOp;
                }
            }

            if (firstOp != null && tileArgs.contains(firstOp)) {
                // If this is the case, we need to move firstOp to next Op
                // Otherwise, the subsequence transform phase will not expand
                // with the new Ops for performing the alignment.
                boolean wasFound = false;
                for (CodeElement<?, ?> codeElement : funcOp.elements().sequential().toList()) {
                    if (wasFound) {
                        firstOp = (Op) codeElement;
                        break;
                    }
                    if (codeElement == firstOp) {
                        wasFound = true;
                    }
                }
            }

            Map<Op, Op.Result> paramMap = new HashMap<>();
            final Op finalFirstOp = firstOp;
            Map<Op, Op.Result> useVarOps = new HashMap<>();
            CoreOp.FuncOp finalFuncOp = funcOp;
            funcOp = funcOp.transform((builder, op) -> {
                if (opsToProcess.contains(op) && op instanceof CoreOp.VarOp) {
                    Op.Result newVarOpResult = builder.add(op);
                    paramMap.put(op, newVarOpResult);
                } else if (op == finalFirstOp) {
                    // if the current op is the same object as the firstOp, then we expand this
                    // op with more ops to accommodate the alignment.

                    // Add the current op
                    builder.add(op);

                    // place new invoke ops here: we need to expand the alignment for all parameters that read/write to global memory

                    CoreOp.ConstantOp constantOp = CoreOp.constant(JavaType.INT, 16);  // Alignment is always to 16 bytes.
                    Op.Result constantValue = builder.add(constantOp);

                    // For each parameter, we add a varLoadOp with the varOp to align, an InvokeOp with the alignment, and a VarOp with the result
                    // to be propagated for the rest of the code tree
                    for (CoreOp.VarOp varTile : tileArgs) {
                        // Insert a varLoadOp for the tile variable
                        Op.Result varLoadOp =  builder.add(CoreOp.varLoad(paramMap.get(varTile)));
                        // Insert a new invoke with the alignment
                        JavaOp.InvokeOp invoke = JavaOp.invoke(TILE_ARRAY_ALIGN, List.of(varLoadOp, constantValue));
                        Op.Result invokeResult = builder.add(invoke);
                        // Insert the new varOp
                        CoreOp.VarOp varOp = CoreOp.var(varTile.varName().concat("_"), invokeResult);
                        Op.Result varOpResult = builder.add(varOp);
                        for (Op.Result u : varTile.result().uses()) {
                            useVarOps.put(u.op(), varOpResult);
                            opsToProcess.add(u.op());
                        }
                        varTable.addIfNeededOrThrow(finalFuncOp.funcName(), varOp, VarTable.HATOpAttribute.TILE);
                    }
                } else if (opsToProcess.contains(op) && op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    // For the rest of the varLoads that loads a Read/Write buffer, we replace it with the new VarOp created during the
                    // op expansion
                    CoreOp.VarAccessOp.VarLoadOp v = CoreOp.varLoad(useVarOps.get(varLoadOp));
                    Op.Result newVarLoad = builder.add(v);
                    builder.context().mapValue(varLoadOp.result(), newVarLoad);
                } else {
                    builder.add(op);
                }
                return builder;
            });
        }
        return funcOp;
    }

    private static final MethodRef TILE_ARRAY_ALIGN = MethodRef.method(TileAlign.class, "align", Tile.class, Object.class, int.class);
    public static class TileAlign {
        public static Tile align(Object inputRef, final int alignment) {
            return null;
        }
    }

    private CoreOp.FuncOp classifyTileVarOp(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        // process Tile-Vars to insert into the VarTable
        // Load operation returns a new Tile (view of the input data in a tile)
        Set<Op> opsToProcess = new HashSet<>();

        // Process nodes after Tile dialect
        funcOp.elements().forEach(element -> {
            switch (element) {
                // Two types of Tile Ops: a) TileContextOp for context operations, and arithmetic ops
                case TileOps.TileContextOp modelOp when modelOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOo ->
                        opsToProcess.add(varOo);
                case ArithMathOps.ArithMathOp arithMathOp when arithMathOp.result().uses().getFirst().declaringElement() instanceof CoreOp.VarOp varOp ->
                        opsToProcess.add(varOp);
                case null, default -> {
                }
            }
        });

        // We have identified the invoke and the varOp associated with it
        return Trxfmr.of(lookup, funcOp).transform(opsToProcess::contains, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                Op.Result opResult = blockBuilder.add(varOp);
                varTable.addIfNeededOrThrow(funcOp.funcName(), opResult.op(), VarTable.HATOpAttribute.TILE);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        List<ActionTransformer> transformers = List.of(
                this::appendAlignment,
                this::classifyTileVarOp
        );
        CoreOp.FuncOp[] f = new CoreOp.FuncOp[]{funcOp};
        transformers.forEach(action -> f[0] = action.apply(lookup, f[0], varTable));
        return f[0];
    }

}
