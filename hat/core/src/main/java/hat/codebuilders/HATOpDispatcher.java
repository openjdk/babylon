/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.codebuilders;

import hat.dialect.ArithMathOps;
import hat.dialect.HATOp;
import hat.dialect.TileOps;
import jdk.incubator.code.Op;
import optkl.codebuilders.BabylonOpDispatcher;
import optkl.codebuilders.ScopeAwareJavaOrC99StyleCodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

import static hat.dialect.HATPtrOp.HATPtrLengthOp;
import static hat.dialect.HATPtrOp.HATPtrLoadOp;
import static hat.dialect.HATPtrOp.HATPtrStoreOp;

/* this should not be too C99 specific but can reference HAT ops.  */
public interface HATOpDispatcher<T extends ScopeAwareJavaOrC99StyleCodeBuilder<T>> extends BabylonOpDispatcher<T, ScopedCodeBuilderContext> {

    T hatPtrLoadOp(HATPtrLoadOp hatPtrLoadOp);

    T hatPtrStoreOp(HATPtrStoreOp hatPtrStoreOp);

    T hatPtrLengthOp(HATPtrLengthOp hatPtrLengthOp);

    T tileConstantOp(ArithMathOps.ConstantOp constantOp);

    T tileIdOp(TileOps.TileIDOp tileIdOp);

    T tileLoadOp(TileOps.LoadOp tileLoadOp);

    T tileAddOp(ArithMathOps.AddOp tileAddOp);

    T tileStoreOp(TileOps.StoreOp tileStoreOp);

    T tileNumOp(TileOps.TileNumOp tileNumOp);

    T tileFullOp(TileOps.TileFullOp tileFullOp);

    T tileShapeOp(TileOps.TileShapeOp tileShapeOp);

    T tileSumOp(TileOps.TileSumOp tileSumOp);

    T tileIndexOp(TileOps.TileIndexOp tileIndexOp);

    T tileTransposeOp(ArithMathOps.TransposeOp tileTransposeOp);

    @Override
    default T recurse(Op op) {
        if (op instanceof HATOp hatOp) {
            switch (hatOp) {
                case HATPtrLoadOp hatPtrLoadOp -> hatPtrLoadOp(hatPtrLoadOp);
                case HATPtrStoreOp hatPtrStoreOp -> hatPtrStoreOp(hatPtrStoreOp);
                case HATPtrLengthOp hatPtrLengthOp -> hatPtrLengthOp(hatPtrLengthOp);
                default -> throw new IllegalStateException("handle nesting of hat op " + op);
            }
        } else if (op instanceof ArithMathOps.ArithMathOp arithMathOps) {
            switch (arithMathOps) {
                case ArithMathOps.ConstantOp constantOp -> tileConstantOp(constantOp);
                case ArithMathOps.AddOp addOp -> tileAddOp(addOp);
                case ArithMathOps.TransposeOp tileTransposeOp -> tileTransposeOp(tileTransposeOp);
                default -> throw new IllegalStateException("handle nesting of Tile AritmeticMath op " + op);
            }
        } else if (op instanceof TileOps.TOp tileOps) {
            switch (tileOps) {
                case TileOps.TileIDOp idOp -> tileIdOp(idOp);
                case TileOps.LoadOp loadOp -> tileLoadOp(loadOp);
                case TileOps.StoreOp storeOp -> tileStoreOp(storeOp);
                case TileOps.TileNumOp tileNumOp -> tileNumOp(tileNumOp);
                case TileOps.TileFullOp tileFullOp -> tileFullOp(tileFullOp);
                case TileOps.TileShapeOp tileShapeOp -> tileShapeOp(tileShapeOp);
                case TileOps.TileSumOp tileSumOp -> tileSumOp(tileSumOp);
                case TileOps.TileIndexOp tileIndexOp -> tileIndexOp(tileIndexOp);
                default -> throw new IllegalStateException("handle nesting of Tile TileOps.TOp " + op);
            }

        } else {
            BabylonOpDispatcher.super.recurse(op);
        }
        return (T) this;
    }
}
