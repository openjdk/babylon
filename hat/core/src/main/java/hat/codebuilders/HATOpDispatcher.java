/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import hat.dialect.*;
import jdk.incubator.code.Op;
import optkl.codebuilders.BabylonOpDispatcher;
import optkl.codebuilders.JavaOrC99StyleCodeBuilder;
import optkl.codebuilders.ScopeAwareJavaOrC99StyleCodeBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

/* this should not be too C99 specific but can reference HAT ops.  */
public interface HATOpDispatcher<T extends ScopeAwareJavaOrC99StyleCodeBuilder<T>> extends BabylonOpDispatcher<T, ScopedCodeBuilderContext> {

    T hatBarrierOp( HATBarrierOp barrierOp);

    T hatLocalVarOp( HATMemoryVarOp.HATLocalVarOp barrierOp);

    T hatPrivateVarOp( HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp);

    T hatThreadIdOp( HATThreadOp hatThreadOp);

    T hatVectorVarOp( HATVectorOp.HATVectorVarOp hatVectorVarOp);

    T hatVectorStoreOp( HATVectorOp.HATVectorStoreView hatFloat4StoreOp);

    T hatBinaryVectorOp( HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp);

    T hatVectorLoadOp( HATVectorOp.HATVectorLoadOp hatVectorLoadOp);

    T hatSelectLoadOp( HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp);

    T hatSelectStoreOp( HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp);

    T hatVectorVarLoadOp( HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp);

    T hatF16VarOp( HATF16Op.HATF16VarOp hatF16VarOp);

    T hatF16BinaryOp( HATF16Op.HATF16BinaryOp hatF16BinaryOp);

    T hatF16VarLoadOp( HATF16Op.HATF16VarLoadOp hatF16VarLoadOp);

    T hatF16ConvOp( HATF16Op.HATF16ConvOp hatF16ConvOp);

    T hatVectorOfOps( HATVectorOp.HATVectorOfOp hatVectorOp);

    T hatVectorMakeOf( HATVectorOp.HATVectorMakeOfOp hatVectorMakeOfOp);

    T hatF16ToFloatConvOp( HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp);

    T hatPrivateVarInitOp( HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp);

    T hatMemoryLoadOp( HATMemoryDefOp.HATMemoryLoadOp hatMemoryLoadOp);

    T hatPtrLoadOp(HATPtrOp.HATPtrLoadOp hatPtrLoadOp);

    T hatPtrStoreOp( HATPtrOp.HATPtrStoreOp hatPtrStoreOp);

    T hatPtrLengthOp( HATPtrOp.HATPtrLengthOp hatPtrLengthOp);


    @Override
    default T recurse(Op op) {
        if (op instanceof HATOp hatOp) {
            switch (hatOp) {
                case HATBarrierOp $ -> hatBarrierOp($);
                case HATMemoryVarOp.HATLocalVarOp $ -> hatLocalVarOp($);
                case HATMemoryVarOp.HATPrivateVarOp $ -> hatPrivateVarOp($);
                case HATMemoryVarOp.HATPrivateInitVarOp $ -> hatPrivateVarInitOp($);
                case HATThreadOp $ -> hatThreadIdOp($);
                case HATVectorOp.HATVectorVarOp $ -> hatVectorVarOp($);
                case HATVectorOp.HATVectorStoreView $ -> hatVectorStoreOp($);
                case HATVectorOp.HATVectorBinaryOp $ -> hatBinaryVectorOp($);
                case HATVectorOp.HATVectorLoadOp $ -> hatVectorLoadOp($);
                case HATVectorOp.HATVectorSelectLoadOp $ -> hatSelectLoadOp($);
                case HATVectorOp.HATVectorSelectStoreOp $ -> hatSelectStoreOp($);
                case HATVectorOp.HATVectorVarLoadOp $ -> hatVectorVarLoadOp($);
                case HATVectorOp.HATVectorOfOp $ -> hatVectorOfOps($);
                case HATF16Op.HATF16VarOp $ -> hatF16VarOp($);
                case HATF16Op.HATF16BinaryOp $ -> hatF16BinaryOp($);
                case HATF16Op.HATF16VarLoadOp $ -> hatF16VarLoadOp($);
                case HATF16Op.HATF16ConvOp $ -> hatF16ConvOp($);
                case HATVectorOp.HATVectorMakeOfOp $ -> hatVectorMakeOf($);
                case HATPtrOp.HATPtrLoadOp $ -> hatPtrLoadOp($);
                case HATPtrOp.HATPtrStoreOp $ -> hatPtrStoreOp($);
                case HATPtrOp.HATPtrLengthOp $ -> hatPtrLengthOp($);
                case HATF16Op.HATF16ToFloatConvOp $ -> hatF16ToFloatConvOp($);
                case HATMemoryDefOp.HATMemoryLoadOp $ -> hatMemoryLoadOp($);
                default -> throw new IllegalStateException("handle nesting of hat op " + op);
            }
        } else {
            BabylonOpDispatcher.super.recurse(op);
        }

        return (T) this;
    }
}
