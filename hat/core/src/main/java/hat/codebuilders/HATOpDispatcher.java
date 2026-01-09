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
import optkl.codebuilders.ScopedCodeBuilderContext;

/* this should not be too C99 specific but can reference HAT ops.  */
public interface HATOpDispatcher<T extends JavaOrC99StyleCodeBuilder<T>> extends BabylonOpDispatcher<T, ScopedCodeBuilderContext> {

    T hatBarrierOp(ScopedCodeBuilderContext buildContext, HATBarrierOp barrierOp);

    T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATLocalVarOp barrierOp);

    T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp);

    T hatThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp hatThreadOp);

    T hatVectorVarOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorVarOp hatVectorVarOp);

    T hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorStoreView hatFloat4StoreOp);

    T hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp);

    T hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorLoadOp hatVectorLoadOp);

    T hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorSelectLoadOp hatVSelectLoadOp);

    T hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorSelectStoreOp hatVSelectStoreOp);

    T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp);

    T hatF16VarOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16VarOp hatF16VarOp);

    T hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16BinaryOp hatF16BinaryOp);

    T hatF16VarLoadOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16VarLoadOp hatF16VarLoadOp);

    T hatF16ConvOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16ConvOp hatF16ConvOp);

    T hatVectorOfOps(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorOfOp hatVectorOp);

    T hatVectorMakeOf(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorMakeOfOp hatVectorMakeOfOp);

    T hatF16ToFloatConvOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16ToFloatConvOp hatF16ToFloatConvOp);

    T hatPrivateVarInitOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp);

    T hatMemoryLoadOp(ScopedCodeBuilderContext buildContext, HATMemoryDefOp.HATMemoryLoadOp hatMemoryLoadOp);

    T hatPtrLoadOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrLoadOp hatPtrLoadOp);

    T hatPtrStoreOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrStoreOp hatPtrStoreOp);

    T hatPtrLengthOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrLengthOp hatPtrLengthOp);

    @Override
    default T recurse(ScopedCodeBuilderContext buildContext, Op op) {
        if (op instanceof HATOp hatOp) {
            switch (hatOp) {
                case HATBarrierOp $ -> hatBarrierOp(buildContext, $);
                case HATMemoryVarOp.HATLocalVarOp $ -> hatLocalVarOp(buildContext, $);
                case HATMemoryVarOp.HATPrivateVarOp $ -> hatPrivateVarOp(buildContext, $);
                case HATMemoryVarOp.HATPrivateInitVarOp $ -> hatPrivateVarInitOp(buildContext, $);
                case HATThreadOp $ -> hatThreadIdOp(buildContext, $);
                case HATVectorOp.HATVectorVarOp $ -> hatVectorVarOp(buildContext, $);
                case HATVectorOp.HATVectorStoreView $ -> hatVectorStoreOp(buildContext, $);
                case HATVectorOp.HATVectorBinaryOp $ -> hatBinaryVectorOp(buildContext, $);
                case HATVectorOp.HATVectorLoadOp $ -> hatVectorLoadOp(buildContext, $);
                case HATVectorOp.HATVectorSelectLoadOp $ -> hatSelectLoadOp(buildContext, $);
                case HATVectorOp.HATVectorSelectStoreOp $ -> hatSelectStoreOp(buildContext, $);
                case HATVectorOp.HATVectorVarLoadOp $ -> hatVectorVarLoadOp(buildContext, $);
                case HATVectorOp.HATVectorOfOp $ -> hatVectorOfOps(buildContext, $);
                case HATF16Op.HATF16VarOp $ -> hatF16VarOp(buildContext, $);
                case HATF16Op.HATF16BinaryOp $ -> hatF16BinaryOp(buildContext, $);
                case HATF16Op.HATF16VarLoadOp $ -> hatF16VarLoadOp(buildContext, $);
                case HATF16Op.HATF16ConvOp $ -> hatF16ConvOp(buildContext, $);
                case HATVectorOp.HATVectorMakeOfOp $ -> hatVectorMakeOf(buildContext, $);
                case HATPtrOp.HATPtrLoadOp $ -> hatPtrLoadOp(buildContext, $);
                case HATPtrOp.HATPtrStoreOp $ -> hatPtrStoreOp(buildContext, $);
                case HATPtrOp.HATPtrLengthOp $ -> hatPtrLengthOp(buildContext, $);
                case HATF16Op.HATF16ToFloatConvOp $ -> hatF16ToFloatConvOp(buildContext, $);
                case HATMemoryDefOp.HATMemoryLoadOp $ -> hatMemoryLoadOp(buildContext, $);
                default -> throw new IllegalStateException("handle nesting of hat op " + op);
            }
        }else{
            BabylonOpDispatcher.super.recurse(buildContext, op);
        }

        return (T) this;
    }
}
