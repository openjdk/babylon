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
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.ParamVar;
import optkl.codebuilders.BabylonCoreOpBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

/* this should not be too C99 specific */
public interface BabylonKernelOpBuilder<T extends HATCodeBuilder<?>> extends BabylonCoreOpBuilder<T, ScopedCodeBuilderContext> {

    T hatBarrierOp(ScopedCodeBuilderContext buildContext, HATBarrierOp barrierOp);

    T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATLocalVarOp barrierOp);

    T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp);

    T hatGlobalThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATGlobalThreadIdOp hatGlobalThreadIdOp);

    T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATGlobalSizeOp hatGlobalSizeOp);

    T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATLocalThreadIdOp hatLocalThreadIdOp);

    T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATLocalSizeOp hatLocalSizeOp);

    T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATBlockThreadIdOp hatBlockThreadIdOp);

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

    default T recurse(ScopedCodeBuilderContext buildContext, Op op) {
        switch (op) {
            case CoreOp.VarAccessOp.VarLoadOp $ -> varLoadOp(buildContext, $);
            case CoreOp.VarAccessOp.VarStoreOp $ -> varStoreOp(buildContext, $);
            case JavaOp.FieldAccessOp.FieldLoadOp $ -> fieldLoadOp(buildContext, $);
            case JavaOp.FieldAccessOp.FieldStoreOp $ -> fieldStoreOp(buildContext, $);
            case JavaOp.ConvOp $ -> convOp(buildContext, $);
            case CoreOp.ConstantOp $ -> constantOp(buildContext, $);
            case CoreOp.YieldOp $ -> yieldOp(buildContext, $);
            case CoreOp.FuncCallOp $ -> funcCallOp(buildContext, $);
            case JavaOp.InvokeOp $ -> invokeOp(buildContext, $);
            case JavaOp.ConditionalExpressionOp $ -> conditionalExpressionOp(buildContext, $);
            case CoreOp.VarOp $ when ParamVar.of($) instanceof ParamVar paramVar -> varOp(buildContext, $,paramVar);
            case CoreOp.VarOp $ -> varOp(buildContext, $);
            case JavaOp.LambdaOp $ -> lambdaOp(buildContext, $);
            case CoreOp.TupleOp $ -> tupleOp(buildContext, $);
            case JavaOp.WhileOp $ -> whileOp(buildContext, $);
            case JavaOp.IfOp $ -> ifOp(buildContext, $);
            case JavaOp.ForOp $ -> forOp(buildContext, $);
            case CoreOp.ReturnOp $ -> returnOp(buildContext, $);
            case JavaOp.LabeledOp $ -> labeledOp(buildContext, $);
            case JavaOp.BreakOp $ -> breakOp(buildContext, $);
            case JavaOp.ContinueOp $ -> continueOp(buildContext, $);
            case JavaOp.BinaryTestOp $ -> binaryTestOp(buildContext, $);
            case JavaOp.BinaryOp $ -> binaryOp(buildContext, $);
            case JavaOp.JavaConditionalOp $ -> conditionalOp(buildContext, $);
            case JavaOp.UnaryOp $ -> unaryOp(buildContext, $);
            case HATBarrierOp $ -> hatBarrierOp(buildContext, $);
            case HATMemoryVarOp.HATLocalVarOp $ -> hatLocalVarOp(buildContext, $);
            case HATMemoryVarOp.HATPrivateVarOp $ -> hatPrivateVarOp(buildContext, $);
            case HATMemoryVarOp.HATPrivateInitVarOp $ -> hatPrivateVarInitOp(buildContext, $);
            case HATThreadOp.HATGlobalThreadIdOp $ -> hatGlobalThreadIdOp(buildContext, $);
            case HATThreadOp.HATGlobalSizeOp $ -> hatGlobalSizeOp(buildContext, $);
            case HATThreadOp.HATLocalThreadIdOp $ -> hatLocalThreadIdOp(buildContext, $);
            case HATThreadOp.HATLocalSizeOp $ -> hatLocalSizeOp(buildContext, $);
            case HATThreadOp.HATBlockThreadIdOp $ -> hatBlockThreadIdOp(buildContext, $);
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
            default -> throw new IllegalStateException("handle nesting of op " + op);
        }
        return (T) this;
    }
}
