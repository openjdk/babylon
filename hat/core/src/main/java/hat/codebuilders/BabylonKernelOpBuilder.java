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
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.ParamVar;
import optkl.codebuilders.BabylonCoreOpBuilder;

/* this should not be too C99 specific */
public interface BabylonKernelOpBuilder<T extends HATCodeBuilder<?>> extends BabylonCoreOpBuilder<T,ScopedCodeBuilderContext> {

    T hatBarrierOp(ScopedCodeBuilderContext buildContext, HATBarrierOp barrierOp);

    T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HATLocalVarOp barrierOp);

    T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HATPrivateVarOp hatLocalVarOp);

    T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HATGlobalThreadIdOp hatGlobalThreadIdOp);

    T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HATGlobalSizeOp hatGlobalSizeOp);

    T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HATLocalThreadIdOp hatLocalThreadIdOp);

    T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HATLocalSizeOp hatLocalSizeOp);

    T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HATBlockThreadIdOp hatBlockThreadIdOp);

    T hatVectorVarOp(ScopedCodeBuilderContext buildContext, HATVectorVarOp hatVectorVarOp);

    T hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HATVectorStoreView hatFloat4StoreOp);

    T hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HATVectorBinaryOp hatVectorBinaryOp);

    T hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HATVectorLoadOp hatVectorLoadOp);

    T hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HATVectorSelectLoadOp hatVSelectLoadOp);

    T hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HATVectorSelectStoreOp hatVSelectStoreOp);

    T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HATVectorVarLoadOp hatVectorVarLoadOp);

    T hatF16VarOp(ScopedCodeBuilderContext buildContext, HATF16VarOp hatF16VarOp);

    T hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16BinaryOp hatF16BinaryOp);

    T hatF16VarLoadOp(ScopedCodeBuilderContext buildContext, HATF16VarLoadOp hatF16VarLoadOp);

    T hatF16ConvOp(ScopedCodeBuilderContext buildContext, HATF16ConvOp hatF16ConvOp);

    T hatVectorOfOps(ScopedCodeBuilderContext buildContext, HATVectorOfOp hatVectorOp);

    T hatVectorMakeOf(ScopedCodeBuilderContext buildContext, HATVectorMakeOfOp hatVectorMakeOfOp);

    T hatF16ToFloatConvOp(ScopedCodeBuilderContext buildContext, HATF16ToFloatConvOp hatF16ToFloatConvOp);

    T hatPrivateVarInitOp(ScopedCodeBuilderContext buildContext, HATPrivateInitVarOp hatPrivateInitVarOp);

    T hatMemoryLoadOp(ScopedCodeBuilderContext buildContext, HATMemoryLoadOp hatMemoryLoadOp);

    T hatPtrLoadOp(ScopedCodeBuilderContext builderContext, HATPtrLoadOp hatPtrLoadOp);

    T hatPtrStoreOp(ScopedCodeBuilderContext builderContext, HATPtrStoreOp hatPtrStoreOp);

    T hatPtrLengthOp(ScopedCodeBuilderContext builderContext, HATPtrLengthOp hatPtrLengthOp);

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
            case HATLocalVarOp $ -> hatLocalVarOp(buildContext, $);
            case HATPrivateVarOp $ -> hatPrivateVarOp(buildContext, $);
            case HATPrivateInitVarOp $ -> hatPrivateVarInitOp(buildContext, $);
            case HATGlobalThreadIdOp $ -> hatGlobalThreadOp(buildContext, $);
            case HATGlobalSizeOp $ -> hatGlobalSizeOp(buildContext, $);
            case HATLocalThreadIdOp $ -> hatLocalThreadIdOp(buildContext, $);
            case HATLocalSizeOp $ -> hatLocalSizeOp(buildContext, $);
            case HATBlockThreadIdOp $ -> hatBlockThreadIdOp(buildContext, $);
            case HATVectorVarOp $ -> hatVectorVarOp(buildContext, $);
            case HATVectorStoreView $ -> hatVectorStoreOp(buildContext, $);
            case HATVectorBinaryOp $ -> hatBinaryVectorOp(buildContext, $);
            case HATVectorLoadOp $ -> hatVectorLoadOp(buildContext, $);
            case HATVectorSelectLoadOp $ -> hatSelectLoadOp(buildContext, $);
            case HATVectorSelectStoreOp $ -> hatSelectStoreOp(buildContext, $);
            case HATVectorVarLoadOp $ -> hatVectorVarLoadOp(buildContext, $);
            case HATVectorOfOp $ -> hatVectorOfOps(buildContext, $);
            case HATF16VarOp $ -> hatF16VarOp(buildContext, $);
            case HATF16BinaryOp $ -> hatF16BinaryOp(buildContext, $);
            case HATF16VarLoadOp $ -> hatF16VarLoadOp(buildContext, $);
            case HATF16ConvOp $ -> hatF16ConvOp(buildContext, $);
            case HATVectorMakeOfOp $ -> hatVectorMakeOf(buildContext, $);
            case HATPtrLoadOp $ -> hatPtrLoadOp(buildContext, $);
            case HATPtrStoreOp $ -> hatPtrStoreOp(buildContext, $);
            case HATPtrLengthOp $ -> hatPtrLengthOp(buildContext, $);
            case HATF16ToFloatConvOp $ -> hatF16ToFloatConvOp(buildContext, $);
            case HATMemoryLoadOp $ -> hatMemoryLoadOp(buildContext, $);
            default -> throw new IllegalStateException("handle nesting of op " + op);
        }
        return (T) this;
    }
}
