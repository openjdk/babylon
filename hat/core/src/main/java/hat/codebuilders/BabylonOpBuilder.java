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

import hat.dialect.HatBarrierOp;
import hat.dialect.HatBlockThreadIdOp;
import hat.dialect.HatVSelectLoadOp;
import hat.dialect.HatVSelectStoreOp;
import hat.dialect.HatVectorBinaryOp;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorStoreView;
import hat.dialect.HatGlobalThreadIdOp;
import hat.dialect.HatGlobalSizeOp;
import hat.dialect.HatLocalSizeOp;
import hat.dialect.HatLocalThreadIdOp;
import hat.dialect.HatLocalVarOp;
import hat.dialect.HatPrivateVarOp;
import hat.dialect.HatVectorVarLoadOp;
import hat.dialect.HatVectorVarOp;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

/* this should not be too C99 specific */
public interface BabylonOpBuilder<T extends HATCodeBuilderWithContext<?>> {

    T varLoadOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp);

    T varStoreOp(ScopedCodeBuilderContext buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp);

    T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp);

    T varOp(ScopedCodeBuilderContext buildContext, CoreOp.VarOp varOp, OpTk.ParamVar paramVar );

    T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp);

    T fieldStoreOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp);

    T unaryOp(ScopedCodeBuilderContext buildContext, JavaOp.UnaryOp unaryOp);

    T binaryOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryOp binaryOp);

    T conditionalOp(ScopedCodeBuilderContext buildContext, JavaOp.JavaConditionalOp conditionalOp);

    T binaryTestOp(ScopedCodeBuilderContext buildContext, JavaOp.BinaryTestOp binaryTestOp);

    T convOp(ScopedCodeBuilderContext buildContext, JavaOp.ConvOp convOp);

    T constantOp(ScopedCodeBuilderContext buildContext, CoreOp.ConstantOp constantOp);

    T yieldOp(ScopedCodeBuilderContext buildContext, CoreOp.YieldOp yieldOp);

    T lambdaOp(ScopedCodeBuilderContext buildContext, JavaOp.LambdaOp lambdaOp);

    T tupleOp(ScopedCodeBuilderContext buildContext, CoreOp.TupleOp tupleOp);

    T funcCallOp(ScopedCodeBuilderContext buildContext, CoreOp.FuncCallOp funcCallOp);

    T ifOp(ScopedCodeBuilderContext buildContext, JavaOp.IfOp ifOp);

    T whileOp(ScopedCodeBuilderContext buildContext, JavaOp.WhileOp whileOp);

    T labeledOp(ScopedCodeBuilderContext buildContext, JavaOp.LabeledOp labeledOp);

    T continueOp(ScopedCodeBuilderContext buildContext, JavaOp.ContinueOp continueOp);

    T breakOp(ScopedCodeBuilderContext buildContext, JavaOp.BreakOp breakOp);

    T forOp(ScopedCodeBuilderContext buildContext, JavaOp.ForOp forOp);

    T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp);

    T conditionalExpressionOp(ScopedCodeBuilderContext buildContext, JavaOp.ConditionalExpressionOp ternaryOp);

    T parenthesisIfNeeded(ScopedCodeBuilderContext buildContext, Op parent, Op child);

    T returnOp(ScopedCodeBuilderContext buildContext, CoreOp.ReturnOp returnOp);

    T barrier(ScopedCodeBuilderContext buildContext, HatBarrierOp barrierOp);

    T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HatLocalVarOp barrierOp);

    T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HatPrivateVarOp hatLocalVarOp);

    T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HatGlobalThreadIdOp hatGlobalThreadIdOp);

    T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HatGlobalSizeOp hatGlobalSizeOp);

    T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HatLocalThreadIdOp hatLocalThreadIdOp);

    T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HatLocalSizeOp hatLocalSizeOp);

    T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HatBlockThreadIdOp hatBlockThreadIdOp);

    T hatVectorVarOp(ScopedCodeBuilderContext buildContext, HatVectorVarOp hatVectorVarOp);

    T hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatFloat4StoreOp);

    T hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp);

    T hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp);

    T hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp);

    T hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp);

    T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HatVectorVarLoadOp hatVectorVarLoadOp);

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
            case CoreOp.VarOp $ when OpTk.paramVar($) instanceof OpTk.ParamVar paramVar -> varOp(buildContext, $,paramVar);
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
            case HatBarrierOp $ -> barrier(buildContext, $);
            case HatLocalVarOp $ -> hatLocalVarOp(buildContext, $);
            case HatPrivateVarOp $ -> hatPrivateVarOp(buildContext, $);
            case HatGlobalThreadIdOp $ -> hatGlobalThreadOp(buildContext, $);
            case HatGlobalSizeOp $ -> hatGlobalSizeOp(buildContext, $);
            case HatLocalThreadIdOp $ -> hatLocalThreadIdOp(buildContext, $);
            case HatLocalSizeOp $ -> hatLocalSizeOp(buildContext, $);
            case HatBlockThreadIdOp $ -> hatBlockThreadIdOp(buildContext, $);
            case HatVectorVarOp $ -> hatVectorVarOp(buildContext, $);
            case HatVectorStoreView $ -> hatVectorStoreOp(buildContext, $);
            case HatVectorBinaryOp $ -> hatBinaryVectorOp(buildContext, $);
            case HatVectorLoadOp $ -> hatVectorLoadOp(buildContext, $);
            case HatVSelectLoadOp $ -> hatSelectLoadOp(buildContext, $);
            case HatVSelectStoreOp $ -> hatSelectStoreOp(buildContext, $);
            case HatVectorVarLoadOp $ -> hatVectorVarLoadOp(buildContext, $);
            default -> throw new IllegalStateException("handle nesting of op " + op);
        }
        return (T) this;
    }
}
