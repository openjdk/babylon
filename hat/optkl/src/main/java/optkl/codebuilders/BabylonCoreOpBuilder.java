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
package optkl.codebuilders;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.ParamVar;

/* this should not be too C99 specific */
public interface BabylonCoreOpBuilder<T extends CodeBuilder<?>, SB extends CodeBuilderContext> {
    T type(SB buildContext, JavaType javaType);

    T varLoadOp(SB buildContext, CoreOp.VarAccessOp.VarLoadOp varLoadOp);

    T varStoreOp(SB buildContext, CoreOp.VarAccessOp.VarStoreOp varStoreOp);

    T varOp(SB buildContext, CoreOp.VarOp varOp);

    T varOp(SB buildContext, CoreOp.VarOp varOp, ParamVar paramVar );

    T fieldLoadOp(SB buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp);

    T fieldStoreOp(SB buildContext, JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp);

    T unaryOp(SB buildContext, JavaOp.UnaryOp unaryOp);

    T binaryOp(SB buildContext, JavaOp.BinaryOp binaryOp);

    T conditionalOp(SB buildContext, JavaOp.JavaConditionalOp conditionalOp);

    T binaryTestOp(SB buildContext, JavaOp.BinaryTestOp binaryTestOp);

    T convOp(SB buildContext, JavaOp.ConvOp convOp);

    T constantOp(SB buildContext, CoreOp.ConstantOp constantOp);

    T yieldOp(SB buildContext, CoreOp.YieldOp yieldOp);

    T lambdaOp(SB buildContext, JavaOp.LambdaOp lambdaOp);

    T tupleOp(SB buildContext, CoreOp.TupleOp tupleOp);

    T funcCallOp(SB buildContext, CoreOp.FuncCallOp funcCallOp);

    T ifOp(SB buildContext, JavaOp.IfOp ifOp);

    T whileOp(SB buildContext, JavaOp.WhileOp whileOp);

    T labeledOp(SB buildContext, JavaOp.LabeledOp labeledOp);

    T continueOp(SB buildContext, JavaOp.ContinueOp continueOp);

    T breakOp(SB buildContext, JavaOp.BreakOp breakOp);

    T forOp(SB buildContext, JavaOp.ForOp forOp);

    T invokeOp(SB buildContext, JavaOp.InvokeOp invokeOp);

    T conditionalExpressionOp(SB buildContext, JavaOp.ConditionalExpressionOp ternaryOp);

    T parenthesisIfNeeded(SB buildContext, Op parent, Op child);

    T returnOp(SB buildContext, CoreOp.ReturnOp returnOp);

    default T recurse(SB buildContext, Op op) {
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

            default -> throw new IllegalStateException("handle nesting of op " + op);
        }
        return (T) this;
    }
}
