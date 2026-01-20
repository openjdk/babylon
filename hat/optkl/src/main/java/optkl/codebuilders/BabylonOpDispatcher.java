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

/* this should not be too C99 specific also cannot reference HAT Ops. */
public interface BabylonOpDispatcher<T extends JavaOrC99StyleCodeBuilder<T,SCBC>, SCBC extends ScopedCodeBuilderContext> {
    T type( JavaType javaType);

    T varLoadOp( CoreOp.VarAccessOp.VarLoadOp varLoadOp);

    T varStoreOp( CoreOp.VarAccessOp.VarStoreOp varStoreOp);

    T varOp( CoreOp.VarOp varOp);

    T varOp( CoreOp.VarOp varOp, ParamVar paramVar );

    T fieldLoadOp( JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp);

    T fieldStoreOp( JavaOp.FieldAccessOp.FieldStoreOp fieldStoreOp);

    T unaryOp( JavaOp.UnaryOp unaryOp);

    T binaryOp( JavaOp.BinaryOp binaryOp);

    T conditionalOp( JavaOp.JavaConditionalOp conditionalOp);

    T binaryTestOp( JavaOp.BinaryTestOp binaryTestOp);

    T convOp( JavaOp.ConvOp convOp);

    T constantOp( CoreOp.ConstantOp constantOp);

    T yieldOp( CoreOp.YieldOp yieldOp);

    T lambdaOp( JavaOp.LambdaOp lambdaOp);

    T tupleOp( CoreOp.TupleOp tupleOp);

    T funcCallOp( CoreOp.FuncCallOp funcCallOp);

    T ifOp( JavaOp.IfOp ifOp);

    T whileOp( JavaOp.WhileOp whileOp);

    T labeledOp( JavaOp.LabeledOp labeledOp);

    T continueOp( JavaOp.ContinueOp continueOp);

    T breakOp( JavaOp.BreakOp breakOp);

    T forOp( JavaOp.ForOp forOp);

    T invokeOp( JavaOp.InvokeOp invokeOp);

    T conditionalExpressionOp( JavaOp.ConditionalExpressionOp ternaryOp);

    T parenthesisIfNeeded( Op parent, Op child);

    T returnOp( CoreOp.ReturnOp returnOp);

    T newOp( JavaOp.NewOp newOp);
    T arrayLoadOp( JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp);
    T arrayStoreOp( JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp);
    T enhancedForOp( JavaOp.EnhancedForOp enhancedForOp);
    T blockOp( JavaOp.BlockOp blockOp);
    T concatOp( JavaOp.ConcatOp concatOp);
    default T recurse( Op op) {
        switch (op) {
            case CoreOp.VarAccessOp.VarLoadOp $ -> varLoadOp( $);
            case CoreOp.VarAccessOp.VarStoreOp $ -> varStoreOp( $);
            case JavaOp.FieldAccessOp.FieldLoadOp $ -> fieldLoadOp( $);
            case JavaOp.FieldAccessOp.FieldStoreOp $ -> fieldStoreOp( $);
            case JavaOp.ConvOp $ -> convOp( $);
            case CoreOp.ConstantOp $ -> constantOp( $);
            case CoreOp.YieldOp $ -> yieldOp( $);
            case CoreOp.FuncCallOp $ -> funcCallOp( $);
            case JavaOp.InvokeOp $ -> invokeOp( $);
            case JavaOp.ConditionalExpressionOp $ -> conditionalExpressionOp( $);
            case CoreOp.VarOp $ when ParamVar.of($) instanceof ParamVar paramVar -> varOp( $,paramVar);
            case CoreOp.VarOp $ -> varOp( $);
            case JavaOp.LambdaOp $ -> lambdaOp( $);
            case CoreOp.TupleOp $ -> tupleOp( $);
            case JavaOp.WhileOp $ -> whileOp( $);
            case JavaOp.IfOp $ -> ifOp( $);
            case JavaOp.ForOp $ -> forOp( $);
            case CoreOp.ReturnOp $ -> returnOp( $);
            case JavaOp.LabeledOp $ -> labeledOp( $);
            case JavaOp.BreakOp $ -> breakOp( $);
            case JavaOp.ContinueOp $ -> continueOp( $);
            case JavaOp.BinaryTestOp $ -> binaryTestOp( $);
            case JavaOp.BinaryOp $ -> binaryOp( $);
            case JavaOp.JavaConditionalOp $ -> conditionalOp( $);
            case JavaOp.UnaryOp $ -> unaryOp( $);
            case JavaOp.NewOp $ -> newOp( $);
            case JavaOp.ArrayAccessOp.ArrayStoreOp  $ ->  arrayStoreOp($);
            case JavaOp.ArrayAccessOp.ArrayLoadOp  $ ->  arrayLoadOp($);
            case JavaOp.EnhancedForOp $ -> enhancedForOp($);
            case JavaOp.BlockOp   $ -> blockOp($);
            case JavaOp.ConcatOp $ -> concatOp($);
            default -> throw new IllegalStateException("handle nesting of op " + op);
        }
        return (T) this;
    }
}
