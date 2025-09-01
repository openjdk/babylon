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
package hat.optools;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.lang.invoke.MethodHandles;

public class OpWrapper<T extends Op> {
    @SuppressWarnings("unchecked")
    public static <O extends Op, OW extends OpWrapper<O>> OW wrap(MethodHandles.Lookup lookup,O op) {
        // We have one special case
        // This is possibly a premature optimization. But it allows us to treat var declarations differently from params.
        // this gets called a lot and we can't wrap yet or we recurse so we
        // use the raw model. Basically we want a different wrapper for VarDeclarations
        // which  relate to func parameters.
        // This saves us asking each time if a var is indeed a func param.
        if (op instanceof CoreOp.VarOp varOp
                && !varOp.isUninitialized()
                && varOp.operands().getFirst() instanceof Block.Parameter parameter
                && parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp) {
                return (OW) new VarFuncDeclarationOpWrapper(varOp, funcOp, parameter);
        }
        return switch (op) {
            case CoreOp.FuncOp $ -> (OW) new FuncOpWrapper(lookup, $);
            case JavaOp.InvokeOp $ -> (OW) new InvokeOpWrapper(lookup, $);
            case JavaOp.LambdaOp $ -> (OW) new LambdaOpWrapper(lookup, $);

            case JavaOp.ForOp $ -> (OW) new ForOpWrapper( $);
            case JavaOp.WhileOp $ -> (OW) new WhileOpWrapper( $);
            case CoreOp.ModuleOp $ -> (OW) new ModuleOpWrapper( $);
            case JavaOp.IfOp $ -> (OW) new IfOpWrapper( $);
            case CoreOp.TupleOp $ -> (OW) new TupleOpWrapper( $);
            case JavaOp.LabeledOp $ -> (OW) new JavaLabeledOpWrapper( $);
            case JavaOp.NotOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper( $);
            case JavaOp.NegOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper( $);
            case JavaOp.BinaryOp $ -> (OW) new BinaryArithmeticOrLogicOperation( $);
            case JavaOp.BinaryTestOp $ -> (OW) new BinaryTestOpWrapper( $);
            case CoreOp.VarOp $ -> (OW) new VarDeclarationOpWrapper( $);
            case CoreOp.YieldOp $ -> (OW) new YieldOpWrapper( $);
            case CoreOp.FuncCallOp $ -> (OW) new FuncCallOpWrapper( $);
            case JavaOp.ConvOp $ -> (OW) new ConvOpWrapper( $);
            case CoreOp.ConstantOp $ -> (OW) new ConstantOpWrapper( $);
            case CoreOp.ReturnOp $ -> (OW) new ReturnOpWrapper( $);
            case CoreOp.VarAccessOp.VarStoreOp $ -> (OW) new VarStoreOpWrapper( $);
            case CoreOp.VarAccessOp.VarLoadOp $ -> (OW) new VarLoadOpWrapper( $);
            case JavaOp.FieldAccessOp.FieldStoreOp $ -> (OW) new FieldStoreOpWrapper( $);
            case JavaOp.FieldAccessOp.FieldLoadOp $ -> (OW) new FieldLoadOpWrapper( $);
            case JavaOp.JavaConditionalOp $ -> (OW) new LogicalOpWrapper( $);
            case JavaOp.ConditionalExpressionOp $ -> (OW) new TernaryOpWrapper( $);
            case JavaOp.BreakOp $ -> (OW) new JavaBreakOpWrapper( $);
            case JavaOp.ContinueOp $ -> (OW) new JavaContinueOpWrapper( $);
            default -> (OW) new OpWrapper<>(op);
        };
    }

    public final T op;
    OpWrapper(T op) {
        this.op = op;
    }

}
