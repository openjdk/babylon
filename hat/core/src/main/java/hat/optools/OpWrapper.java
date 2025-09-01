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
        // This is possibly a premature optimization. But it allows us to treat vardeclarations differently from params.
        if (op instanceof CoreOp.VarOp varOp && !varOp.isUninitialized()) {
            // this gets called a lot and we can't wrap yet or we recurse so we
            // use the raw model. Basically we want a different wrapper for VarDeclarations
            // which  relate to func parameters.
            // This saves us asking each time if a var is indeed a func param.

            if (varOp.operands().getFirst() instanceof Block.Parameter parameter &&
                    parameter.invokableOperation() instanceof CoreOp.FuncOp funcOp) {
                return (OW) new VarFuncDeclarationOpWrapper(lookup,varOp, funcOp, parameter);
            }
        }
        return switch (op) {
            case CoreOp.ModuleOp $ -> (OW) new ModuleOpWrapper(lookup, $);
            case JavaOp.ForOp $ -> (OW) new ForOpWrapper(lookup, $);
            case JavaOp.WhileOp $ -> (OW) new WhileOpWrapper(lookup, $);
            case JavaOp.IfOp $ -> (OW) new IfOpWrapper(lookup, $);
            case JavaOp.NotOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper(lookup, $);
            case JavaOp.NegOp $ -> (OW) new UnaryArithmeticOrLogicOpWrapper(lookup, $);
            case JavaOp.BinaryOp $ -> (OW) new BinaryArithmeticOrLogicOperation(lookup, $);
            case JavaOp.BinaryTestOp $ -> (OW) new BinaryTestOpWrapper(lookup, $);
            case CoreOp.FuncOp $ -> (OW) new FuncOpWrapper(lookup, $);
            case CoreOp.VarOp $ -> (OW) new VarDeclarationOpWrapper(lookup, $);
            case CoreOp.YieldOp $ -> (OW) new YieldOpWrapper(lookup, $);
            case CoreOp.FuncCallOp $ -> (OW) new FuncCallOpWrapper(lookup, $);
            case JavaOp.ConvOp $ -> (OW) new ConvOpWrapper(lookup, $);
            case CoreOp.ConstantOp $ -> (OW) new ConstantOpWrapper(lookup, $);
            case CoreOp.ReturnOp $ -> (OW) new ReturnOpWrapper(lookup, $);
            case CoreOp.VarAccessOp.VarStoreOp $ -> (OW) new VarStoreOpWrapper(lookup, $);
            case CoreOp.VarAccessOp.VarLoadOp $ -> (OW) new VarLoadOpWrapper(lookup, $);
            case JavaOp.FieldAccessOp.FieldStoreOp $ -> (OW) new FieldStoreOpWrapper(lookup, $);
            case JavaOp.FieldAccessOp.FieldLoadOp $ -> (OW) new FieldLoadOpWrapper(lookup, $);
            case JavaOp.InvokeOp $ -> (OW) new InvokeOpWrapper(lookup, $);
            case CoreOp.TupleOp $ -> (OW) new TupleOpWrapper(lookup, $);
            case JavaOp.LambdaOp $ -> (OW) new LambdaOpWrapper(lookup, $);
            case JavaOp.JavaConditionalOp $ -> (OW) new LogicalOpWrapper(lookup, $);
            case JavaOp.ConditionalExpressionOp $ -> (OW) new TernaryOpWrapper(lookup, $);
            case JavaOp.LabeledOp $ -> (OW) new JavaLabeledOpWrapper(lookup, $);
            case JavaOp.BreakOp $ -> (OW) new JavaBreakOpWrapper(lookup, $);
            case JavaOp.ContinueOp $ -> (OW) new JavaContinueOpWrapper(lookup, $);
            default -> (OW) new OpWrapper<>(lookup,op);
        };
    }

    public final T op;
    public final MethodHandles.Lookup lookup;
    OpWrapper( MethodHandles.Lookup lookup,T op) {
        this.lookup= lookup;
        this.op = op;
    }

}
