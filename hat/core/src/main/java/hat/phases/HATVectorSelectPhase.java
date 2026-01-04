/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.phases;

import hat.callgraph.KernelCallGraph;
import hat.dialect.HATVectorOp;
import hat.types._V;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Invoke;
import optkl.Trxfmr;

import java.util.HashMap;
import java.util.Map;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.asOpFromResultOrNull;
import static optkl.Trxfmr.copyLocation;

public record HATVectorSelectPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        record InvokeVar(JavaOp.InvokeOp invokeOp, CoreOp.VarAccessOp.VarLoadOp varLoadOp){
            // recursive
            static String vectorNameOrThrow(Value v) {
                return switch (asOpFromResultOrNull(v)){
                    case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->vectorNameOrThrow(varLoadOp.operands().getFirst()); // recurse
                    case HATVectorOp vectorOp ->vectorOp.varName();
                    default -> throw new IllegalStateException("failed to find vector name");
                };
            }
            String name(){
                return vectorNameOrThrow(varLoadOp.operands().getFirst());
            }
            //recursive
            private CoreOp.VarOp findVarOpOrNull(Value v) {
                return switch (asOpFromResultOrNull(v)){
                    case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->findVarOpOrNull(varLoadOp.operands().getFirst()); //recurse
                    case CoreOp.VarOp varOp->varOp;
                    default ->  null;
                };
            }
            public CoreOp.VarOp varOpFromOperand(int idx){
                return findVarOpOrNull(invokeOp.operands().get(idx));
            }
            public TypeElement returnType() {
                return invokeOp.resultType();
            }
            int laneIdx() {
                return "xyzw".indexOf(invokeOp.invokeDescriptor().name().charAt(0));
            }

        }

        Map<CodeElement<?,?>, InvokeVar> ceToInvokeVar = new HashMap<>();
        Invoke.stream(lookup(),funcOp)
                .filter(invoke ->
                        invoke.named("x","y","z","w")
                                && invoke.refIs(_V.class)
                                && invoke.opFromFirstOperandAsResultOrThrow() instanceof CoreOp.VarAccessOp.VarLoadOp)
                .map(invoke ->
                        new InvokeVar(invoke.op(),invoke.varLoadOpFromFirstOperandAsResultOrNull())
                )
                .forEach(invokeVar ->{
                    ceToInvokeVar.put(invokeVar.invokeOp,invokeVar);
                    ceToInvokeVar.put(invokeVar.varLoadOp,invokeVar);
                });

        return new Trxfmr(funcOp).transform(ceToInvokeVar::containsKey,(blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (invokeOpHelper(lookup(),op) instanceof Invoke invoke
                    && ceToInvokeVar.get(invoke.op()) instanceof InvokeVar invokeVar) {
                Op newOp = invoke.returnsVoid()
                        ?
                        // Code Model Pattern:
                        //  %16 : java.type:"hat.types.Float4" = var.load %15 @loc="63:28";
                        //  %17 : java.type:"float" = invoke %16 @loc="63:28" @java.ref:"hat.types.Float4::x():float";

                        new HATVectorOp.HATVectorSelectStoreOp(
                                invokeVar.name(),
                                invokeVar.laneIdx(),
                                invokeVar.varOpFromOperand(1),
                                context.getValues(invokeVar.invokeOp.operands())
                        )
                        :
                        // Pattern from the code mode:
                        // %20 : java.type:"hat.types.Float4" = var.load %15 @loc="64:13";
                        // %21 : java.type:"float" = var.load %19 @loc="64:18";
                        // invoke %20 %21 @loc="64:13" @java.ref:"hat.types.Float4::x(float):void";
                        new HATVectorOp.HATVectorSelectLoadOp(
                                invokeVar.name(),
                                invokeVar.returnType(),
                                invokeVar.laneIdx(),
                                context.getValues(invokeVar.invokeOp.operands())
                        );

                context.mapValue(invokeVar.invokeOp.result(), blockBuilder.op(
                        copyLocation(invokeVar.invokeOp, newOp)));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        }).funcOp();
    }

}
