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
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Invoke;
import optkl.util.CallSite;
import optkl.util.Regex;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.asOpFromResultOrNull;
import static optkl.OpTkl.transform;
import static optkl.Trxfmr.copyLocation;

public record HATVectorSelectPhase(KernelCallGraph kernelCallGraph) implements HATPhase {

    int laneIdxOrThrow(String fieldName) {
        return "xyzw".indexOf(fieldName.charAt(0));
     /*   return switch (fieldName) {
            case "x" -> 0;
            case "y" -> 1;
            case "z" -> 2;
            case "w" -> 3;
            default -> throw new RuntimeException("fieldName not x,y,z,w");
        };*/
    }

    // recursive
    private String vectorNameOrThrow(Value v) {
       return switch (asOpFromResultOrNull(v)){
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->vectorNameOrThrow(varLoadOp.operands().getFirst()); // recurse
            case HATVectorOp vectorOp ->vectorOp.varName();
            default -> throw new IllegalStateException("failed to find vector name");
        };
    }


    //recursive
    private CoreOp.VarOp findVarOpOrNull(Value v) {
        return switch (asOpFromResultOrNull(v)){
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->findVarOpOrNull(varLoadOp.operands().getFirst()); //recurse
            case CoreOp.VarOp varOp->varOp;
            default ->  null;
        };
    }

    // Code Model Pattern:
    //  %16 : java.type:"hat.types.Float4" = var.load %15 @loc="63:28";
    //  %17 : java.type:"float" = invoke %16 @loc="63:28" @java.ref:"hat.types.Float4::x():float";
    private CoreOp.FuncOp vloadSelectPhase(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "vloadSelectPhase");
        before(here, funcOp);
        Set<CoreOp.VarAccessOp.VarLoadOp> varLoadOps =new HashSet<>();
        Map<JavaOp.InvokeOp, String> invokeToVectorName = new HashMap<>();
        Invoke.stream(lookup(),funcOp)
                .filter(invoke -> !invoke.returnsVoid() && invoke.named("x","y","z","w") && invoke.refIs(_V.class))
                .forEach(invoke -> {
                    var varLoadOp = invoke.varLoadOpFromFirstOperandAsResultOrThrow();
                    invokeToVectorName.put(invoke.op(), vectorNameOrThrow(varLoadOp.operands().getFirst()));
                    varLoadOps.add(varLoadOp);
                });

        funcOp = transform(here, funcOp
                ,ce-> (varLoadOps.contains(ce)||invokeToVectorName.containsKey(ce)),
                (blockBuilder, op) -> {
            if (invokeOpHelper(lookup(), op) instanceof Invoke invoke) {
                blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(
                        copyLocation(invoke.op(), new HATVectorOp.HATVectorSelectLoadOp(
                                        invokeToVectorName.get(invoke.op()),
                                        invoke.returnType(),
                                        laneIdxOrThrow(invoke.name()),
                                        blockBuilder.context().getValues(invoke.op().operands())
                                )
                        )
                ));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                blockBuilder.context().mapValue(varLoadOp.result(), blockBuilder.context().getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        });

        after(here, funcOp);
        return funcOp;
    }

    // Pattern from the code mode:
    // %20 : java.type:"hat.types.Float4" = var.load %15 @loc="64:13";
    // %21 : java.type:"float" = var.load %19 @loc="64:18";
    // invoke %20 %21 @loc="64:13" @java.ref:"hat.types.Float4::x(float):void";
    private CoreOp.FuncOp vstoreSelectPhase(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "vstoreSelectPhase");
        before(here, funcOp);
        Set<CoreOp.VarAccessOp.VarLoadOp> varLoads = new HashSet<>();
        Map<JavaOp.InvokeOp, CoreOp.VarAccessOp.VarLoadOp> invokeToVarLoadOp  = new HashMap<>();
        Invoke.stream(lookup(),funcOp)
                .filter($ -> $.named("x","y","z","w") && $.returnsVoid() &&  $.refIs(_V.class))
                .forEach($ ->{
                    var varLoadOp = $.varLoadOpFromFirstOperandAsResultOrThrow();
                    invokeToVarLoadOp.put($.op(),varLoadOp);
                    varLoads.add(varLoadOp);
                });

        funcOp = transform(here, funcOp,
                ce->varLoads.contains(ce)||invokeToVarLoadOp.containsKey(ce), // only the nodes we mapped/selected
                (blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (invokeOpHelper(lookup(),op) instanceof Invoke invoke) {
                List<Value> outputOperandsInvokeOp = context.getValues( invoke.op().operands());
                context.mapValue(invoke.op().result(), blockBuilder.op(copyLocation(invoke.op(), new HATVectorOp.HATVectorSelectStoreOp(
                                vectorNameOrThrow(invokeToVarLoadOp.get(invoke.op()).operands().getFirst()),
                                invoke.returnType(),
                                laneIdxOrThrow(invoke.name()),
                                // The operand 1 in the store is the address (lane)
                                // The operand 1 in the store is the storeValue
                                findVarOpOrNull(outputOperandsInvokeOp.get(1)),
                                outputOperandsInvokeOp
                        )
                )));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        });

        after(here, funcOp);
        return funcOp;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        funcOp = vloadSelectPhase(funcOp);
        funcOp = vstoreSelectPhase(funcOp);
        return funcOp;
    }

}
