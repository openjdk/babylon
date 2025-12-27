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
import hat.dialect.HATVectorSelectLoadOp;
import hat.dialect.HATVectorSelectStoreOp;
import hat.dialect.HATVectorOp;
import hat.types._V;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.CallSite;
import optkl.OpTkl;
import optkl.util.Regex;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static hat.optools.RefactorMe.inspectAllInterfaces;
import static optkl.OpTkl.asOpFromResultOrNull;
import static optkl.OpTkl.isMethod;
import static optkl.OpTkl.transform;

public record HATVectorSelectPhase(KernelCallGraph kernelCallGraph) implements HATPhase {
    private static final Regex xyzw = Regex.of("[xyzw]");

    private boolean isVectorLane(JavaOp.InvokeOp invokeOp) {
        return isMethod(invokeOp, n->xyzw.matches(n));
    }
    static boolean isVectorOperation(JavaOp.InvokeOp invokeOp, boolean laneOk) {
        String typeElement = invokeOp.invokeDescriptor().refType().toString();
        Set<Class<?>> interfaces;
        try {
            Class<?> aClass = Class.forName(typeElement); // WHY?
            interfaces = inspectAllInterfaces(aClass);
        } catch (ClassNotFoundException _) {
            return false;
        }
        return interfaces.contains(_V.class) && laneOk;
    }
    int getLane(String fieldName) {
        return switch (fieldName) {
            case "x" -> 0;
            case "y" -> 1;
            case "z" -> 2;
            case "w" -> 3;
            default -> -1;
        };
    }

    // recursive
    private String findNameVector(Value v) {
        if (OpTkl.asOpFromResultOrNull(v) instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp.operands().getFirst());
        } else if (OpTkl.asOpFromResultOrNull(v)  instanceof HATVectorOp vectorViewOp) {
            return vectorViewOp.varName();
        }
        throw new IllegalStateException("recurse fail findNameVector");

    }


    //recursive
    private CoreOp.VarOp findVarOp(Value v) {
        if (asOpFromResultOrNull(v) instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVarOp(varLoadOp.operands().getFirst());
        } else if (asOpFromResultOrNull(v) instanceof CoreOp.VarOp varOp) {
            return varOp;
        }
        return null;

    }

    // Code Model Pattern:
    //  %16 : java.type:"hat.types.Float4" = var.load %15 @loc="63:28";
    //  %17 : java.type:"float" = invoke %16 @loc="63:28" @java.ref:"hat.types.Float4::x():float";
    private CoreOp.FuncOp vloadSelectPhase(CoreOp.FuncOp funcOp) {
        var here = CallSite.of(this.getClass(), "vloadSelectPhase");
        before(here, funcOp);
        Stream<CodeElement<?, ?>> vectorSelectOps = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp, isVectorLane(invokeOp)) && (invokeOp.resultType() != JavaType.VOID)) {
                            Value inputOperand = invokeOp.operands().getFirst();
                            if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                consumer.accept(invokeOp);
                                consumer.accept(varLoadOp);
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorSelectOps.collect(Collectors.toSet());
        funcOp = transform(here, funcOp, (blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputInvokeOp = invokeOp.operands();
                for (Value v : inputInvokeOp) {
                    if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        List<Value> outputOperandsInvokeOp = context.getValues(inputInvokeOp);
                        int lane = getLane(invokeOp.invokeDescriptor().name());
                        HATVectorOp vSelectOp;
                        String name = findNameVector(varLoadOp.operands().getFirst());
                        if (invokeOp.resultType() != JavaType.VOID) {
                            vSelectOp = new HATVectorSelectLoadOp(name, invokeOp.resultType(), lane, outputOperandsInvokeOp);
                        } else {
                            throw new RuntimeException("VSelect Load Op must return a value!");
                        }
                        Op.Result hatSelectResult = blockBuilder.op(vSelectOp);
                        vSelectOp.setLocation(invokeOp.location());
                        context.mapValue(invokeOp.result(), hatSelectResult);
                    }
                }
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // Pass the value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
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
        //TODO is this side table safe?
        Stream<CodeElement<?, ?>> float4NodesInvolved = OpTkl.elements(here, funcOp)
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp, isVectorLane(invokeOp))) {
                            List<Value> inputOperandsInvoke = invokeOp.operands();
                            Value inputOperand = inputOperandsInvoke.getFirst();
                            if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                consumer.accept(invokeOp);
                                consumer.accept(varLoadOp);
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());
        funcOp = transform(here, funcOp, (blockBuilder, op) -> {
            CodeContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputInvokeOp = invokeOp.operands();
                Value v = inputInvokeOp.getFirst();

                if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    List<Value> outputOperandsInvokeOp = context.getValues(inputInvokeOp);
                    int lane = getLane(invokeOp.invokeDescriptor().name());
                    HATVectorOp vSelectOp;
                    String name = findNameVector(varLoadOp.operands().getFirst());
                    if (invokeOp.resultType() == JavaType.VOID) {
                        // The operand 1 in the store is the address (lane)
                        // The operand 1 in the store is the storeValue
                        CoreOp.VarOp resultOp = findVarOp(outputOperandsInvokeOp.get(1));
                        vSelectOp = new HATVectorSelectStoreOp(name, invokeOp.resultType(), lane, resultOp, outputOperandsInvokeOp);
                    } else {
                        throw new RuntimeException("VSelect Store Op must return a value!");
                    }
                    Op.Result resultVStore = blockBuilder.op(vSelectOp);
                    vSelectOp.setLocation(invokeOp.location());
                    context.mapValue(invokeOp.result(), resultVStore);
                }

            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // Pass the value
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
