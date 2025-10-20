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

import hat.Accelerator;
import hat.dialect.HATVectorSelectLoadOp;
import hat.dialect.HATVectorSelectStoreOp;
import hat.dialect.HATVectorViewOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyVectorSelectPhase implements HATDialect {

    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    public HATDialectifyVectorSelectPhase(Accelerator accelerator) {
        this.accelerator = accelerator;
    }

    private boolean isVectorLane(JavaOp.InvokeOp invokeOp) {
        return isMethod(invokeOp, "x")
                || isMethod(invokeOp, "y")
                || isMethod(invokeOp, "z")
                || isMethod(invokeOp, "w");
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

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
        String invokeClass = invokeOp.invokeDescriptor().refType().toString();
        boolean isHatVectorType = invokeClass.startsWith("hat.buffer.Float");
        return isHatVectorType
                && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                && (isVectorLane(invokeOp));
    }

    private String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().get(0));
    }

    private String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorViewOp vectorViewOp) {
                return vectorViewOp.varName();
            }
            return null;
        }
    }

    private CoreOp.VarOp findVarOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVarOp(varLoadOp.operands().get(0));
    }

    private CoreOp.VarOp findVarOp(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVarOp(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && r.op() instanceof CoreOp.VarOp varOp) {
                return varOp;
            }
            return null;
        }
    }


    // Code Model Pattern:
    //  %16 : java.type:"hat.buffer.Float4" = var.load %15 @loc="63:28";
    //  %17 : java.type:"float" = invoke %16 @loc="63:28" @java.ref:"hat.buffer.Float4::x():float";

    private CoreOp.FuncOp vloadSelectPhase(CoreOp.FuncOp funcOp) {

        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[BEFORE] VSelect Load Transform: " + funcOp.toText());
        }
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp) && invokeOp.resultType() != JavaType.VOID) {
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
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }

        var here = OpTk.CallSite.of(HATDialectifyVectorSelectPhase.class, "vloadSelectPhase");
        funcOp = OpTk.transform(here, funcOp,(blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputInvokeOp = invokeOp.operands();
                for (Value v : inputInvokeOp) {
                    if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                        List<Value> outputOperandsInvokeOp = context.getValues(inputInvokeOp);
                        int lane = getLane(invokeOp.invokeDescriptor().name());
                        HATVectorViewOp vSelectOp;
                        String name = findNameVector(varLoadOp);
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

        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[After] VSelect Load Transform: " + funcOp.toText());
        }
        return funcOp;
    }

    // Pattern from the code mode:
    // %20 : java.type:"hat.buffer.Float4" = var.load %15 @loc="64:13";
    // %21 : java.type:"float" = var.load %19 @loc="64:18";
    // invoke %20 %21 @loc="64:13" @java.ref:"hat.buffer.Float4::x(float):void";
    private CoreOp.FuncOp vstoreSelectPhase(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[BEFORE] VSelect Store Transform " + funcOp.toText());
        }
        var here = OpTk.CallSite.of(HATDialectifyVectorSelectPhase.class,"vstoreSelectPhase");
        //TODO is this side table safe?
        Stream<CodeElement<?, ?>> float4NodesInvolved = OpTk.elements(here,funcOp)
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp)) {
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
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }

        funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputInvokeOp = invokeOp.operands();
                Value v = inputInvokeOp.getFirst();

                if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    List<Value> outputOperandsInvokeOp = context.getValues(inputInvokeOp);
                    int lane = getLane(invokeOp.invokeDescriptor().name());
                    HATVectorViewOp vSelectOp;
                    String name = findNameVector(varLoadOp);
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

        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[AFTER] VSelect Store Transform: " + funcOp.toText());
        }
        return funcOp;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        funcOp = vloadSelectPhase(funcOp);
        funcOp = vstoreSelectPhase(funcOp);
        return funcOp;
    }

}
