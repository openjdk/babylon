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
import hat.Config;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATPrivateVarOp;
import hat.dialect.HATVectorAddOp;
import hat.dialect.HATVectorDivOp;
import hat.dialect.HATVectorLoadOp;
import hat.dialect.HATVectorMulOp;
import hat.dialect.HATVectorSubOp;
import hat.dialect.HATVectorVarLoadOp;
import hat.dialect.HATVectorVarOp;
import hat.dialect.HATVectorViewOp;
import hat.dialect.HATVectorBinaryOp;
import hat.optools.OpTk;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import javax.xml.crypto.dsig.Transform;
import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class HATDialectifyVectorOpPhase implements HATDialect{

    protected final Accelerator accelerator;
    @Override  public Accelerator accelerator(){
        return this.accelerator;
    }
    private final OpView vectorOperation;

    public HATDialectifyVectorOpPhase(Accelerator accelerator, OpView vectorOperation) {
       this.accelerator = accelerator;
        this.vectorOperation = vectorOperation;
    }



    private HATVectorBinaryOp.OpType getBinaryOpType(JavaOp.InvokeOp invokeOp) {
        return switch (invokeOp.invokeDescriptor().name()) {
            case "add" -> HATVectorBinaryOp.OpType.ADD;
            case "sub" -> HATVectorBinaryOp.OpType.SUB;
            case "mul" -> HATVectorBinaryOp.OpType.MUL;
            case "div" -> HATVectorBinaryOp.OpType.DIV;
            default -> throw new RuntimeException("Unknown binary op " + invokeOp.invokeDescriptor().name());
        };
    }

    public enum OpView {
        FLOAT4_LOAD("float4View"),
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div");
        final String methodName;
        OpView(String methodName) {
            this.methodName = methodName;
        }
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
        TypeElement typeElement = invokeOp.resultType();
        boolean isHatVectorType = typeElement.toString().startsWith("hat.buffer.Float");
        return isHatVectorType
                && OpTk.isIfaceBufferMethod(accelerator.lookup, invokeOp)
                && isMethod(invokeOp, vectorOperation.methodName);
    }

    private String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().get(0));
    }

    private String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorViewOp hatVectorViewOp) {
                return hatVectorViewOp.varName();
            }
            return null;
        }
    }

    private boolean findIsSharedOrPrivate(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findIsSharedOrPrivate(varLoadOp.operands().get(0));
    }

    private boolean findIsSharedOrPrivate(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivate(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && (r.op() instanceof HATLocalVarOp || r.op() instanceof HATPrivateVarOp)) {
                return true;
            }
            return false;
        }
    }

    private HATVectorBinaryOp buildVectorBinaryOp(HATVectorBinaryOp.OpType opType, String varName, TypeElement resultType, List<Value> outputOperands) {
        return switch (opType) {
            case ADD -> new HATVectorAddOp(varName, resultType, outputOperands);
            case SUB -> new HATVectorSubOp(varName, resultType, outputOperands);
            case MUL -> new HATVectorMulOp(varName, resultType, outputOperands);
            case DIV -> new HATVectorDivOp(varName, resultType, outputOperands);
        };
    }

    private CoreOp.FuncOp dialectifyVectorLoad(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases())
            IO.println("[BEFORE] Vector Load Ops: " + funcOp.toText());
            Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (isVectorOperation(invokeOp)) {
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }

        var here = OpTk.CallSite.of(HATDialectifyVectorOpPhase.class, "dialectifyVectorLoad" );
        funcOp = OpTk.transform(here, funcOp,(blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                // Don't insert the invoke node
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                boolean isShared = findIsSharedOrPrivate(invokeOp.operands().getFirst());
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = invokeOp.operands();
                        List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                        HATVectorViewOp memoryViewOp = new HATVectorLoadOp(varOp.varName(), varOp.resultType(), invokeOp.resultType(), 4, isShared, outputOperandsVarOp);
                        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
                        memoryViewOp.setLocation(varOp.location());
                        context.mapValue(invokeOp.result(), hatLocalResult);
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                // pass value
                //context.mapValue(varOp.result(), context.getValue(varOp.operands().getFirst()));
                List<Value> inputOperandsVarOp = varOp.operands();
                List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                HATVectorViewOp memoryViewOp = new HATVectorVarOp(varOp.varName(), varOp.resultType(), 4, outputOperandsVarOp);
                Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
                memoryViewOp.setLocation(varOp.location());
                context.mapValue(varOp.result(), hatLocalResult);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                // pass value
                context.mapValue(varLoadOp.result(), context.getValue(varLoadOp.operands().getFirst()));
            }
            return blockBuilder;
        });
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[AFTER] Vector Load Ops: " + funcOp.toText());
        }
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(CoreOp.FuncOp funcOp) {
        Map<JavaOp.InvokeOp, HATVectorBinaryOp.OpType> binaryOperation = new HashMap<>();
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[BEFORE] Vector Binary Ops: " + funcOp.toText());
        }
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (isVectorOperation(invokeOp)) {
                                        HATVectorBinaryOp.OpType binaryOpType = getBinaryOpType(invokeOp);
                                        binaryOperation.put(invokeOp, binaryOpType);
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }

        var here = OpTk.CallSite.of(HATDialectifyVectorOpPhase.class, "dialectifyVectorBinaryOps");
        funcOp = OpTk.transform(here, funcOp, nodesInvolved::contains, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
           // if (!nodesInvolved.contains(op)) {
             //   blockBuilder.op(op);
            //} else
                if (op instanceof JavaOp.InvokeOp invokeOp) {
                Op.Result result = invokeOp.result();
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(inputOperands);
                List<Op.Result> collect = result.uses().stream().toList();
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        HATVectorBinaryOp.OpType binaryOpType = binaryOperation.get(invokeOp);
                        HATVectorViewOp memoryViewOp = buildVectorBinaryOp(binaryOpType, varOp.varName(), invokeOp.resultType(), outputOperands);
                        Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
                        memoryViewOp.setLocation(varOp.location());
                        context.mapValue(invokeOp.result(), hatVectorOpResult);
                        break;
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                List<Value> inputOperandsVarOp = varOp.operands();
                List<Value> outputOperandsVarOp = context.getValues(inputOperandsVarOp);
                HATVectorViewOp memoryViewOp = new HATVectorVarOp(varOp.varName(), varOp.resultType(), 4, outputOperandsVarOp);
                Op.Result hatVectorResult = blockBuilder.op(memoryViewOp);
                memoryViewOp.setLocation(varOp.location());
                context.mapValue(varOp.result(), hatVectorResult);
            }
            return blockBuilder;
        });
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[AFTER] Vector Binary Ops: " + funcOp.toText());
        }
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorBinaryWithContatenationOps(CoreOp.FuncOp funcOp) {
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[BEFORE] Vector Contact Binary Ops: " + funcOp.toText());
        }
        Map<JavaOp.InvokeOp, HATVectorBinaryOp.OpType> binaryOperation = new HashMap<>();
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp)) {
                            List<Value> inputOperandsInvoke = invokeOp.operands();
                            for (Value inputOperand : inputOperandsInvoke) {
                                if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                    HATVectorBinaryOp.OpType binaryOpType = getBinaryOpType(invokeOp);
                                    binaryOperation.put(invokeOp, binaryOpType);
                                    consumer.accept(varLoadOp);
                                    consumer.accept(invokeOp);
                                }
                            }
                        }
                    } else if (codeElement instanceof HATVectorBinaryOp hatVectorBinaryOp) {
                        List<Value> inputOperandsInvoke = hatVectorBinaryOp.operands();
                        for (Value inputOperand : inputOperandsInvoke) {
                            if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                consumer.accept(varLoadOp);
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }
        var here = OpTk.CallSite.of(HATDialectifyVectorOpPhase.class, "dialectifyBinaryWithConcatenation");
        funcOp = OpTk.transform(here, funcOp, (blockBuilder, op) -> {
            CopyContext context = blockBuilder.context();
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                List<Value> inputOperands = invokeOp.operands();
                List<Value> outputOperands = context.getValues(inputOperands);
                HATVectorViewOp memoryViewOp = buildVectorBinaryOp(binaryOperation.get(invokeOp), "null", invokeOp.resultType(), outputOperands);
                Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
                memoryViewOp.setLocation(invokeOp.location());
                context.mapValue(invokeOp.result(), hatVectorOpResult);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                List<Value> inputOperandsVarLoad = varLoadOp.operands();
                List<Value> outputOperandsVarLoad = context.getValues(inputOperandsVarLoad);
                String varLoadName = findNameVector(varLoadOp);
                HATVectorViewOp memoryViewOp = new HATVectorVarLoadOp(varLoadName, varLoadOp.resultType(), outputOperandsVarLoad);
                Op.Result hatVectorResult = blockBuilder.op(memoryViewOp);
                memoryViewOp.setLocation(varLoadOp.location());
                context.mapValue(varLoadOp.result(), hatVectorResult);
            }
            return blockBuilder;
        });
        if (accelerator.backend.config().showCompilationPhases()) {
            IO.println("[AFTER] Vector Binary Ops: " + funcOp.toText());
        }
        return funcOp;
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        if (Objects.requireNonNull(vectorOperation) == OpView.FLOAT4_LOAD) {
            funcOp = dialectifyVectorLoad(funcOp);
        } else {
            // Find binary operations
            funcOp = dialectifyVectorBinaryOps(funcOp);
            funcOp = dialectifyVectorBinaryWithContatenationOps(funcOp);
        }
        return funcOp;
    }
}
