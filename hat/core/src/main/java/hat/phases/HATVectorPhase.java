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
import hat.dialect.BinaryOpEnum;
import hat.dialect.HATVectorOp;
import hat.types.Vector;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public abstract sealed class HATVectorPhase implements HATPhase
        permits HATVectorPhase.AddPhase, HATVectorPhase.DivPhase, HATVectorPhase.Float2LoadPhase, HATVectorPhase.Float4LoadPhase
      , HATVectorPhase.MulPhase, HATVectorPhase.MakeMutable, HATVectorPhase.SubPhase, HATVectorPhase.Float4OfPhase{
    private final KernelCallGraph kernelCallGraph;
    @Override public KernelCallGraph kernelCallGraph(){
        return kernelCallGraph;
    }


    public enum VectorOperation {
        FLOAT4_LOAD("float4View"),
        FLOAT2_LOAD("float2View"),
        OF("of"),
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div"),
        MAKE_MUTABLE("makeMutable");
        final String methodName;

        VectorOperation(String methodName) {
            this.methodName = methodName;
        }
    }

    private final VectorOperation vectorOperation;

    public HATVectorPhase(KernelCallGraph kernelCallGraph, VectorOperation vectorOperation) {
        this.kernelCallGraph = kernelCallGraph;
        this.vectorOperation = vectorOperation;
    }


    private void addVectorVarOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, Vector.Shape vectorShape) {
        HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarOp(
                varOp.varName(),
                varOp.resultType(),
                vectorShape,
                blockBuilder.context().getValues(varOp.operands())
        );
        blockBuilder.context().mapValue(varOp.result(), blockBuilder.op(copyLocation(varOp,memoryViewOp)));
    }

    private CoreOp.FuncOp dialectifyVectorLoad(CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Map<JavaOp.InvokeOp, CoreOp.VarOp> invokeToVar = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup(),funcOp).forEach(v ->{
             if (v.firstOperandAsInvoke() instanceof Invoke i && i.returns(Vector.class) && i.named(vectorOperation.methodName)){
                 Vector.Shape vectorShape = HATPhaseUtils.getVectorShapeFromInvokeReturnType(lookup(), i.op());
                 vectorShapeMap.put(i.op(), vectorShape);
                 vectorShapeMap.put(v.op(), vectorShape);
                 invokeToVar.put(i.op(),v.op());
             }
        });

        return Trxfmr.of(this,funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (Invoke.invoke(lookup(),op) instanceof Invoke invoke) {
                var varOp = invokeToVar.get(invoke.op());
                Vector.Shape shape = HATPhaseUtils.getVectorShapeFromInvokeReturnType(lookup(),invoke.op());
                HATVectorOp memoryViewOp =  HATPhaseUtils.isSharedOrPrivate(invoke.resultFromFirstOperandOrNull())
                        ? new HATVectorOp.HATVectorLoadOp.HATSharedVectorLoadOp(
                                varOp.varName(),
                                varOp.resultType(),
                                shape,
                                blockBuilder.context().getValues(invoke.op().operands()))
                        : new HATVectorOp.HATVectorLoadOp.HATPrivateVectorLoadOp(
                                varOp.varName(),
                                varOp.resultType(),
                                shape,
                                blockBuilder.context().getValues(invoke.op().operands())
                );
                blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(copyLocation(varOp,memoryViewOp)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }


    private HATVectorOp.HATVectorBinaryOp buildVectorBinaryOp(String varName, BinaryOpEnum opType,
                                                              Vector.Shape vectorShape, List<Value> outputOperands) {
        return switch (opType) {
            case ADD -> new HATVectorOp.HATVectorBinaryOp.HATVectorAddOp(varName, vectorShape, outputOperands);
            case SUB -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName,  vectorShape, outputOperands);
            case MUL -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName,  vectorShape, outputOperands);
            case DIV -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName,  vectorShape, outputOperands);
        };
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Map<JavaOp.InvokeOp, CoreOp.VarOp> invokeToVar = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup(),funcOp).forEach(v -> {
            if (v.firstOperandAsInvoke() instanceof Invoke i && i.named(vectorOperation.methodName) && i.returns(Vector.class)) {
                Vector.Shape vectorShape = HATPhaseUtils.getVectorShapeFromInvokeReturnType(lookup(), i.op());
                vectorShapeMap.put(i.op(), vectorShape);
                vectorShapeMap.put(v.op(), vectorShape);
                invokeToVar.put(i.op(), v.op());
            }
        });

        return Trxfmr.of(this,funcOp).transform( vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var varOp = invokeToVar.get(invokeOp);
                HATVectorOp memoryViewOp =  buildVectorBinaryOp(
                        varOp.varName(),
                        BinaryOpEnum.of(invokeOp),
                        vectorShapeMap.get(invokeOp),
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeToVar.get(invokeOp),memoryViewOp)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private  Map<Op, Vector.Shape> getVectorMetaDataMap(CoreOp.FuncOp funcOp){
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Invoke.stream(lookup(),funcOp).
                filter(i -> i.returns(Vector.class) && i.named(vectorOperation.methodName) && i.onlyUse() instanceof CoreOp.VarOp)
                .forEach(i -> {
                    Vector.Shape vectorShape = HATPhaseUtils.getVectorShapeFromInvokeReturnType(lookup(), i.op());
                    vectorShapeMap.put(i.op(), vectorShape);
                    vectorShapeMap.put(i.onlyUse(), vectorShape);
                });
        return vectorShapeMap;
    }


    private CoreOp.FuncOp dialectifyVectorOf(CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = getVectorMetaDataMap(funcOp);

        return Trxfmr.of(this,funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var vectorShape = vectorShapeMap.get(invokeOp);
                HATVectorOp.HATVectorOfOp memoryViewOp = new HATVectorOp.HATVectorOfOp(
                        invokeOp.resultType(),
                        vectorShape,
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp,memoryViewOp)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyMutableOf(CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = getVectorMetaDataMap(funcOp);
        return Trxfmr.of(this,funcOp).transform(ce->vectorShapeMap.containsKey(ce), (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var vectorShape = vectorShapeMap.get(invokeOp);
                HATVectorOp.HATVectorMakeOfOp makeOf = new HATVectorOp.HATVectorMakeOfOp(
                        HATPhaseUtils.findVectorVarNameOrNull(invokeOp.operands().getFirst()),
                        invokeOp.resultType(),
                        vectorShape.lanes(),
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp,makeOf)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }



    private CoreOp.FuncOp dialectifyVectorBinaryWithConcatenationOps(CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
        funcOp.elements().forEach(codeElement->{
                    if (invoke(lookup(),codeElement) instanceof Invoke invoke && invoke.returns(Vector.class) && invoke.named(vectorOperation.methodName)) {
                            invoke.op().operands().stream()// this can't be replaced with findFirst
                                    .filter(operand->operand instanceof Op.Result && ((Op.Result) operand).op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                                    .map(operand->(CoreOp.VarAccessOp.VarLoadOp) ((Op.Result)operand).op())
                                    .forEach(varLoadOp -> {
                                        nodesInvolved.add(varLoadOp);
                                        nodesInvolved.add(invoke.op());
                                    });
                    } else if (codeElement instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {
                        hatVectorBinaryOp.operands().stream()
                                .filter(operand->operand instanceof Op.Result && ((Op.Result) operand).op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                                .map(operand->(CoreOp.VarAccessOp.VarLoadOp) ((Op.Result)operand).op())
                                .forEach(nodesInvolved::add);
                    }
                });


           return Trxfmr.of(this,funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
                 if (op instanceof JavaOp.InvokeOp invokeOp) {
                     HATVectorOp memoryViewOp = buildVectorBinaryOp(
                             HATPhaseUtils.findVectorVarNameOrNull(invokeOp.operands().getFirst()),
                             BinaryOpEnum.of(invokeOp),
                             HATPhaseUtils.getVectorShapeFromInvokeReturnType(lookup(),invokeOp),
                             blockBuilder.context().getValues(invokeOp.operands())
                     );
                     blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp,memoryViewOp)));
                } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                     HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarLoadOp(
                             HATPhaseUtils.findVectorVarNameOrNull(varLoadOp),
                             varLoadOp.resultType(),
                             HATPhaseUtils.getVectorShapeOrNullFromVarLoad(varLoadOp),
                             blockBuilder.context().getValues(varLoadOp.operands())
                     );
                     blockBuilder.context().mapValue(varLoadOp.result(), blockBuilder.op(copyLocation(varLoadOp,memoryViewOp)));
                }
                return blockBuilder;
            }).funcOp();
    }

    @Override
    public CoreOp.FuncOp apply(CoreOp.FuncOp funcOp) {
        switch (Objects.requireNonNull(vectorOperation)) {
            case FLOAT4_LOAD -> funcOp = dialectifyVectorLoad(funcOp);
            case FLOAT2_LOAD ->  funcOp = dialectifyVectorLoad(funcOp);
            case OF -> funcOp = dialectifyVectorOf(funcOp);
            case MAKE_MUTABLE -> funcOp = dialectifyMutableOf(funcOp);
            default -> {
                // Find binary operations
                funcOp = dialectifyVectorBinaryOps(funcOp);
                funcOp = dialectifyVectorBinaryWithConcatenationOps(funcOp);
            }
        }
        return funcOp;
    }

    public static final class AddPhase extends HATVectorPhase {
        public AddPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.ADD);
        }
    }

    public static final class DivPhase extends HATVectorPhase {

        public DivPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.DIV);
        }
    }

    public static final class MakeMutable extends HATVectorPhase {

        public MakeMutable(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.MAKE_MUTABLE);
        }
    }

    public static final class Float4LoadPhase extends HATVectorPhase {

        public Float4LoadPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.FLOAT4_LOAD);
        }
    }

    public static final class Float2LoadPhase extends HATVectorPhase {

        public Float2LoadPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.FLOAT2_LOAD);
        }
    }

    public static final class Float4OfPhase extends HATVectorPhase {

        public Float4OfPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.OF);
        }
    }

    public static final class MulPhase extends HATVectorPhase {

        public MulPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.MUL);
        }
    }

    public static final class SubPhase extends HATVectorPhase {

        public SubPhase(KernelCallGraph kernelCallGraph) {
            super(kernelCallGraph, VectorOperation.SUB);
        }
    }
}
