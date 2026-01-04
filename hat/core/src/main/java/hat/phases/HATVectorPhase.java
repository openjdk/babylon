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
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.Invoke;
import optkl.Trxfmr;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static hat.dialect.HATPhaseUtils.VectorMetaData;
import static hat.dialect.HATPhaseUtils.getVectorTypeInfo;
import static optkl.Invoke.invokeOpHelper;

public abstract sealed class HATVectorPhase implements HATPhase
        permits HATVectorPhase.AddPhase, HATVectorPhase.DivPhase, HATVectorPhase.Float2LoadPhase, HATVectorPhase.Float4LoadPhase
      , HATVectorPhase.MulPhase, HATVectorPhase.MakeMutable, HATVectorPhase.SubPhase, HATVectorPhase.Float4OfPhase{
    private final KernelCallGraph kernelCallGraph;
@Override public KernelCallGraph kernelCallGraph(){
    return kernelCallGraph;
}


    // recursive
    private static TypeElement findVectorTypeElement(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorTypeElement(varLoadOp.operands().getFirst());
    }
    // recursive
    private static TypeElement findVectorTypeElement(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorTypeElement(varLoadOp); // recurse
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorElementType();
            }
            return null;
        }
    }

    //recursive
    public static int getWidth(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getWidth(varLoadOp.operands().getFirst());
    }
    //recursive
    private static int getWidth(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getWidth(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorN();
            }
            return -1;
        }
    }
    //recursive
    private boolean findIsSharedOrPrivate(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findIsSharedOrPrivate(varLoadOp.operands().getFirst());
    }

    //recursive
    private boolean findIsSharedOrPrivate(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findIsSharedOrPrivate(varLoadOp); //recurse
            case HATMemoryVarOp.HATLocalVarOp _, HATMemoryVarOp.HATPrivateVarOp _ -> true;
            default -> false;
        };
    }

    // recursive
    public static String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().getFirst());
    }

    // recursive
    public static String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.varName();
            }
            return null;
        }
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

    private HATVectorOp.HATVectorBinaryOp buildVectorBinaryOp(BinaryOpEnum opType, String varName, TypeElement resultType,
                                                              TypeElement vectorElementType, int witdh, List<Value> outputOperands) {
        return switch (opType) {
            case ADD -> new HATVectorOp.HATVectorBinaryOp.HATVectorAddOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case SUB -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case MUL -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case DIV -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName, resultType, vectorElementType, witdh, outputOperands);
        };
    }

    private void insertVectorLoadOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp, CoreOp.VarOp varOp, boolean isShared) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData metaData = getVectorTypeInfo(lookup(),invokeOp);
        HATVectorOp memoryViewOp = new HATVectorOp.HATVectorLoadOp(varOp.varName(), varOp.resultType(), metaData.vectorTypeElement(), metaData.lanes(), isShared, outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    private void insertVectorVarOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = varOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData vmd = vectorMetaData.get(varOp);
        HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarOp(varOp.varName(), varOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(varOp.result(), hatLocalResult);
    }

    public void insertBinaryOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp,
                               Map<Op, VectorMetaData> vectorMetaData, Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(inputOperands);
        BinaryOpEnum binaryOpType = binaryOperation.get(invokeOp);
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorOp memoryViewOp = buildVectorBinaryOp(binaryOpType, varOp.varName(),
                invokeOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperands);
        Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatVectorOpResult);
    }

    private void insertVectorVarLoadOp(Block.Builder blockBuilder, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        List<Value> inputOperandsVarLoad = varLoadOp.operands();
        List<Value> outputOperandsVarLoad = blockBuilder.context().getValues(inputOperandsVarLoad);
        String varLoadName = findNameVector(varLoadOp);
        int lanes = getWidth(varLoadOp);
        TypeElement vectorElementType = findVectorTypeElement(varLoadOp);
        HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarLoadOp(varLoadName, varLoadOp.resultType(), vectorElementType, lanes, outputOperandsVarLoad);
        Op.Result hatVectorResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varLoadOp.location());
        blockBuilder.context().mapValue(varLoadOp.result(), hatVectorResult);
    }

    public void insertVectorBinaryOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp,
                                     Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(inputOperands);
        VectorMetaData vectorMetaData = getVectorTypeInfo(lookup(),invokeOp);
        HATVectorOp memoryViewOp = buildVectorBinaryOp(binaryOperation.get(invokeOp), "null", invokeOp.resultType(), vectorMetaData.vectorTypeElement(), vectorMetaData.lanes(), outputOperands);
        Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatVectorOpResult);
    }

    public void insertVectorOfOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp,
                                 Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorOp.HATVectorOfOp memoryViewOp = new HATVectorOp.HATVectorOfOp(invokeOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    public void insertVectorMakeOfOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp,
                                     Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        String varName = findNameVector(invokeOp.operands().getFirst());
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorOp.HATVectorMakeOfOp makeOf = new HATVectorOp.HATVectorMakeOfOp(varName, invokeOp.resultType(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(makeOf);
        makeOf.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    private CoreOp.FuncOp dialectifyVectorLoad(CoreOp.FuncOp funcOp) {
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (invokeOpHelper(lookup(),result.op()) instanceof Invoke invoke) {
                                    if (invoke.returns(_V.class) && invoke.named(vectorOperation.methodName)){
                                          //  isVectorOperation(invokeOpHelper(lookup(),invokeOp))) {
                                        // Associate both ops to the vectorTypeInfo for easy
                                        // access to type and lanes
                                        VectorMetaData vectorTypeInfo = getVectorTypeInfo(lookup(),invoke.op());
                                        vectorMetaData.put(invoke.op(), vectorTypeInfo);
                                        vectorMetaData.put(varOp, vectorTypeInfo);
                                        consumer.accept(invoke.op());
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());

        return new Trxfmr(funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                boolean isShared = findIsSharedOrPrivate(invokeOp.operands().getFirst());
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        insertVectorLoadOp(blockBuilder, invokeOp, varOp, isShared);
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                insertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyVectorOf(CoreOp.FuncOp funcOp) {
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        Stream<CodeElement<?, ?>> vectorNodes = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke
                         &&invoke.returns(_V.class) && invoke.named(vectorOperation.methodName) ) {
                            consumer.accept(invoke.op());
                            Set<Op.Result> uses = invoke.op().result().uses();
                            for (Op.Result result : uses) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    VectorMetaData vectorTypeInfo = getVectorTypeInfo(lookup(),invoke.op());
                                    vectorMetaData.put(invoke.op(), vectorTypeInfo);
                                    vectorMetaData.put(varOp, vectorTypeInfo);
                                }
                            }
                        }

                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodes.collect(Collectors.toSet());

        return new Trxfmr(funcOp).transform(_->true, (blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                insertVectorOfOp(blockBuilder, invokeOp, vectorMetaData);
            } else if (op instanceof CoreOp.VarOp varOp) {
                insertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(CoreOp.FuncOp funcOp) {

        Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation = new HashMap<>();
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();

        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (invokeOpHelper(lookup(),result.op()) instanceof Invoke invoke ) {
                                    if (invoke.returns(_V.class) && invoke.named(vectorOperation.methodName)) {
                                        BinaryOpEnum binaryOpType = BinaryOpEnum.of(invoke.op());
                                        binaryOperation.put(invoke.op(), binaryOpType);
                                        VectorMetaData vectorTypeInfo = getVectorTypeInfo(lookup(),invoke.op());
                                        vectorMetaData.put(invoke.op(), vectorTypeInfo);
                                        vectorMetaData.put(varOp, vectorTypeInfo);
                                        consumer.accept(invoke.op());
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());

        return new Trxfmr(funcOp).transform( nodesInvolved::contains, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        insertBinaryOp(blockBuilder, varOp, invokeOp, vectorMetaData, binaryOperation);
                        break;
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                insertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyMutableOf(CoreOp.FuncOp funcOp) {
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke) {
                        if (invoke.returns(_V.class) && invoke.named(vectorOperation.methodName)) {
                            consumer.accept(invoke.op());
                            VectorMetaData vectorTypeInfo = getVectorTypeInfo(lookup(),invoke.op());
                            vectorMetaData.put(invoke.op(), vectorTypeInfo);
                            Set<Op.Result> uses = invoke.op().result().uses();
                            for (Op.Result result : uses) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    vectorMetaData.put(varOp, vectorTypeInfo);
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());

        funcOp = new Trxfmr(funcOp).transform(_->true, (blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                insertVectorMakeOfOp(blockBuilder, invokeOp, vectorMetaData);
            } else if (op instanceof CoreOp.VarOp varOp) {
                insertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        }).funcOp();
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorBinaryWithConcatenationOps(CoreOp.FuncOp funcOp) {
        Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation = new HashMap<>();
        Stream<CodeElement<?, ?>> vectorNodes = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (invokeOpHelper(lookup(),codeElement) instanceof Invoke invoke) {
                        if (invoke.returns(_V.class) && invoke.named(vectorOperation.methodName)) {
                            List<Value> inputOperandsInvoke = invoke.op().operands();
                            for (Value inputOperand : inputOperandsInvoke) {
                                if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                    BinaryOpEnum binaryOpType = BinaryOpEnum.of(invoke.op());
                                    binaryOperation.put(invoke.op(), binaryOpType);
                                    consumer.accept(varLoadOp);
                                    consumer.accept(invoke.op());
                                }
                            }
                        }
                    } else if (codeElement instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {
                        List<Value> inputOperandsInvoke = hatVectorBinaryOp.operands();
                        for (Value inputOperand : inputOperandsInvoke) {
                            if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                consumer.accept(varLoadOp);
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodes.collect(Collectors.toSet());
        if (!nodesInvolved.isEmpty()) {
            funcOp = new Trxfmr(funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
                 if (op instanceof JavaOp.InvokeOp invokeOp) {
                    insertVectorBinaryOp(blockBuilder, invokeOp, binaryOperation);
                } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    insertVectorVarLoadOp(blockBuilder, varLoadOp);
                }
                return blockBuilder;
            }).funcOp();
        }
        return funcOp;
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
