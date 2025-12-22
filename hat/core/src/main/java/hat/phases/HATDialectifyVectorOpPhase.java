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
import hat.dialect.BinaryOpEnum;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATPrivateVarOp;
import hat.dialect.HATVectorAddOp;
import hat.dialect.HATVectorDivOp;
import hat.dialect.HATVectorLoadOp;
import hat.dialect.HATVectorMakeOfOp;
import hat.dialect.HATVectorMulOp;
import hat.dialect.HATVectorOfOp;
import hat.dialect.HATVectorSubOp;
import hat.dialect.HATVectorVarLoadOp;
import hat.dialect.HATVectorVarOp;
import hat.dialect.HATVectorOp;
import hat.dialect.HATVectorBinaryOp;
import hat.optools.OpTk;
import hat.types._V;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.LookupCarrier;
import optkl.OpTkl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static hat.dialect.HATPhaseUtils.VectorMetaData;
import static hat.dialect.HATPhaseUtils.findNameVector;
import static hat.dialect.HATPhaseUtils.findVectorTypeElement;
import static hat.dialect.HATPhaseUtils.getVectorTypeInfo;
import static hat.dialect.HATPhaseUtils.getWitdh;
import static optkl.OpTkl.isAssignable;
import static optkl.OpTkl.isMethod;
import static optkl.OpTkl.transform;

public abstract class HATDialectifyVectorOpPhase implements HATDialect {

    protected final LookupCarrier lookupCarrier;

    @Override
    public LookupCarrier lookupCarrier() {
        return this.lookupCarrier;
    }

    private final OpView vectorOperation;

    public HATDialectifyVectorOpPhase(LookupCarrier lookupCarrier, OpView vectorOperation) {
        this.lookupCarrier = lookupCarrier;
        this.vectorOperation = vectorOperation;
    }



    public enum OpView {
        FLOAT4_LOAD("float4View"),
        FLOAT2_LOAD("float2View"),
        OF("of"),
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div"),
        MAKE_MUTABLE("makeMutable");
        final String methodName;

        OpView(String methodName) {
            this.methodName = methodName;
        }
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
           return (invokeOp.resultType() instanceof JavaType jt
                   && isAssignable(lookupCarrier.lookup(), jt, _V.class)
                   && isMethod(invokeOp, n->n.equals(vectorOperation.methodName))
           );
    }

    private boolean findIsSharedOrPrivate(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findIsSharedOrPrivate(varLoadOp.operands().get(0));
    }

    private boolean findIsSharedOrPrivate(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivate(varLoadOp);
        } else {
            if (v instanceof CoreOp.Result r && (r.op() instanceof HATLocalVarOp || r.op() instanceof HATPrivateVarOp)) {
                return true;
            }
            return false;
        }
    }

    private HATVectorBinaryOp buildVectorBinaryOp(BinaryOpEnum opType, String varName, TypeElement resultType, TypeElement vectorElementType, int witdh, List<Value> outputOperands) {
        return switch (opType) {
            case ADD -> new HATVectorAddOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case SUB -> new HATVectorSubOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case MUL -> new HATVectorMulOp(varName, resultType, vectorElementType, witdh, outputOperands);
            case DIV -> new HATVectorDivOp(varName, resultType, vectorElementType, witdh, outputOperands);
        };
    }

    private void insertVectorLoadOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp, CoreOp.VarOp varOp, boolean isShared) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData metaData = getVectorTypeInfo(invokeOp);
        HATVectorOp memoryViewOp = new HATVectorLoadOp(varOp.varName(), varOp.resultType(), metaData.vectorTypeElement(), metaData.lanes(), isShared, outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    private void inertVectorVarOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = varOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData vmd = vectorMetaData.get(varOp);
        HATVectorOp memoryViewOp = new HATVectorVarOp(varOp.varName(), varOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(varOp.result(), hatLocalResult);
    }

    public void insertBinaryOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, JavaOp.InvokeOp invokeOp, Map<Op, VectorMetaData> vectorMetaData, Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(inputOperands);
        BinaryOpEnum binaryOpType = binaryOperation.get(invokeOp);
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorOp memoryViewOp = buildVectorBinaryOp(binaryOpType, varOp.varName(), invokeOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperands);
        Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatVectorOpResult);
    }

    private void insertVectorVarLoadOp(Block.Builder blockBuilder, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        List<Value> inputOperandsVarLoad = varLoadOp.operands();
        List<Value> outputOperandsVarLoad = blockBuilder.context().getValues(inputOperandsVarLoad);
        String varLoadName = findNameVector(varLoadOp);
        int lanes = getWitdh(varLoadOp);
        TypeElement vectorElementType = findVectorTypeElement(varLoadOp);
        HATVectorOp memoryViewOp = new HATVectorVarLoadOp(varLoadName, varLoadOp.resultType(), vectorElementType, lanes, outputOperandsVarLoad);
        Op.Result hatVectorResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(varLoadOp.location());
        blockBuilder.context().mapValue(varLoadOp.result(), hatVectorResult);
    }

    public void insertVectorBinaryOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp, Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation) {
        List<Value> inputOperands = invokeOp.operands();
        List<Value> outputOperands = blockBuilder.context().getValues(inputOperands);
        VectorMetaData vectorMetaData = getVectorTypeInfo(invokeOp);
        HATVectorOp memoryViewOp = buildVectorBinaryOp(binaryOperation.get(invokeOp), "null", invokeOp.resultType(), vectorMetaData.vectorTypeElement(), vectorMetaData.lanes(), outputOperands);
        Op.Result hatVectorOpResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatVectorOpResult);
    }

    public void insertVectorOfOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp, Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorOfOp memoryViewOp = new HATVectorOfOp(invokeOp.resultType(), vmd.vectorTypeElement(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(memoryViewOp);
        memoryViewOp.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    public void insertVectorMakeOfOp(Block.Builder blockBuilder, JavaOp.InvokeOp invokeOp, Map<Op, VectorMetaData> vectorMetaData) {
        List<Value> inputOperandsVarOp = invokeOp.operands();
        List<Value> outputOperandsVarOp = blockBuilder.context().getValues(inputOperandsVarOp);
        String varName = findNameVector(invokeOp.operands().getFirst());
        VectorMetaData vmd = vectorMetaData.get(invokeOp);
        HATVectorMakeOfOp makeOf = new HATVectorMakeOfOp(varName, invokeOp.resultType(), vmd.lanes(), outputOperandsVarOp);
        Op.Result hatLocalResult = blockBuilder.op(makeOf);
        makeOf.setLocation(invokeOp.location());
        blockBuilder.context().mapValue(invokeOp.result(), hatLocalResult);
    }

    private CoreOp.FuncOp dialectifyVectorLoad(CoreOp.FuncOp funcOp) {
        var here = OpTkl.CallSite.of(this.getClass(), "dialectifyVectorLoad");
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        before(here, funcOp);
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (isVectorOperation(invokeOp)) {
                                        // Associate both ops to the vectorTypeInfo for easy
                                        // access to type and lanes
                                        VectorMetaData vectorTypeInfo = getVectorTypeInfo(invokeOp);
                                        vectorMetaData.put(invokeOp, vectorTypeInfo);
                                        vectorMetaData.put(varOp, vectorTypeInfo);
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
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
                // Don't insert the invoke node
                Op.Result result = invokeOp.result();
                List<Op.Result> collect = result.uses().stream().toList();
                boolean isShared = findIsSharedOrPrivate(invokeOp.operands().getFirst());
                for (Op.Result r : collect) {
                    if (r.op() instanceof CoreOp.VarOp varOp) {
                        insertVectorLoadOp(blockBuilder, invokeOp, varOp, isShared);
                    }
                }
            } else if (op instanceof CoreOp.VarOp varOp) {
                inertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorOf(CoreOp.FuncOp funcOp) {
        var here = OpTkl.CallSite.of(this.getClass(), "dialectifyVectorOf");
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        before(here, funcOp);
        Stream<CodeElement<?, ?>> vectorNodes = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp)) {
                            consumer.accept(invokeOp);
                            Set<Op.Result> uses = invokeOp.result().uses();
                            for (Op.Result result : uses) {
                                if (result.op() instanceof CoreOp.VarOp varOp) {
                                    consumer.accept(varOp);
                                    VectorMetaData vectorTypeInfo = getVectorTypeInfo(invokeOp);
                                    vectorMetaData.put(invokeOp, vectorTypeInfo);
                                    vectorMetaData.put(varOp, vectorTypeInfo);
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodes.collect(Collectors.toSet());

        funcOp = transform(here, funcOp, (blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                insertVectorOfOp(blockBuilder, invokeOp, vectorMetaData);
            } else if (op instanceof CoreOp.VarOp varOp) {
                inertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(CoreOp.FuncOp funcOp) {
        var here = OpTkl.CallSite.of(this.getClass(), "dialectifyVectorBinaryOps");
        before(here, funcOp);
        Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation = new HashMap<>();
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();

        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof CoreOp.VarOp varOp) {
                        List<Value> inputOperandsVarOp = varOp.operands();
                        for (Value inputOperand : inputOperandsVarOp) {
                            if (inputOperand instanceof Op.Result result) {
                                if (result.op() instanceof JavaOp.InvokeOp invokeOp) {
                                    if (isVectorOperation(invokeOp)) {
                                        BinaryOpEnum binaryOpType = BinaryOpEnum.of(invokeOp);
                                        binaryOperation.put(invokeOp, binaryOpType);
                                        VectorMetaData vectorTypeInfo = getVectorTypeInfo(invokeOp);
                                        vectorMetaData.put(invokeOp, vectorTypeInfo);
                                        vectorMetaData.put(varOp, vectorTypeInfo);
                                        consumer.accept(invokeOp);
                                        consumer.accept(varOp);
                                    }
                                }
                            }
                        }
                    }
                });

        Set<CodeElement<?, ?>> nodesInvolved = float4NodesInvolved.collect(Collectors.toSet());

        funcOp = transform(here, funcOp, nodesInvolved::contains, (blockBuilder, op) -> {
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
                inertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyMutableOf(CoreOp.FuncOp funcOp) {
        var here = OpTkl.CallSite.of(this.getClass(), "dialectifyMutableOf");
        before(here, funcOp);
        Map<Op, VectorMetaData> vectorMetaData = new HashMap<>();
        Stream<CodeElement<?, ?>> float4NodesInvolved = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp)) {
                            consumer.accept(invokeOp);
                            VectorMetaData vectorTypeInfo = getVectorTypeInfo(invokeOp);
                            vectorMetaData.put(invokeOp, vectorTypeInfo);
                            Set<Op.Result> uses = invokeOp.result().uses();
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

        funcOp = transform(here, funcOp, (blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                insertVectorMakeOfOp(blockBuilder, invokeOp, vectorMetaData);
            } else if (op instanceof CoreOp.VarOp varOp) {
                inertVectorVarOp(blockBuilder, varOp, vectorMetaData);
            }
            return blockBuilder;
        });
        after(here, funcOp);
        return funcOp;
    }

    private CoreOp.FuncOp dialectifyVectorBinaryWithConcatenationOps(CoreOp.FuncOp funcOp) {
        var here = OpTkl.CallSite.of(this.getClass(), "dialectifyBinaryWithConcatenation");
        before(here, funcOp);
        Map<JavaOp.InvokeOp, BinaryOpEnum> binaryOperation = new HashMap<>();
        Stream<CodeElement<?, ?>> vectorNodes = funcOp.elements()
                .mapMulti((codeElement, consumer) -> {
                    if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                        if (isVectorOperation(invokeOp)) {
                            List<Value> inputOperandsInvoke = invokeOp.operands();
                            for (Value inputOperand : inputOperandsInvoke) {
                                if (inputOperand instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                                    BinaryOpEnum binaryOpType = BinaryOpEnum.of(invokeOp);
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

        Set<CodeElement<?, ?>> nodesInvolved = vectorNodes.collect(Collectors.toSet());
        if (nodesInvolved.isEmpty()) {
            return funcOp;
        }
        funcOp = transform(here, funcOp, (blockBuilder, op) -> {
            if (!nodesInvolved.contains(op)) {
                blockBuilder.op(op);
            } else if (op instanceof JavaOp.InvokeOp invokeOp) {
                insertVectorBinaryOp(blockBuilder, invokeOp, binaryOperation);
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                insertVectorVarLoadOp(blockBuilder, varLoadOp);
            }
            return blockBuilder;
        });
        after(here, funcOp);
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

    public static class AddPhase extends HATDialectifyVectorOpPhase {

        public AddPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.ADD);
        }
    }

    public static class DivPhase extends HATDialectifyVectorOpPhase {

        public DivPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.DIV);
        }
    }

    public static class MakeMutable extends HATDialectifyVectorOpPhase {

        public MakeMutable(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.MAKE_MUTABLE);
        }
    }

    public static class Float4LoadPhase extends HATDialectifyVectorOpPhase {

        public Float4LoadPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.FLOAT4_LOAD);
        }
    }

    public static class Float2LoadPhase extends HATDialectifyVectorOpPhase {

        public Float2LoadPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.FLOAT2_LOAD);
        }
    }

    public static class Float4OfPhase extends HATDialectifyVectorOpPhase {

        public Float4OfPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.OF);
        }
    }

    public static class MulPhase extends HATDialectifyVectorOpPhase {

        public MulPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.MUL);
        }
    }

    public static class SubPhase extends HATDialectifyVectorOpPhase {

        public SubPhase(LookupCarrier lookupCarrier) {
            super(lookupCarrier, OpView.SUB);
        }
    }
}
