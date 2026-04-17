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

import hat.dialect.BinaryOpEnum;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import optkl.IfaceValue.Vector;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.copyLocation;

public abstract sealed class HATVectorPhase implements HATPhase
        permits HATVectorPhase.AddPhase, HATVectorPhase.DivPhase, HATVectorPhase.Float2LoadPhase, HATVectorPhase.Float4LoadPhase
        , HATVectorPhase.MulPhase, HATVectorPhase.MakeMutable, HATVectorPhase.SubPhase, HATVectorPhase.Float4OfPhase {



    // recursive
    public static String findVectorVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorVarNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    public static String findVectorVarNameOrNull(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorVarNameOrNull(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.varName();
            }
            return null;
        }
    }
    //recursive
    public static boolean isSharedOrPrivate(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isSharedOrPrivate(varLoadOp.operands().getFirst());
    }

    //recursive
    public static boolean isSharedOrPrivate(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isSharedOrPrivate(varLoadOp); //recurse
            case HATMemoryVarOp.HATLocalVarOp _, HATMemoryVarOp.HATPrivateVarOp _ -> true;
            default -> false;
        };
    }


    //recursive
    public static Vector.Shape getVectorShapeOrNullFromVarLoad(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getVectorShapeOrNull(varLoadOp.operands().getFirst());
    }
    private static Vector.Shape getVectorShapeOrNull(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorShapeOrNullFromVarLoad(varLoadOp);
        } else if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
            return hatVectorOp.vectorShape();
        }
        return null;
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

    public HATVectorPhase( VectorOperation vectorOperation) {
        this.vectorOperation = vectorOperation;
    }


    private void addVectorVarOp(Block.Builder blockBuilder, CoreOp.VarOp varOp, Vector.Shape vectorShape) {
        HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarOp(
                varOp.varName(),
                varOp.resultType(),
                vectorShape,
                blockBuilder.context().getValues(varOp.operands())
        );
        blockBuilder.context().mapValue(varOp.result(), blockBuilder.op(copyLocation(varOp, memoryViewOp)));
    }

    private CoreOp.FuncOp dialectifyVectorLoad(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Map<JavaOp.InvokeOp, CoreOp.VarOp> invokeToVar = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup, funcOp).forEach(variable -> {
            if (variable.firstOperandAsInvoke() instanceof Invoke invoke
                    && invoke.returns(Vector.class)
                    && invoke.named(vectorOperation.methodName)) {
                Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
                vectorShapeMap.put(invoke.op(), vectorShape);
                vectorShapeMap.put(variable.op(), vectorShape);
                invokeToVar.put(invoke.op(), variable.op());
            }
        });

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (Invoke.invoke(lookup, op) instanceof Invoke invoke) {
                var varOp = invokeToVar.get(invoke.op());
                Vector.Shape shape = getVectorShape(invoke.lookup(), invoke.returnType());
                HATVectorOp memoryViewOp = isSharedOrPrivate(invoke.resultFromFirstOperandOrNull())
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
                blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(copyLocation(varOp, memoryViewOp)));
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
            case SUB -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName, vectorShape, outputOperands);
            case MUL -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName, vectorShape, outputOperands);
            case DIV -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName, vectorShape, outputOperands);
        };
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Map<JavaOp.InvokeOp, CoreOp.VarOp> invokeToVar = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup, funcOp).forEach(variable -> {
            if (variable.firstOperandAsInvoke() instanceof Invoke invoke
                    && invoke.named(vectorOperation.methodName)
                    && invoke.returns(Vector.class)) {
                Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
                vectorShapeMap.put(invoke.op(), vectorShape);
                vectorShapeMap.put(variable.op(), vectorShape);
                invokeToVar.put(invoke.op(), variable.op());
            }
        });

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var varOp = invokeToVar.get(invokeOp);
                HATVectorOp memoryViewOp = buildVectorBinaryOp(
                        varOp.varName(),
                        BinaryOpEnum.of(invokeOp),
                        vectorShapeMap.get(invokeOp),
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeToVar.get(invokeOp), memoryViewOp)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private Map<Op, Vector.Shape> getVectorShapeMap(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        Invoke.stream(lookup, funcOp).
                filter(i -> i.returns(Vector.class)
                        && i.named(vectorOperation.methodName)
                        && i.opFromOnlyUseOrNull() instanceof CoreOp.VarOp)
                .forEach(i -> {
                    Vector.Shape vectorShape = getVectorShape(i.lookup(), i.returnType());
                    vectorShapeMap.put(i.op(), vectorShape);
                    vectorShapeMap.put(i.opFromOnlyUseOrNull(), vectorShape);
                });
        return vectorShapeMap;
    }


    private CoreOp.FuncOp dialectifyVectorOf(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = getVectorShapeMap(lookup,funcOp);

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var vectorShape = vectorShapeMap.get(invokeOp);
                HATVectorOp.HATVectorOfOp memoryViewOp = new HATVectorOp.HATVectorOfOp(
                        invokeOp.resultType(),
                        vectorShape,
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp, memoryViewOp)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }

    private CoreOp.FuncOp dialectifyMutableOf(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Map<Op, Vector.Shape> vectorShapeMap = getVectorShapeMap(lookup,funcOp);
        return Trxfmr.of(lookup, funcOp).transform(ce -> vectorShapeMap.containsKey(ce), (blockBuilder, op) -> {
            if (op instanceof JavaOp.InvokeOp invokeOp) {
                var vectorShape = vectorShapeMap.get(invokeOp);
                HATVectorOp.HATVectorMakeOfOp makeOf = new HATVectorOp.HATVectorMakeOfOp(
                        findVectorVarNameOrNull(invokeOp.operands().getFirst()),
                        invokeOp.resultType(),
                        vectorShape.lanes(),
                        blockBuilder.context().getValues(invokeOp.operands())
                );
                blockBuilder.context().mapValue(invokeOp.result(), blockBuilder.op(copyLocation(invokeOp, makeOf)));
            } else if (op instanceof CoreOp.VarOp varOp) {
                addVectorVarOp(blockBuilder, varOp, vectorShapeMap.get(varOp));
            }
            return blockBuilder;
        }).funcOp();
    }


    private CoreOp.FuncOp dialectifyVectorBinaryWithConcatenationOps(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
        funcOp.elements().forEach(codeElement -> {
            if (invoke(lookup, codeElement) instanceof Invoke invoke
                    && invoke.returns(Vector.class) && invoke.named(vectorOperation.methodName)) {
                invoke.op().operands().stream()// this can't be replaced with findFirst
                        .filter(operand -> operand instanceof Op.Result && ((Op.Result) operand).op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                        .map(operand -> (CoreOp.VarAccessOp.VarLoadOp) ((Op.Result) operand).op())
                        .forEach(varLoadOp -> {
                            nodesInvolved.add(varLoadOp);
                            nodesInvolved.add(invoke.op());
                        });
            } else if (codeElement instanceof HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {
                hatVectorBinaryOp.operands().stream()
                        .filter(operand -> operand instanceof Op.Result && ((Op.Result) operand).op() instanceof CoreOp.VarAccessOp.VarLoadOp)
                        .map(operand -> (CoreOp.VarAccessOp.VarLoadOp) ((Op.Result) operand).op())
                        .forEach(nodesInvolved::add);
            }
        });


        return Trxfmr.of(lookup, funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
            if (invoke(lookup, op) instanceof Invoke invoke) {
                HATVectorOp memoryViewOp = buildVectorBinaryOp(
                        findVectorVarNameOrNull(invoke.op().operands().getFirst()),
                        BinaryOpEnum.of(invoke.op()),
                        getVectorShape(invoke.lookup(), invoke.returnType()),
                        blockBuilder.context().getValues(invoke.op().operands())
                );
                blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.op(copyLocation(invoke.op(), memoryViewOp)));
            } else if (op instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                HATVectorOp memoryViewOp = new HATVectorOp.HATVectorVarLoadOp(
                        findVectorVarNameOrNull(varLoadOp),
                        varLoadOp.resultType(),
                        getVectorShapeOrNullFromVarLoad(varLoadOp),
                        blockBuilder.context().getValues(varLoadOp.operands())
                );
                blockBuilder.context().mapValue(varLoadOp.result(), blockBuilder.op(copyLocation(varLoadOp, memoryViewOp)));
            }
            return blockBuilder;
        }).funcOp();
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        switch (Objects.requireNonNull(vectorOperation)) {
            case FLOAT4_LOAD -> funcOp = dialectifyVectorLoad(lookup,funcOp);
            case FLOAT2_LOAD -> funcOp = dialectifyVectorLoad(lookup,funcOp);
            case OF -> funcOp = dialectifyVectorOf(lookup,funcOp);
            case MAKE_MUTABLE -> funcOp = dialectifyMutableOf(lookup,funcOp);
            default -> {
                // Find binary operations
                funcOp = dialectifyVectorBinaryOps(lookup,funcOp);
                funcOp = dialectifyVectorBinaryWithConcatenationOps(lookup,funcOp);
            }
        }
        return funcOp;
    }

    public static final class AddPhase extends HATVectorPhase {
        public AddPhase() {
            super( VectorOperation.ADD);
        }
    }

    public static final class DivPhase extends HATVectorPhase {

        public DivPhase() {
            super(VectorOperation.DIV);
        }
    }

    public static final class MakeMutable extends HATVectorPhase {

        public MakeMutable() {
            super(VectorOperation.MAKE_MUTABLE);
        }
    }

    public static final class Float4LoadPhase extends HATVectorPhase {

        public Float4LoadPhase() {
            super(VectorOperation.FLOAT4_LOAD);
        }
    }

    public static final class Float2LoadPhase extends HATVectorPhase {

        public Float2LoadPhase() {
            super( VectorOperation.FLOAT2_LOAD);
        }
    }

    public static final class Float4OfPhase extends HATVectorPhase {

        public Float4OfPhase() {
            super( VectorOperation.OF);
        }
    }

    public static final class MulPhase extends HATVectorPhase {

        public MulPhase() {
            super( VectorOperation.MUL);
        }
    }

    public static final class SubPhase extends HATVectorPhase {

        public SubPhase() {
            super( VectorOperation.SUB);
        }
    }
}
