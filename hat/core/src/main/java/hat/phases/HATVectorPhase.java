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

import hat.device.NonMappableIface;
import hat.dialect.HATVectorOp;
import optkl.IfaceValue.Vector;
import jdk.incubator.code.Block;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.OpHelper;
import optkl.Trxfmr;
import optkl.VarTable;

import java.lang.invoke.MethodHandles;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke;

public final class HATVectorPhase implements HATPhase {

    private String functionName;

    private MethodHandles.Lookup lookup = null;

    public boolean isSharedOrPrivate(Op op) {
        return isSharedOrPrivate(op.operands().getFirst());
    }

    private static final Set<String> NON_MAPPABLE_IFACE_INSTANCES = Set.of("createshared", "createlocal", "createprivate");

    public boolean isSharedOrPrivate(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isSharedOrPrivate(varLoadOp); //recurse
            case CoreOp.VarOp varOp -> {
                // extra analysis
                Value first = varOp.operands().getFirst();
                if (first instanceof Block.Parameter) {
                    // if the var comes from a parameter, then it is global memory
                    yield false;
                }
                // otherwise we continue traversal
                yield isSharedOrPrivate(varOp);
            }
            case JavaOp.InvokeOp invoke -> {
                // If we get an invoke, we need to get method name, and check the following

                // warp to Invoke
                if (lookup == null) {
                    throw new IllegalStateException("Lookup has not been initialized");
                }

                Stream<Invoke> stream = OpHelper.Invoke.stream(lookup, invoke);
                Optional<Invoke> invokeOptional = stream.findFirst();
                // Check for the right class
                if (invokeOptional.isPresent() && invokeOptional.get().refIs(NonMappableIface.class)) {
                    // check for the method name
                    String lowerCase = invoke.invokeReference().name().toLowerCase();
                    yield NON_MAPPABLE_IFACE_INSTANCES.contains(lowerCase);
                }
                yield false;
            }
            default -> false;
        };
    }

    //recursive
    public Vector.Shape getVectorShapeOrNullFromVarLoad(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getVectorShapeOrNull(varLoadOp.operands().getFirst());
    }

    private Vector.Shape getVectorShapeOrNull(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorShapeOrNullFromVarLoad(varLoadOp);
        } else if (v.declaringElement() instanceof HATVectorOp hatVectorOp) {
            return hatVectorOp.vectorShape();
        }
        return null;
    }

    public enum VOp {
        FLOAT4_LOAD("float4View"),
        FLOAT2_LOAD("float2View"),
        OF("of"),
        ADD("add"),
        SUB("sub"),
        MUL("mul"),
        DIV("div"),
        MAKE_MUTABLE("makeMutable");
        final String methodName;

        VOp(String methodName) {
            this.methodName = methodName;
        }
    }

    private void varOpVector(Block.Builder blockBuilder, CoreOp.VarOp varOp, VarTable varTable) {
        Op.Result result = blockBuilder.add(varOp);
        varTable.addIfNeededOrThrow(functionName, result.op(), VarTable.HATOpAttribute.VECTOR);
    }

    private CoreOp.FuncOp dialectifyVectorLoad(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation) {
        this.lookup = lookup;
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup, funcOp).forEach(variable -> {
            if (variable.firstOperandAsInvoke() instanceof Invoke invoke
                    && invoke.returns(Vector.class)
                    && invoke.named(vectorOperation.methodName)) {
                Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
                vectorShapeMap.put(variable.op(), vectorShape);
            }
        });

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                varOpVector(blockBuilder, varOp, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private CoreOp.FuncOp dialectifyVectorOf(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation) {
        Map<Op, Vector.Shape> vectorShapeMap = getVectorShapeMap(lookup, funcOp, vectorOperation);
        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                varOpVector(blockBuilder, varOp, varTable);
            } else {
                blockBuilder.add(op);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private CoreOp.FuncOp dialectifyVectorBinaryOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation) {
        Map<Op, Vector.Shape> vectorShapeMap = new HashMap<>();
        OpHelper.Named.Variable.stream(lookup, funcOp).forEach(variable -> {
            if (variable.firstOperandAsInvoke() instanceof Invoke invoke
                    && invoke.named(vectorOperation.methodName)
                    && invoke.returns(Vector.class)) {
                Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
                vectorShapeMap.put(variable.op(), vectorShape);
            }
        });

        return Trxfmr.of(lookup, funcOp).transform(vectorShapeMap::containsKey, (blockBuilder, op) -> {
            if (op instanceof CoreOp.VarOp varOp) {
                varOpVector(blockBuilder, varOp, varTable);
            }
            return blockBuilder;
        }, varTable).funcOp();
    }

    private Map<Op, Vector.Shape> getVectorShapeMap(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VOp vectorOperation) {
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

//    private CoreOp.FuncOp transformBinaryOperation(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation) {
//        funcOp = dialectifyVectorBinaryOps(lookup, funcOp, varTable, vectorOperation);
//        return dialectifyVectorBinaryWithConcatenationOps(lookup, funcOp, varTable, vectorOperation);
//    }

//    private boolean isVectorOperation(CodeElement<?, ?> codeElement, VOp vectorOperation) {
//        return invoke(lookup, codeElement) instanceof Invoke invoke && invoke.returns(Vector.class) && invoke.named(vectorOperation.methodName);
//    }

//    private CoreOp.FuncOp dialectifyVectorBinaryWithConcatenationOps(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation) {
//        Set<CodeElement<?, ?>> nodesInvolved = new HashSet<>();
//        funcOp.elements().forEach(codeElement -> {
//            if (isVectorOperation(codeElement, vectorOperation)) {
//                Invoke invoke = invoke(lookup, codeElement);
//                invoke.op().operands().stream()// this can't be replaced with findFirst
//                        .filter(operand -> operand instanceof Op.Result && ((Op.Result) operand).op() instanceof CoreOp.VarAccessOp.VarLoadOp)
//                        .map(operand -> (CoreOp.VarAccessOp.VarLoadOp) ((Op.Result) operand).op())
//                        .forEach(_ -> nodesInvolved.add(invoke.op()));
//            }
//        });
//
//        return Trxfmr.of(lookup, funcOp).transform(nodesInvolved::contains, (blockBuilder, op) -> {
//            if (invoke(lookup, op) instanceof Invoke invoke) {
//                blockBuilder.add(op);
////                HATVectorOp vectorOp = new HATVectorOp.HATVectorBinaryOp(
////                        OpHelper.findVectorVarNameOrNull(invoke.op().operands().getFirst()),
////                        BinaryOpEnum.of(invoke.op()),
////                        invoke.returnType(),
////                        getVectorShape(invoke.lookup(), invoke.returnType()),
////                        blockBuilder.context().getValues(invoke.op().operands())
////                );
////                blockBuilder.context().mapValue(invoke.op().result(), blockBuilder.add(copyLocation(invoke.op(), vectorOp)));
//            }
//            return blockBuilder;
//        }, varTable).funcOp();
//    }

    @FunctionalInterface
    public interface VectorTransformer {
        CoreOp.FuncOp apply(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable, VOp vectorOperation);
    }

    private final Map<VOp, VectorTransformer> vectorTransformers;

    public HATVectorPhase() {
        vectorTransformers = new LinkedHashMap<>();
        vectorTransformers.put(VOp.FLOAT4_LOAD, this::dialectifyVectorLoad); // done
        vectorTransformers.put(VOp.FLOAT2_LOAD, this::dialectifyVectorLoad); // done
        vectorTransformers.put(VOp.OF, this::dialectifyVectorOf);            // done
        vectorTransformers.put(VOp.MAKE_MUTABLE, this::dialectifyVectorOf);  // done
        vectorTransformers.put(VOp.ADD, this::dialectifyVectorBinaryOps);
        vectorTransformers.put(VOp.SUB, this::dialectifyVectorBinaryOps);
        vectorTransformers.put(VOp.MUL, this::dialectifyVectorBinaryOps);
        vectorTransformers.put(VOp.DIV, this::dialectifyVectorBinaryOps);
    }

    @Override
    public CoreOp.FuncOp transform(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp, VarTable varTable) {
        this.functionName = funcOp.funcName();
        for (VOp vectorOperation : vectorTransformers.keySet()) {
            VectorTransformer transformer = vectorTransformers.get(vectorOperation);
            funcOp = transformer.apply(lookup, funcOp, varTable, vectorOperation);
        }
        return funcOp;
    }
}
