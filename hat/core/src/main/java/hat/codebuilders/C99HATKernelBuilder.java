/*
 * Copyright (c) 2024-2026, Oracle and/or its affiliates. All rights reserved.
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
package hat.codebuilders;

import hat.KernelContext;
import hat.buffer.BF16Array;
import hat.callgraph.KernelCallGraph;
import hat.device.NonMappableIface;
import hat.dialect.HATBarrierOp;
import hat.dialect.HATPtrOp;
import hat.dialect.HATTensorOp;
import hat.dialect.HATThreadOp;
import hat.phases.HATArrayViewPhase;
import hat.phases.HATFP16Phase;
import hat.phases.HATPhaseUtils;
import hat.phases.HATPhaseUtils.InvokeVar;
import hat.types.BF16;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.Block;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import hat.types.S16ImplOfF16;
import optkl.IfaceValue;
import jdk.incubator.code.Value;
import optkl.OpHelper;
import optkl.VarTable;
import optkl.codebuilders.ScopedCodeBuilderContext;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Schema;
import jdk.incubator.code.Op;
import optkl.FuncOpParams;
import optkl.util.Regex;
import optkl.util.Mutable;
import jdk.incubator.code.dialect.core.CoreOp;

import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Stream;

import static hat.buffer.F16Array.F16Impl;
import static hat.codebuilders.C99VecAndMatHandler.handleInvoke;
import static hat.codebuilders.C99VecAndMatHandler.handleType;
import static hat.codebuilders.C99VecAndMatHandler.isVecInvoke;
import static hat.codebuilders.C99VecAndMatHandler.isVecOrMatType;
import static hat.phases.HATPhaseUtils.NON_MAPPABLE_IFACE;
import static hat.phases.HATPhaseUtils.findIsSharedOrPrivateSpace;
import static hat.phases.HATPhaseUtils.findVectorVarNameOrNull;
import static hat.phases.HATPhaseUtils.getVectorShapeFromOperandN;
import static hat.phases.HATPhaseUtils.isArrayReference;
import static hat.phases.HATPhaseUtils.isAttributeSharedOrPrivate;
import static hat.phases.HATPhaseUtils.isF16Local;
import static hat.phases.HATPhaseUtils.isInvokeFromNarrowTypeConversion;
import static hat.phases.HATPhaseUtils.isInvokeLoadingFromOnChipMemory;
import static hat.phases.HATPhaseUtils.isMathOperation;
import static hat.phases.HATPhaseUtils.isOperandF32;
import static hat.phases.HATPhaseUtils.isS16BinaryOp;
import static hat.phases.HATPhaseUtils.isS16Conversion;
import static hat.phases.HATPhaseUtils.isS16ToFloatConversion;
import static hat.phases.HATPhaseUtils.isSharedOrPrivate;
import static hat.phases.HATPhaseUtils.isTensorOperation;
import static hat.phases.HATPhaseUtils.isVectorBinaryOperation;
import static hat.phases.HATPhaseUtils.isVectorOperation;
import static hat.phases.HATPhaseUtils.isVectorSelectOperation;
import static hat.phases.HATPhaseUtils.isVectorView;
import static hat.phases.HATPhaseUtils.mapLane;
import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke.invoke;

import static optkl.OpHelper.opFromFirstOperandOrNull;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilder<T> implements HATOpDispatcher<T> {

    protected final KernelCallGraph kernelCallGraph;
    private final VarTable varTable;

    public static final String VLOADN = "VLOADN";
    public static final String VSTOREN = "VSTOREN";
    public static final String N = "N";
    public static final String IS_LOCAL = "isLocal";
    public static final String VLOAD = "vload";
    public static final String VSTORE = "vstore";
    public static final String VECTOR = "VECTOR_";
    public static final String VECTOR_VAL = "vVal";
    public static final String ELEMENT_TYPE = "elementType";
    public static final String VECTOR_OF = "VECTOR_OF";
    public static final String VSELECT_LOAD = "VSELECT_LOAD";
    public static final String VSELECT_STORE = "VSELECT_STORE";
    public static final String F16_OF = "F16_OF";
    public static final String BF16_OF = "BF16_OF";
    public static final String F16_TO_FLOAT_0 = "F16_TO_FLOAT_0";
    public static final String F16_TO_FLOAT_1 = "F16_TO_FLOAT_1";
    public static final String BF16_TO_FLOAT_0 = "BF16_TO_FLOAT_0";
    public static final String BF16_TO_FLOAT_1 = "BF16_TO_FLOAT_1";

    protected C99HATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
        this.kernelCallGraph = kernelCallGraph;
        this.varTable = kernelCallGraph.getVarTable();
    }

    protected boolean useVectors() {
        return kernelCallGraph.useVectors();
    }

    protected boolean useS16Types() {
        return !kernelCallGraph.accessedFP16Classes.isEmpty() || useTensors();
    }

    protected boolean useThreadConstruct(String construct) {
        return kernelCallGraph.accessedKernelContextFields.contains(construct);
    }

    protected boolean useAtomic() {
        return kernelCallGraph.isUsesAtomics();
    }

    protected boolean useBarrier() {
        return kernelCallGraph.isUsesBarrier();
    }

    protected boolean useTensors() {
        return kernelCallGraph.useTensors();
    }

    public final T HAT_KERNEL() {
        return keyword("HAT_KERNEL");
    }

    public final T HAT_FUNC() {
        return keyword("HAT_FUNC");
    }

    public final T HAT_GLOBAL_MEM() {
        return keyword("HAT_GLOBAL_MEM");
    }

    public final T HAT_LOCAL_MEM() {
        return keyword("HAT_LOCAL_MEM");
    }

    public final T HAT_BARRIER() {
        return keyword("HAT_BARRIER");
    }

    public final T HAT_GIX() {
        return id("HAT_GIX");
    }

    public final T HAT_GIY() {
        return id("HAT_GIY");
    }

    public final T HAT_GIZ() {
        return id("HAT_GIZ");
    }

    public final T HAT_GSX() {
        return id("HAT_GSX");
    }

    public final T HAT_GSY() {
        return id("HAT_GSY");
    }

    public final T HAT_GSZ() {
        return id("HAT_GSZ");
    }

    public final T HAT_LIX() {
        return id("HAT_LIX");
    }

    public final T HAT_LIY() {
        return id("HAT_LIY");
    }

    public final T HAT_LIZ() {
        return id("HAT_LIZ");
    }

    public final T HAT_LSX() {
        return id("HAT_LSX");
    }

    public final T HAT_LSY() {
        return id("HAT_LSY");
    }

    public final T HAT_LSZ() {
        return id("HAT_LSZ");
    }

    public final T HAT_BIX() {
        return id("HAT_BIX");
    }

    public final T HAT_BIY() {
        return id("HAT_BIY");
    }

    public final T HAT_BIZ() {
        return id("HAT_BIZ");
    }

    public final T HAT_BSX() {
        return id("HAT_BSX");
    }

    public final T HAT_BSY() {
        return id("HAT_BSY");
    }

    public final T HAT_BSZ() {
        return id("HAT_BSZ");
    }

    public final T HAT_WARP_SIZE() {
        return hatWarpSize();
    }

    @Override
    public final T hatThreadIdOp(HATThreadOp threadOp) {
        return (switch (threadOp) {
            case HATThreadOp.HAT_LI.HAT_LIX _ -> HAT_LIX();
            case HATThreadOp.HAT_LI.HAT_LIY _ -> HAT_LIY();
            case HATThreadOp.HAT_LI.HAT_LIZ _ -> HAT_LIZ();
            case HATThreadOp.HAT_LS.HAT_LSX _ -> HAT_LSX();
            case HATThreadOp.HAT_LS.HAT_LSY _ -> HAT_LSY();
            case HATThreadOp.HAT_LS.HAT_LSZ _ -> HAT_LSZ();
            case HATThreadOp.HAT_GI.HAT_GIX _ -> HAT_GIX();
            case HATThreadOp.HAT_GI.HAT_GIY _ -> HAT_GIY();
            case HATThreadOp.HAT_GI.HAT_GIZ _ -> HAT_GIZ();
            case HATThreadOp.HAT_GS.HAT_GSX _ -> HAT_GSX();
            case HATThreadOp.HAT_GS.HAT_GSY _ -> HAT_GSY();
            case HATThreadOp.HAT_GS.HAT_GSZ _ -> HAT_GSZ();
            case HATThreadOp.HAT_BI.HAT_BIX _ -> HAT_BIX();
            case HATThreadOp.HAT_BI.HAT_BIY _ -> HAT_BIY();
            case HATThreadOp.HAT_BI.HAT_BIZ _ -> HAT_BIZ();
            case HATThreadOp.HAT_BS.HAT_BSX _ -> HAT_BSX();
            case HATThreadOp.HAT_BS.HAT_BSY _ -> HAT_BSY();
            case HATThreadOp.HAT_BS.HAT_BSZ _ -> HAT_BSZ();
            case HATThreadOp.HAT_WARP_SIZE  _ -> HAT_WARP_SIZE();
        });
    }

    public final T kernelDeclaration(CoreOp.FuncOp funcOp) {
        return HAT_KERNEL().sp().voidType().sp().funcName(funcOp);
    }

    public final T functionDeclaration(JavaType javaType, CoreOp.FuncOp funcOp) {
        return HAT_FUNC().sp().type(javaType).sp().funcName(funcOp);
    }

    public final boolean isHalfType(Schema.IfaceType ifaceType) {
        return ifaceType.iface.isAssignableFrom(F16.class)
                || ifaceType.iface.isAssignableFrom(F16Impl.class);
    }

    public final boolean isbfloat16(Schema.IfaceType ifaceType) {
        return ifaceType.iface.isAssignableFrom(BF16.class)
                || ifaceType.iface.isAssignableFrom(BF16Array.BF16Impl.class);
    }

    public final T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword()
                .sp()
                .structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .sp()
                .suffix_s(ifaceType.iface.getSimpleName())
                .braceNlIndented(_ -> {
                    int fieldCount = ifaceType.fields.size();
                    var fieldIdx = Mutable.of(0);
                    semicolonNlSeparated(
                            ifaceType.fields,
                            field -> {
                                boolean isLast = fieldIdx.get() == fieldCount - 1;
                                if (field instanceof Schema.FieldNode.AbstractPrimitiveField primitiveField) {
                                    if (isHalfType(ifaceType)) {
                                        type("half");
                                    } else if (isbfloat16(ifaceType)) {
                                        type("BFLOAT16");
                                    } else {
                                        type(primitiveField.type.getSimpleName());
                                    }
                                    sp().type(primitiveField.name);
                                    if (primitiveField instanceof Schema.FieldNode.PrimitiveArray array) {
                                        if (array instanceof Schema.FieldNode.PrimitiveFieldControlledArray) {
                                            if (isLast && ifaceType.parent == null) {
                                                sbrace(_ -> literal(1));
                                            } else {
                                                boolean[] done = new boolean[]{false};
                                                if (boundSchema != null) {
                                                    boundSchema.boundArrayFields().forEach(a -> {
                                                        if (a.field.equals(array)) {
                                                            sbrace(_ -> literal(a.len));
                                                            done[0] = true;
                                                        }
                                                    });
                                                    if (!done[0]) {
                                                        throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                                    }
                                                } else {
                                                    throw new IllegalStateException("bound schema is null  !");
                                                }
                                            }
                                        } else if (array instanceof Schema.FieldNode.PrimitiveFixedArray fixed) {
                                            sbrace(_ -> literal(Math.max(1, fixed.len)));
                                        } else {
                                            throw new IllegalStateException("what kind of array ");
                                        }
                                    }
                                } else if (field instanceof Schema.FieldNode.AbstractIfaceField ifaceField) {
                                    suffix_t(ifaceField.ifaceType.iface);
                                    sp().type(ifaceField.name);
                                    if (ifaceField instanceof Schema.FieldNode.IfaceArray array) {
                                        if (array instanceof Schema.FieldNode.IfaceFieldControlledArray fieldControlledArray) {
                                            if (isLast && ifaceType.parent == null) {
                                                sbrace(_ -> literal(1));
                                            } else {
                                                boundSchema.boundArrayFields().stream()
                                                        .filter(a -> a.field.equals(ifaceField))
                                                        .findFirst()
                                                        .ifPresentOrElse(
                                                                a -> sbrace(_ -> literal(a.len)),
                                                                () -> {
                                                                    throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                                                });
                                            }
                                        } else if (array instanceof Schema.FieldNode.IfaceFixedArray fixed) {
                                            sbrace(_ -> literal(Math.max(1, fixed.len)));
                                        } else {
                                            throw new IllegalStateException("what kind of array ");
                                        }
                                    }
                                } else if (field instanceof Schema.SchemaNode.Padding padding) {
                                    u08Type().sp().identifierWithRandomSuffix("pad$", 5).sbrace(_ -> intValue((int) (padding.len)));//; emitText(toC99(padding));
                                } else {
                                    throw new IllegalStateException("hmm");
                                }
                                fieldIdx.set(fieldIdx.get() + 1);
                            });
                    semicolon();
                }).suffix_t(ifaceType.iface).semicolon().nl().nl();
        return self();
    }

    /**
     * Generates a suffix from a set of n-random characters from a set of legal characters in C99.
     */
    public final T identifierWithRandomSuffix(String prefix, final int len) {
        var sb = new StringBuilder();
        final var LEGAL_CHARS = "_$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        ThreadLocalRandom.current() //
                .ints(len, 0, LEGAL_CHARS.length()) //
                .mapToObj(LEGAL_CHARS::charAt) //
                .forEach(sb::append);
        id(prefix + sb);
        return self();
    }

    @Override
    public final T hatBarrierOp(HATBarrierOp barrierOp) {
        return HAT_BARRIER();
    }

    public final T types() {
        return typedefKeyword().sp().s08Type("byte").snl()
                .typedefKeyword().sp().s08Type("boolean").snl()
                .typedefStruct(KernelContext.class, _ -> s32Type("dimensions").semicolon()).nl();
    }

    @Override
    public final T fieldLoadOp(JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        var fieldAccess = fieldAccess(scopedCodeBuilderContext().lookup(), fieldLoadOp);
        if (fieldAccess.operandCount() == 0 && fieldAccess.isPrimitive()) {
            literal(fieldAccess.getStaticFinalPrimitiveValue().toString());

            // Experiment: if it is float, then generate "f"
            PrimitiveType primitiveType = (PrimitiveType) fieldLoadOp.resultType();
            if (primitiveType.toBasicType() == JavaType.FLOAT) {
                emitText("f");
            }
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    @Override
    public final T type(JavaType javaType) {
        // TODO: I would not include this method in the visitor, rather just the tree-nodes. Types can be resolved when needed directly.

        if (isVecOrMatType(scopedCodeBuilderContext().lookup(), javaType)) {
            handleType(self(), javaType);
        } else if (javaType instanceof ClassType classType
                && OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, IfaceValue.class)
                && !OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, S16ImplOfF16.class)
        ) {
            HAT_GLOBAL_MEM().sp().suffix_t(classType).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, KernelContext.class)) {
            HAT_GLOBAL_MEM().sp().suffix_t(KernelContext.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, F16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().sp().suffix_t(F16Impl.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, BF16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().sp().suffix_t(BF16Array.BF16Impl.class).asterisk();
        } else {
            type(javaType.toString());
        }
        return self();
    }

    public final T kernelMethod(CoreOp.FuncOp funcOp) {
        scopedCodeBuilderContext().funcScope(funcOp, () -> {
            nl();
            functionDeclaration((JavaType) funcOp.body().yieldType(), funcOp);
            parenNlIndented(_ ->
                    commaNlSeparated(
                            new FuncOpParams(funcOp).list(),
                            this::declareParam
                    )
            );

            braceNlIndented(_ ->
                    nlSeparated(
                            OpHelper.Statement.statements(funcOp.bodies().getFirst().entryBlock()),
                            this::statement
                    )
            );
        });
        return self();
    }

    public final T kernelEntrypoint() {
        nl();
        scopedCodeBuilderContext().funcScope(scopedCodeBuilderContext().funcOp(), () ->
                kernelDeclaration(scopedCodeBuilderContext().funcOp())
                        .parenNlIndented(_ -> commaNlSeparated(
                                scopedCodeBuilderContext().paramTable.list(),
                                this::declareParam)
                        )
                        .braceNlIndented(_ -> nlSeparated(
                                        OpHelper.Statement.statements(scopedCodeBuilderContext().funcOp().bodies().getFirst().entryBlock()),
                                        this::statement
                                )
                        )
        );
        return self();
    }

    public final T f16Type() {
        return suffix_t(F16.class);
    }

    public final T bf16Type() {
        return suffix_t(BF16.class);
    }

    protected T f16OrBF16(Class<?> float16Class) {
        if (F16.class.isAssignableFrom(float16Class)) {
            return f16Type();
        } else if (BF16.class.isAssignableFrom(float16Class)) {
            return bf16Type();
        } else {
            throw new IllegalStateException("Unexpected value: " + float16Class);
        }
    }

    private boolean isMixedFirstOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATFP16Phase.FIRST_OP;
    }

    private boolean isMixedSecondOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATFP16Phase.LAST_OP;
    }

    public final T builtin_float2bfloat16() {
        return id("floatTobfloat16");
    }

    public final T builtin_bfloat16ToFloat() {
        return id("bfloat16Tofloat");
    }

    protected static final String VALUE = "value";
    protected static final String ARRAY = "array";

    private T binaryOperationsForBfloat16(Invoke invoke) {
        var lookup = scopedCodeBuilderContext().lookup();
        boolean isFirstOperandReference = isArrayReference(lookup, invoke.op().operands().get(0));
        boolean isSecondOperandReference = isArrayReference(lookup, invoke.op().operands().get(1));
        final byte f32Mixed;
        if (!isFirstOperandReference && isOperandF32(invoke.op().operands().get(0))) {
            f32Mixed = HATFP16Phase.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(invoke.op().operands().get(1))) {
            f32Mixed = HATFP16Phase.LAST_OP;
        } else {
            f32Mixed = 0x00;
        }

        paren(_ -> bf16Type());
        brace(_ -> {
            paren(_ -> {
                builtin_float2bfloat16();
                oparen();
                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();// open
                }
                recurse(OpHelper.asResultOrThrow(invoke.op().operands().getFirst()).op());

                if (isFirstOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().getFirst())) {
                    dot().id(VALUE);
                }

                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    cparen(); //closed
                }
                sp().id(matchSymbol(invoke.name())).sp();

                if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();
                }

                recurse(OpHelper.asResultOrThrow(invoke.op().operands().get(1)).op());
                if (isSecondOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().get(1))) {
                    dot().id(VALUE);
                }

                if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
                    cparen();
                }
            });
            cparen();
        });
        return self();
    }

    public String matchSymbol(String operationName) {
        return switch (operationName) {
            case "add" -> "+";
            case "sub" -> "-";
            case "mul" -> "*";
            case "div" -> "/";
            default -> "";
        };
    }

    public T hatF16BinaryOp(Invoke invoke, Class<?> float16Class) {
        if (BF16.class.isAssignableFrom(float16Class)) {
            return binaryOperationsForBfloat16(invoke);
        }
        paren(_ -> f16Type());
        return brace(_ ->
                paren(_ -> {
                    recurse(OpHelper.asResultOrThrow(invoke.op().operands().getFirst()).op());
                    boolean isFirstOperandReference = isArrayReference(scopedCodeBuilderContext.lookup(), invoke.op().operands().get(0));
                    boolean isSecondOperandReference = isArrayReference(scopedCodeBuilderContext.lookup(), invoke.op().operands().get(1));
                    if (isFirstOperandReference) {
                        rarrow().id(VALUE);
                    } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().getFirst())) {
                        dot().id(VALUE);
                    } else {
                        blockComment("hatF16BinaryOp not a result !!");
                    }
                    sp().id(matchSymbol(invoke.name())).sp();
                    recurse(OpHelper.asResultOrThrow(invoke.op().operands().get(1)).op());
                    if (isSecondOperandReference) {
                        rarrow().id(VALUE);
                    } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().get(1))) {
                        dot().id(VALUE);
                    } else {
                        blockComment("hatF16BinaryOp not a value !!");
                    }
                })
        );
    }

    public final T generateVectorOf(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape) {
        return id(VECTOR_OF + vectorShape.lanes()).paren(_ ->
                id(vectorShape.codeType().toString())
                .comma()
                .commaSpaceSeparated(invokeOp.operands(),
                        operand -> recurse(OpHelper.asResultOrThrow(operand).op())));
    }

    public final T generateOnChipMemoryLoad(JavaOp.InvokeOp invoke) {
        return recurse(OpHelper.asResultOrThrow(invoke.operands().getFirst()).op())
                .dot().id(invoke.invokeReference().name())
                .when(invoke.operands().size() > 1, _ -> // If the hatMemoryLoadOp has more than 1 operand, the second is the index
                        sbrace(_ -> recurse(OpHelper.asResultOrThrow(invoke.operands().get(1)).op()))
                );
    }

    public final T hatPtrLoadOp(HATPtrOp.HATPtrLoadOp hatPtrLoadOp) {
        ptrAccess(hatPtrLoadOp);
        return self();
    }

    @Override
    public final T hatPtrStoreOp(HATPtrOp.HATPtrStoreOp hatPtrStoreOp) {
        ptrAccess(hatPtrStoreOp).equals().recurse(((Op.Result) hatPtrStoreOp.operands().getLast()).op());
        return self();
    }

    @Override
    public final T hatPtrLengthOp(HATPtrOp.HATPtrLengthOp hatPtrLengthOp) {
        ptrAccess(hatPtrLengthOp);
        return self();
    }

    private T ptrAccess(HATPtrOp hatPtrOp) {
        id(hatPtrOp.name());
        boolean isLocalOrPrivateDS = false;
        if (((Op.Result) hatPtrOp.operands().getFirst()).op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
            if (resolve instanceof CoreOp.VarOp varOp) {
                Value value = varOp.operands().getFirst();
                if (value.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {

                    Stream<Invoke> stream = OpHelper.Invoke.stream(scopedCodeBuilderContext.lookup(), invokeOp);
                    Optional<Invoke> invoke = stream.findFirst();
                    // Check for the right class
                    if (invoke.isPresent() && invoke.get().refIs(NonMappableIface.class)) {
                        // check for the method name
                        String lowerCase = invokeOp.invokeReference().name().toLowerCase();
                        isLocalOrPrivateDS = NON_MAPPABLE_IFACE.contains(lowerCase);
                    }
                }
            }
        }

        dotOrArrow(isLocalOrPrivateDS);

        if (hatPtrOp instanceof HATPtrOp.HATPtrLengthOp) {
            id("length");
        } else {
            final boolean finalIsLocalOrPrivateDS = isLocalOrPrivateDS;
            id(ARRAY).sbrace(_ -> {
                paren(_ -> id("long")); // is this a cast (long)  maybe cast(_->typeName("long"))?
                paren(_ -> {
                    if (hatPtrOp.strides().size() > 1) {
                        paren(_ -> recurse(((Op.Result) hatPtrOp.operands().get(2)).op()));
                        asterisk().id(hatPtrOp.name());
                        dotOrArrow(finalIsLocalOrPrivateDS).id(hatPtrOp.strides() != null ? hatPtrOp.strides().getFirst() : "width");
                        add().paren(_ -> recurse(((Op.Result) hatPtrOp.operands().get(1)).op()));
                    } else {
                        recurse(((Op.Result) hatPtrOp.operands().get(1)).op());
                    }
                });
            });
        }
        return self();
    }

    /**
     * <code>
     * float bfloat16Tofloat(ushort bf16) {
     * b16_t b;
     * b.s[0] = 0;
     * b.s[1] = s;
     * return b.f;
     * }
     * </code>
     *
     * @param parameterName
     * @return
     */
    public final T build_builtin_bfloat16ToFloat(String parameterName) {
        String b16 = "b16";
        String s = "s";
        String f = "f";
        return funcDef(_ -> f32Type(),
                _ -> builtin_bfloat16ToFloat(),
                _ -> u16Type(parameterName),
                _ -> bfloat16Type(b16).snl()
                        .id(b16).dot().id(s).sbrace(_ -> intConstZero()).equals().intConstZero().snl()
                        .id(b16).dot().id(s).sbrace(_ -> intConstOne()).equals().constant(parameterName).snl()
                        .returnKeyword(_ -> id(b16).dot().id(f)));
    }

    /**
     * <code>
     * ushort floatTobfloat16(float f) {
     * b16_t b = {f};
     * uint32_t bits = b.i;
     * short sign_bit = (short)((bits & 0x8000_0000) >> 16);
     * int lsb    = bits & 0x1_0000;
     * int round  = bits & 0x0_8000;
     * int sticky = bits & 0x0_7FFF;
     * if (round != 0 && ((lsb | sticky) != 0 )) {
     * bits += 0x1_0000;
     * }
     * return (short) (((bits >> 16 ) | sign_bit) & 0xffff);
     * }
     * </code>
     *
     * @param parameterName
     * @return
     */
    public final T build_builtin_float2bfloat16(String parameterName) {
        String idBFloat16 = "b";
        return funcDef(
                _ -> u16Type(),
                _ -> builtin_float2bfloat16(),
                _ -> f32Type(parameterName),
                _ -> assign(_ -> bfloat16Type(idBFloat16),
                        _ -> brace(_ -> id(parameterName)).snl())
                        .assign(_ -> u32Type("bits"), _ -> id(idBFloat16).dot().id("i")).snl()
                        .assign(_ -> u16Type("sign_bit"), _ -> cast(_ -> s16Type()).paren(_ -> paren(_ -> id("bits").ampersand().constant("0x80000000")).rightShift(16))).snl()
                        .assign(_ -> s32Type("lsb"), _ -> id("bits").ampersand().constant("0x10000")).snl()
                        .assign(_ -> s32Type("round"), _ -> id("bits").ampersand().constant("0x08000")).snl()
                        .assign(_ -> s32Type("sticky"), _ -> id("bits").ampersand().constant("0x07FFF")).snl()
                        .ifTrueCondition(_ -> id("round").sp().ne().sp().intConstZero().condAnd().paren(_ -> paren(_ -> id("lsb").bitwiseOR().id("sticky")).ne().intConstZero()),
                                _ -> id("bits").sp().plusEquals().sp().constant("0x10000"))
                        .returnKeyword(_ -> cast(_ -> u16Type()).paren(_ -> paren(_ -> paren(_ -> id("bits").rightShift(16)).bitwiseOR().id("sign_bit")).ampersand().constant("0xffff"))));
    }

    @Override
    public final T varLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp varOp -> varName(varOp);
            case null, default -> {
            }
        }
        return self();
    }

    @Override
    public final T varStoreOp(CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
        Op op = scopedCodeBuilderContext().resolve(varStoreOp.operands().getFirst());

        //TODO see if VarLikeOp marker interface fixes this

        // TODO: each of these is delegating to varName().... maybe varName should be handling these types.

        // When the op is intended to operate as VarOp, then we need to include it in the following switch.
        // This is because HAT has its own dialect, and some of the Ops operate on HAT Types (not included in the Java
        // dialect). For instance, private data structures, local data structures, vector types, etc.
        switch (op) {
            case CoreOp.VarOp varOp -> varName(varOp);
            case null, default -> throw new IllegalStateException("What type of varStoreOp is this?");
        }
        equals().parenthesisIfNeeded(varStoreOp, ((Op.Result) varStoreOp.operands().get(1)).op());
        return self();
    }

    @Override
    public final T convOp(JavaOp.ConvOp convOp) {
        // TODO: I think we need to work out how to handle doubles. If I remove this OpenCL on MAC complains (no FP64)
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type((JavaType) convOp.resultType()));
        }
        parenthesisIfNeeded(convOp, ((Op.Result) convOp.operands().getFirst()).op());
        return self();
    }

    static Regex atomicIncRegex = Regex.of("(atomic.*)Inc");

    private boolean isInvokeFromSharedOrPrivate(Op.Result instance, OpHelper.Invoke invoke) {
        boolean isLocalOrPrivateDS = false;
        VarTable vTable = kernelCallGraph.getVarTable();
        if (instance.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                && scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst()) instanceof CoreOp.VarOp varOp
                && vTable.doesVarOpExist(scopedCodeBuilderContext.funcOp().funcName(), varOp)) {
            VarTable.HATOpAttribute attribute = vTable.getAttributeOrThrow(scopedCodeBuilderContext.funcOp().funcName(), varOp);
            isLocalOrPrivateDS = isAttributeSharedOrPrivate(attribute, invoke);
        }
        return isLocalOrPrivateDS;
    }

    private boolean isVectorNamed(Invoke invoke, String name) {
        return invoke.name().equalsIgnoreCase(name);
    }

    private void handleVectorOperations(Invoke invoke) {
        IfaceValue.Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
        if (isVectorNamed(invoke, "float4view") || isVectorNamed(invoke, "float2view")) {
            // could be local(shared) or global
            boolean isSharedOrPrivate = isSharedOrPrivate(invoke.lookup(), invoke.op());
            Value source = invoke.op().operands().getFirst();
            Value index = invoke.op().operands().get(1);
            generateVectorLoad(source, index, vectorShape, isSharedOrPrivate);
        } else if (isVectorNamed(invoke, "of")) {
            generateVectorOf(invoke.op(), vectorShape);
        } else if (isVectorNamed(invoke, "makeMutable")) {
            var name = findVectorVarNameOrNull(invoke.op().operands().getFirst());
            id(name);
        } else if (isVectorBinaryOperation(invoke)) {
            handleVectorBinaryOperation(invoke);
        } else {
            throw new IllegalStateException("[CodeGen] Vector Operation not found: " + invoke.name());
        }
    }

    private void handleVectorView(Invoke invoke) {
        IfaceValue.Vector.Shape vectorShape = getVectorShapeFromOperandN(invoke.lookup(), invoke.op(), 1);
        boolean isShared = findIsSharedOrPrivateSpace(invoke.op().operands().getFirst());
        String vectorName = findVectorVarNameOrNull(invoke.op().operands().get(1));
        hatVectorStoreOp(invoke.op().operands().get(0), invoke.op().operands().get(2), vectorShape, isShared, vectorName, invoke.op());
    }

    private void handleVectorSelect(Invoke invoke) {
        InvokeVar invokeVar = new InvokeVar(invoke.op(), invoke.varLoadOpFromFirstOperandOrNull());
        if (invoke.returnsVoid()) {
            hatSelectStoreOp(invoke, invokeVar);
        } else {
            hatSelectLoadOp(invoke, invokeVar);
        }
    }

    private void handleFloatToS16Conversion(Invoke invoke) {
        if (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) instanceof Class<? extends S16ImplOfF16> reducedFloatType) {
            hatF16ConvOp(invoke.op(), reducedFloatType);
        } else {
            throw new IllegalStateException("[CodeGen] Unhandled op: " + invoke.name());
        }
    }

    private void handleS16ToFloatConversion(Invoke invoke) {
        // s16 type -> float conversion
        // Obtain the reducedFLoatType
        if (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) instanceof Class<? extends S16ImplOfF16> reducedFloatType) {
            boolean isF16Local = isF16Local(invoke.op().operands().getFirst());
            boolean wasFloat = invoke.opFromFirstOperandOrNull() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp && varLoadOp.resultType().equals(JavaType.FLOAT);
            // generate
            hatF16ToFloatConvOp(invoke, reducedFloatType, wasFloat, isF16Local);
        } else {
            throw new IllegalStateException("[CodeGen] No reduced float type");
        }
    }

    private void handleS16BinaryOperation(Invoke invoke) {
        if (S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.refType()) instanceof Class<? extends S16ImplOfF16> category) {
            // generate
            hatF16BinaryOp(invoke, category);
        } else {
            throw new IllegalStateException("[CodeGen] Unhandled op: " + invoke.name());
        }
    }

    private void handleVectorBinaryOperation(Invoke invoke) {
        hatBinaryVectorOp(invoke);
    }

    private void handleTensorOperation(Invoke invoke) {
        switch (invoke.name()) {
            case "create" -> hatTensorCreateOperation(invoke);
            case "zeros" -> hatTensorCreateOperation(invoke);
            case "fill" -> hatTensorFill(invoke);
            default -> throw new IllegalStateException("[CodeGen] Unknown op: " + invoke.name());
        }
    }

    private void handleIFaceInvoke(Invoke invoke) {
        if (invoke instanceof Invoke.Virtual && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex(atomicIncRegex)) {
            if (invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instanceResult) {
                atomicInc(instanceResult,
                        ((Regex.Match) atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                );
            }
        } else if (isInvokeLoadingFromOnChipMemory(scopedCodeBuilderContext().lookup(), invoke.op())) {
            // equivalent to custom ops for private and shared memory
            generateOnChipMemoryLoad(invoke.op());
        } else if (invoke instanceof Invoke.Virtual && invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instance) {
            // Attention: Since F16.toFloat operations are supported, it should be possible to
            // implement a load from global memory from an F16Array and directly use it for a math operation.
            // In this case, we need to add an extra parenthesis.
            boolean narrowTypeCast = isInvokeFromNarrowTypeConversion(scopedCodeBuilderContext.lookup(), invoke.op());
            parenWhen(narrowTypeCast, _ -> {
                parenWhen(
                        invoke.operandCount() > 1
                                && invoke(scopedCodeBuilderContext().lookup(), instance.op()) instanceof Invoke invoke0
                                && invoke0.returnsClassType()
                        ,
                        // When we have patterns like:
                        //
                        // myiFaceArray.array().value(storeAValue);
                        //
                        // We need to generate extra parenthesis to make the struct pointer accessor "->" correct.
                        // This is a common pattern when we have a IFace type that contains a subtype based on
                        // struct or union.
                        // An example of this is for the type F16Array.
                        // The following expression checks that the current invokeOp has at least 2 operands:
                        // Why 2?
                        // - The first one is another invokeOp to load the inner struct from an IFace data structure.
                        //   The first operand is also assignable.
                        // - The second one is the store value, but this depends on the semantics and definition
                        //   of the user code.
                        _ -> {
                            when(invoke.returnsClassType(), _ -> ampersand());
                            recurse(instance.op());
                        });

                // Check if the varOpLoad that could follow corresponds to a local/private type
                // We need to check for the HATMemoryVarOp until we replace all HAT<>VarOps with CoreOp.VarOp
                boolean isLocalOrPrivateDS = isInvokeFromSharedOrPrivate(instance, invoke);
                dotOrArrow(isLocalOrPrivateDS);

                funcName(invoke.op());

                if (invoke.returnsVoid()) {//   setter
                    switch (invoke.operandCount()) {
                        case 2 -> {
                            if (invoke.opFromOperandNOrNull(1) instanceof Op op) {
                                equals().recurse(op);
                            }
                        }
                        case 3 -> {
                            if (invoke.opFromOperandNOrThrow(1) instanceof Op op1
                                    && invoke.opFromOperandNOrThrow(2) instanceof Op op2) {
                                sbrace(_ -> recurse(op1)).equals().recurse(op2);
                            }
                        }
                        default -> throw new IllegalStateException("How ");
                    }
                } else if (invoke.opFromOperandNOrNull(1) instanceof Op op) {
                    sbrace(_ -> recurse(op));
                }
            });
        }
    }

    private boolean isIFaceValue(Invoke invoke) {
        return invoke.refIs(IfaceValue.class);
    }

    @Override
    public final T invokeOp(JavaOp.InvokeOp invokeOp) {
        MethodHandles.Lookup lookup = scopedCodeBuilderContext().lookup();
        var invoke = invoke(lookup, invokeOp);
        if (isVecInvoke(invoke)) { // hacked for vec op calls.
            handleInvoke(self(), invoke);
        } else if (isVectorOperation(lookup, invokeOp)) {
            handleVectorOperations(invoke);
        } else if (isVectorView(lookup, invokeOp)) {
            handleVectorView(invoke);
        } else if (isVectorSelectOperation(invoke)) {
            handleVectorSelect(invoke);
        } else if (isS16Conversion(invoke)) {
            handleFloatToS16Conversion(invoke);
        } else if (isS16ToFloatConversion(invoke)) {
            handleS16ToFloatConversion(invoke);
        } else if (isS16BinaryOp(invoke)) {
            handleS16BinaryOperation(invoke);
        } else if (isTensorOperation(invoke)) {
            handleTensorOperation(invoke);
        } else if (isIFaceValue(invoke)) {
            handleIFaceInvoke(invoke);
        } else if (isMathOperation(invoke)) {
            generateMathIntrinsicOperation(invoke);
        } else {
            generateFunctionCall(invoke);
        }
        return self();
    }

    private void generateFunctionCall(Invoke invoke) {
        funcName(invoke.op()).paren(_ ->
                commaSpaceSeparated(invoke.op().operands(),
                        op -> {
                            if (op instanceof Op.Result result) {
                                recurse(result.op());
                            }
                        })
        );
    }

    private VarTable.HATOpAttribute getDeviceRegion(CoreOp.VarOp varOp) {
        return varTable.getAttributeOrThrow(scopedCodeBuilderContext.funcOp().funcName(), varOp);
    }

    private void genericVarOp(CoreOp.VarOp varOp) {
        // Original varOp
        if (scopedCodeBuilderContext().isVarOpFinal(varOp)) {
            constKeyword().sp();
        }
        type((JavaType) varOp.varValueType()).sp().varName(varOp).sp().equals().sp();
        var first = varOp.operands().getFirst();
        switch (first) {
            case Op.Result result -> parenthesisIfNeeded(varOp, result.op());
            case Block.Parameter parameter -> {
                // for debugging
                var r = parameter.uses().iterator().next();
                blockInlineComment("param " + r);
            }
            // for debugging
            default -> blockInlineComment("look at varOp " + first);
        }
    }

    @Override
    public T varOp(CoreOp.VarOp varOp) {
        if (varOp.isUninitialized()) {
            type((JavaType) varOp.varValueType()).sp().varName(varOp);
        } else {
            // First, we look at the attribute table for each varOp
            VarTable.HATOpAttribute attribute = getDeviceRegion(varOp);

            // If attribute exits, we apply codegen based on attribute since there is a pre-search and
            // categorization about the corresponding OpenCL code to be generated.
            switch (attribute) {
                case NARROW -> varOpForNarrowType(varOp);
                case VECTOR -> varOpForVectors(varOp);
                case INIT_SHARED -> varOpInit(varOp);
                case SHARED -> varOpLocalMemory(varOp);
                case PRIVATE -> varOpPrivateMemory(varOp);
                case TENSOR -> varOpTensor(varOp);
                case null -> genericVarOp(varOp);
                default -> throw new IllegalStateException("Unexpected HATOpAttribute: " + attribute);
            }
        }
        return self();
    }

    @Override
    public T arrayLoadOp(JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp) {
        if (HATArrayViewPhase.isVectorOp(scopedCodeBuilderContext.lookup(), arrayLoadOp)) {
            var vectorShape = getVectorShape(scopedCodeBuilderContext.lookup(), arrayLoadOp.resultType());
            boolean deviceAllocated = HATArrayViewPhase.isLocalSharedOrPrivate(arrayLoadOp);
            // generate
            generateVectorLoad(arrayLoadOp.arrayOperand(), arrayLoadOp.indexOperand(), vectorShape, deviceAllocated);
        } else {
            recurse(((Op.Result) arrayLoadOp.operands().get(0)).op());
            sbrace(_ -> recurse(((Op.Result) arrayLoadOp.operands().get(1)).op()));
        }
        return self();
    }

    @Override
    public T arrayStoreOp(JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp) {
        if (HATArrayViewPhase.isVectorOp(scopedCodeBuilderContext.lookup(), arrayStoreOp)) {
            Op varOp = opFromFirstOperandOrNull(((Op.Result) arrayStoreOp.operands().getLast()).op());
            String name = HATArrayViewPhase.hatPtrName(varOp);
            var vectorShape = getVectorShape(scopedCodeBuilderContext.lookup(), arrayStoreOp.operands().getLast().type());
            boolean deviceAllocated = HATArrayViewPhase.isLocalSharedOrPrivate(arrayStoreOp);
            Op.Result dest = OpHelper.resultFromFirstOperandOrThrow(arrayStoreOp);
            Value index = arrayStoreOp.operands().get(1);
            hatVectorStoreOp(dest, index, vectorShape, deviceAllocated, name, arrayStoreOp);
        } else {
            recurse(((Op.Result) arrayStoreOp.operands().get(0)).op());
            sbrace(_ -> recurse(((Op.Result) arrayStoreOp.operands().get(1)).op()));
            sp().equals().sp();
            recurse(((Op.Result) arrayStoreOp.operands().get(2)).op());
        }
        return self();
    }

    protected void genFieldAccess(Value operand, boolean isReference) {
        if (isReference) {
            rarrow().id(VALUE);
        } else if (!OpHelper.isPrimitiveResult(operand)) {
            dot().id(VALUE);
        }
    }

    private Class<?> isNarrowType(Invoke invoke) {
        if (invoke.returnsClassType() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke, (ClassType) invoke.returnType()) instanceof Class<? extends S16ImplOfF16> float16Type) {
            return float16Type;
        }
        return null;
    }

    private void generateMathIntrinsicOperation(Invoke invoke) {
        // if the resulting type is a narrowed-type (e.g., bfloat16, or half float)
        Class<?> float16Class = isNarrowType(invoke);
        if (float16Class != null) {
            paren(_ ->
                    f16OrBF16(float16Class))
                    .brace(_ -> {
                        id(mapMathIntrinsic(invoke.name()));
                        // For each operand, obtain if it is a reference from global memory or device memory.
                        List<Boolean> referenceList = invoke.op()
                                .operands()
                                .stream()
                                .map(value -> isArrayReference(scopedCodeBuilderContext.lookup(), value))
                                .toList();
                        paren(_ -> {
                            int[] counter = {0};
                            commaSpaceSeparated(invoke.op().operands(), op -> {
                                recurse(OpHelper.asResultOrThrow(op).op());
                                genFieldAccess(op, referenceList.get(counter[0]++));
                            });
                        });
                    });
        } else {
            id(mapMathIntrinsic(invoke.name()));
            paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(), op ->
                            recurse(OpHelper.asResultOrThrow(op).op())
                    )
            );
        }
    }

    public abstract T defines();

    protected abstract T atomicInc(Op.Result instanceResult, String name);

    protected List<String> getMacroVectorParamsLoad() {
        return List.of(N, ADDDR, INDEX, IS_LOCAL);
    }

    protected List<String> getMacroVectorParamsStore() {
        return List.of(N, ADDDR, INDEX, IS_LOCAL, VECTOR_VAL);
    }

    protected T generateVectorLoad(Value source, Value index, IfaceValue.Vector.Shape vectorShape, boolean deviceAllocated) {
        return id(VLOADN).paren(_ ->
                id(String.valueOf(vectorShape.lanes()))
                        .comma()
                        .recurseResultOrThrow(source)
                        .comma()
                        .paren(_ -> recurseResultOrThrow(index))
                        .comma()
                        .either(deviceAllocated, _ -> intConstOne(), _ -> intConstZero()));
    }

    protected T hatVectorStoreOp(Value dest, Value index, IfaceValue.Vector.Shape vectorShape, boolean deviceAllocated, String name, Op op) {
        return id(VSTOREN).paren(_ ->
                id(String.valueOf(vectorShape.lanes()))
                .comma()
                .recurseResultOrThrow(dest)
                .comma()
                .paren(_ -> recurseResultOrThrow(index))
                .comma()
                .either(deviceAllocated, _ -> intConstOne(), _ -> intConstZero())
                .comma()
                .id(name));
    }

    protected T hatSelectStoreOp(OpHelper.Invoke invoke, InvokeVar invokeVar) {
        return id(VSELECT_STORE).paren( _-> {
            if (invoke.op().operands().getFirst().declaringElement() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp vLoadOp) {
                recurse(vLoadOp);
            } else {
                id(invokeVar.name());
            }
            comma().id(HATPhaseUtils.mapLane(invokeVar.laneIdx())).comma();
            String resolvedName = invokeVar.resolveName();
            either (resolvedName != null,
                    _-> varName(resolvedName),
                    _-> recurseResultOrThrow(invoke.op().operands().get(1))
            );
        });
    }

    protected T hatSelectLoadOp(OpHelper.Invoke invoke, InvokeVar invokeVar) {
        return id(VSELECT_LOAD).paren( _ -> {
            if (invoke.op().operands().getFirst().declaringElement() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp vLoadOp) {
                recurse(vLoadOp);
            } else {
                id(invokeVar.name());
            }
            comma().id(mapLane(invokeVar.laneIdx()));
        });
    }

    protected T hatF16ConvOp(JavaOp.InvokeOp invokeOp, Class<?> reducedFloatType) {
        return either(F16.class.isAssignableFrom(reducedFloatType),
                _ -> id(F16_OF),
                _ -> id(BF16_OF))
                .paren(_ -> recurseResultOrThrow(invokeOp.operands().getFirst()));
    }

    protected String obtainMacroForS16ToFloatConversion(Class<?> reducedFloatType, boolean isF16Local) {
        if (F16.class.isAssignableFrom(reducedFloatType)) {
            if (isF16Local) {
                return F16_TO_FLOAT_1;
            }  else {
                return F16_TO_FLOAT_0;
            }
        } else {
            if (isF16Local) {
                return BF16_TO_FLOAT_1;
            } else  {
                return BF16_TO_FLOAT_0;
            }
        }
    }

    protected T hatF16ToFloatConvOp(Invoke invoke, Class<?> reducedFloatType, boolean wasFloat, boolean isF16Local) {
        String macro = obtainMacroForS16ToFloatConversion(reducedFloatType, isF16Local);
        id(macro).paren(_ -> recurseResultOrThrow(invoke.op().operands().getFirst()));
        return self();
    }

    protected List<Integer> obtainShapeTensor(Value shapeValue) {
        List<Integer> shape = new ArrayList<>();
        obtainShapeTensor(shapeValue, shape);
        if (shape.size() != 3) {
            throw new IllegalStateException("Shape must have three values, but it has " + shape.size());
        }
        return shape;
    }

    protected List<Integer> getShapeFromTensorVarOp(CoreOp.VarOp tensorVarOp) {
        Value tensorCreateValueOp = tensorVarOp.operands().getFirst();
        if (tensorCreateValueOp.declaringElement() instanceof JavaOp.InvokeOp tensorCreateOp) {
            // First parameter: shapeValue
            Value valueShape = tensorCreateOp.operands().getFirst();
            return obtainShapeTensor(valueShape);
        } else {
            throw new IllegalStateException("Expected an InvokeOp, but found: " + tensorCreateValueOp.declaringElement().getClass());
        }
    }

    protected List<Integer> getShapeFromTensorCreateValue(Value tensorCreateValue) {
        if (tensorCreateValue.declaringElement() instanceof JavaOp.InvokeOp tensorInvokeOp) {
            // The first parameter is the shape -> analysis of the shape
            return obtainShapeTensor(tensorInvokeOp.operands().getFirst());
        } else {
            throw new IllegalStateException("Expected an InvokeOp, but found: " + tensorCreateValue.declaringElement().getClass());
        }
    }

    protected List<Integer> processShapeTensor(List<Value> shapeOperands, List<Integer> shape) {
        for (Value shapeOperand : shapeOperands) {
            while (!(shapeOperand.declaringElement() instanceof CoreOp.ConstantOp)) {
                if (shapeOperand.declaringElement() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
                    shapeOperand = varLoadOp.varOperand();
                } else if (shapeOperand.declaringElement() instanceof CoreOp.VarOp varOp) {
                    shapeOperand = varOp.operands().getFirst();
                }
            }
            if (shapeOperand.declaringElement() instanceof CoreOp.ConstantOp constantOp) {
                shape.add((int) constantOp.value());
            } else {
                throw new IllegalStateException("Error: expected to find a ConstantOp, but found a " + shapeOperand.declaringElement().getClass());
            }
        }
        return shape;
    }

    protected List<Integer> obtainShapeTensor(Value shapeValue, List<Integer> shape) {
        switch (shapeValue.declaringElement()) {
            case JavaOp.InvokeOp invokeOp when invokeOp.invokeReference().name().equals("shape") -> {
                List<Value> shapeOperands = invokeOp.operands();
                return processShapeTensor(shapeOperands, shape);
            }
            case CoreOp.VarAccessOp varAccessOp -> obtainShapeTensor(varAccessOp.varOperand(), shape);
            case CoreOp.VarOp varOp -> obtainShapeTensor(varOp.operands().getFirst(), shape);
            case HATTensorOp.TensorShapeOp tensorShapeOp -> {
                return processShapeTensor(tensorShapeOp.operands(), shape);
            }
            default ->
                    throw new IllegalStateException("Op not expected: Found: " + shapeValue.declaringElement().getClass());
        }
        return shape;
    }

    protected T indexForTensor(boolean isColumnMajor, Value iIndex, Value jIndex, Value ldSize) {
        Value a = isColumnMajor ? iIndex : jIndex;
        Value b = isColumnMajor ? jIndex : iIndex;

        if (a instanceof Op.Result r) {
            recurse(r.op());
        }
        plus();
        oparen();
        if (b instanceof Op.Result r) {
            recurse(r.op());
        }
        mul();
        if (ldSize instanceof Op.Result r) {
            recurse(r.op());
        }
        cparen();
        return self();
    }

    protected boolean isColumnMajor(Value tensorLayout) {
        if (tensorLayout.declaringElement() instanceof JavaOp.InvokeOp invokeOp) {
            var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
            if (invoke.resultTypeIs(Tensor.ColumMajor.class)) {
                return true;
            } else if (invoke.resultTypeIs(Tensor.RowMajor.class)) {
                return false;
            } else {
                throw new RuntimeException("[Error]");
            }
        }
        return false;
    }

    @Override
    public T hatTensorShapeOp(HATTensorOp.TensorShapeOp tensorShapeOp) {
        return self();
    }


    protected abstract T hatBinaryVectorOp(OpHelper.Invoke binOp);

    protected abstract T varOpForNarrowType(CoreOp.VarOp varOp);

    protected abstract T varOpForVectors(CoreOp.VarOp varOp);

    protected abstract T varOpInit(CoreOp.VarOp varOp);

    protected abstract T varOpLocalMemory(CoreOp.VarOp varOp);

    protected abstract T varOpPrivateMemory(CoreOp.VarOp varOp);

    protected abstract T varOpTensor(CoreOp.VarOp varOp);

    protected abstract T hatTensorCreateOperation(Invoke invoke);

    protected abstract T hatTensorFill(Invoke invoke);

    protected abstract String mapMathIntrinsic(String name);

    protected abstract T hatWarpSize();

}
