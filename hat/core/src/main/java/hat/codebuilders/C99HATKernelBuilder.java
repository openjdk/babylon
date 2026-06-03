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

import hat.HATMath;
import hat.KernelContext;
import hat.buffer.BF16Array;
import hat.callgraph.KernelCallGraph;
import hat.device.NonMappableIface;
import hat.dialect.HATBarrierOp;
import hat.dialect.HATPtrOp;
import hat.dialect.HATThreadOp;
import hat.phases.HATArrayViewPhase;
import hat.phases.HATFP16Phase;
import hat.types.BF16;
import hat.types.F16;
import jdk.incubator.code.Block;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.dialect.core.VarType;
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
import optkl.codebuilders.CodeBuilder;

import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.Optional;
import java.util.SequencedSet;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static hat.buffer.F16Array.F16Impl;
import static java.lang.invoke.MethodHandles.lookup;
import static optkl.IfaceValue.Vector.getVectorShape;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke.invoke;

import static optkl.OpHelper.VarAccess.varAccess;
import static optkl.OpHelper.opFromFirstOperandOrNull;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilder<T> implements HATOpDispatcher<T> {
    protected  final KernelCallGraph kernelCallGraph;
    private final VarTable varTable;

    protected static boolean isOperandF32(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp varLoadOp -> varLoadOp.varType().valueType() == JavaType.FLOAT; //recurse
            case CoreOp.VarOp varOp -> varOp.resultType().valueType() == JavaType.FLOAT;
            default -> false;
        };
    }

    //recursive
    protected  boolean isArrayReference(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isArrayReference(varLoadOp); // recurse
            case CoreOp.VarOp varOp ->
                    varOp.operands().getFirst() instanceof Op.Result varOpResult
                            && invoke(lookup(),varOpResult.op()) instanceof OpHelper.Invoke invoke
                            && invoke.named("array")
                            && !isInvokeLoadingFromOnChipMemory(invoke.op());
            default -> false;
        };
    }

    //recursive
    private  boolean isArrayReference( CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isArrayReference(varLoadOp.operands().getFirst());
    }

    protected C99HATKernelBuilder(KernelCallGraph kernelCallGraph, ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
        this.kernelCallGraph = kernelCallGraph;
        this.varTable = kernelCallGraph.getVarTable();
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

    @Override
    public final T hatThreadIdOp( HATThreadOp threadOp) {
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
        });
    }

    public final T kernelDeclaration(CoreOp.FuncOp funcOp) {
        return HAT_KERNEL().sp().voidType().sp().funcName(funcOp);
    }

    public final  T functionDeclaration( JavaType javaType, CoreOp.FuncOp funcOp) {
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
                                                    .filter(a->a.field.equals(ifaceField))
                                                    .findFirst()
                                                    .ifPresentOrElse(
                                                            a-> sbrace(_ -> literal(a.len)),
                                                            ()->{
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
                            u08Type().sp().identifierWithRandomSuffix("pad$",5).sbrace(_->intValue((int)(padding.len)));//; emitText(toC99(padding));
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
    public  final  T identifierWithRandomSuffix(String prefix, final int len) {
        var sb = new StringBuilder();
        final var LEGAL_CHARS = "_$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        ThreadLocalRandom.current() //
                .ints(len, 0, LEGAL_CHARS.length()) //
                .mapToObj(LEGAL_CHARS::charAt) //
                .forEach(sb::append);
        id(prefix+sb);
        return self();
    }

    @Override
    public final T hatBarrierOp(HATBarrierOp barrierOp) {
        return HAT_BARRIER();
    }

    public abstract T defines();

    public final  T types() {
        return
                 typedefKeyword().sp().s08Type("byte").snl()
                .typedefKeyword().sp().s08Type("boolean").snl()
                .typedefStruct(KernelContext.class, _ -> s32Type("dimensions").semicolon()).nl();
    }

    @Override
    public final T fieldLoadOp( JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        var fieldAccess = fieldAccess(scopedCodeBuilderContext().lookup(),fieldLoadOp);
        if (fieldAccess.operandCount()==0 && fieldAccess.isPrimitive()) {
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
    public final  T type( JavaType javaType) {
        // TODO: I would not include this method in the visitor, rather just the tree-nodes. Types can be resolved when needed directly.

        if (C99VecAndMatHandler.isVecOrMatType(scopedCodeBuilderContext().lookup(),javaType)){
            C99VecAndMatHandler.handleType(self(),javaType);
        } else if (javaType instanceof ClassType classType
                && OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, IfaceValue.class)
                && !OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, S16ImplOfF16.class)
        ) {
            HAT_GLOBAL_MEM().sp().suffix_t(classType).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, KernelContext.class)) {
            HAT_GLOBAL_MEM().sp().suffix_t(KernelContext.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType,F16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().sp().suffix_t(F16Impl.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType,BF16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().sp().suffix_t(BF16Array.BF16Impl.class).asterisk();
        } else {
            type(javaType.toString());
        }
        return self();
    }


    public final  T kernelMethod(CoreOp.FuncOp funcOp) {
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

    public final  T kernelEntrypoint() {
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

    protected boolean isMathLib(Optional<Invoke> invoke) {
        return invoke.isPresent() && !invoke.get().returnsVoid() && invoke.get().returnsClassType() && invoke.get().refIs(HATMath.class);
    }

     protected T f16OrBF16(Class<?> float16Class){
        if (F16.class.isAssignableFrom(float16Class)){
            return f16Type();
        }else if (BF16.class.isAssignableFrom(float16Class)){
            return bf16Type();
        }else {
            throw new IllegalStateException("Unexpected value: " + float16Class);
        }
    }

    protected boolean isMixedFirstOperand(byte f32Mixed) {
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

    public static final String VALUE = "value";



    private T binaryOperationsForBfloat16(Invoke invoke) {

        boolean isFirstOperandReference = isArrayReference(invoke.op().operands().get(0));
        boolean isSecondOperandReference = isArrayReference(invoke.op().operands().get(1));
        final byte f32Mixed;
        if (!isFirstOperandReference && isOperandF32(invoke.op().operands().get(0))) {
            f32Mixed = HATFP16Phase.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(invoke.op().operands().get(1))) {
            f32Mixed = HATFP16Phase.LAST_OP;
        } else {
            f32Mixed = 0x00;
        }

        paren(_-> bf16Type());
        brace(_-> {
            paren(_-> {
                builtin_float2bfloat16();
                oparen();
                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();// open
                }
                recurse( OpHelper.asResultOrThrow(invoke.op().operands().getFirst()).op());

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
        paren(_-> f16Type());
        return brace(_->
                paren(_-> {
                    recurse( OpHelper.asResultOrThrow(invoke.op().operands().getFirst()).op());
                    boolean isFirstOperandReference = isArrayReference(invoke.op().operands().get(0));
                    boolean isSecondOperandReference = isArrayReference(invoke.op().operands().get(1));
                    if (isFirstOperandReference) {
                        rarrow().id(VALUE);
                    } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().getFirst())) {
                        dot().id(VALUE);
                    } else {
                        blockComment("hatF16BinaryOp not a result !!");
                    }
                    sp().id(matchSymbol(invoke.name())).sp();
                    recurse( OpHelper.asResultOrThrow(invoke.op().operands().get(1)).op());
                    if (isSecondOperandReference) {
                        rarrow().id(VALUE);
                    } else if (!OpHelper.isPrimitiveResult(invoke.op().operands().get(1))) {
                        dot().id(VALUE);
                    }else {
                        blockComment("hatF16BinaryOp not a value !!");
                    }
                })
        );
    }

    private String findVarNameOrNull(Value v) {
        return (v instanceof Op.Result r) ? switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findVarNameOrNull(varLoadOp); //recurse
            case CoreOp.VarOp varOp -> varOp.varName();
            default -> null;
        } : null;
    }

    private String findVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVarNameOrNull(varLoadOp.operands().getFirst());
    }

    public final T hatF16VarLoadOp(CoreOp.VarAccessOp.VarLoadOp hatF16VarLoadOp) {
        id(findVarNameOrNull(hatF16VarLoadOp));

        // Since all VarOps now are the same, we need to distinguish if it comes from a global load,
        // or private/shared load.
        if (hatF16VarLoadOp.operands().getFirst().declaringElement() instanceof CoreOp.VarOp varOp
                && varOp.operands().getFirst().declaringElement() instanceof JavaOp.InvokeOp invokeOp
                && !isInvokeLoadingFromOnChipMemory(invokeOp)) {
            // VarLoad from Global Memory with an InvokeOp
            Stream<Invoke> stream = OpHelper.Invoke.stream(kernelCallGraph.lookup(), invokeOp);
            Optional<OpHelper.Invoke> invoke = stream.findFirst();
            if (isMathLib(invoke)) {
                dot();
            } else if (invoke.isPresent() && invoke.get().refIs(S16ImplOfF16.class)) {
                // This extra condition is due to all operation are implemented via invoke.
                // Thus, we need to make sure the type to know if it is a reference from
                // global memory.
                dot();
            } else {
                rarrow();
            }
        } else {
            dot();
        }
        return id(VALUE);
    }

    public abstract T genVectorIdentifier(IfaceValue.Vector.Shape vectorShape);

    public final T generateVectorOf(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape) {
        return genVectorIdentifier(vectorShape)
                .paren(_ -> commaSpaceSeparated(invokeOp.operands(),operand -> recurse(OpHelper.asResultOrThrow(operand).op())));
    }

    public final T generateOnChipMemoryLoad(JavaOp.InvokeOp invoke) {
        return recurse(OpHelper.asResultOrThrow(invoke.operands().getFirst()).op())
                .dot().id(invoke.invokeReference().name())
                .when(invoke.operands().size() > 1,_-> // If the hatMemoryLoadOp has more than 1 operand, the second is the index
                        sbrace(_-> recurse( OpHelper.asResultOrThrow(invoke.operands().get(1)).op()))
                );
    }

    public final T hatPtrLoadOp( HATPtrOp.HATPtrLoadOp hatPtrLoadOp) {
        ptrAccess(hatPtrLoadOp);
        return self();
    }

    @Override
    public final T hatPtrStoreOp( HATPtrOp.HATPtrStoreOp hatPtrStoreOp) {
        ptrAccess(hatPtrStoreOp).equals().recurse( ((Op.Result) hatPtrStoreOp.operands().getLast()).op());
        return self();
    }

    @Override
    public final  T hatPtrLengthOp( HATPtrOp.HATPtrLengthOp hatPtrLengthOp) {
        ptrAccess(hatPtrLengthOp);
        return self();
    }

    private static final Set<String> NON_MAPPABLE_IFACE = Set.of("createshared", "createlocal", "createprivate");

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

        either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

        if (hatPtrOp instanceof HATPtrOp.HATPtrLengthOp) {
            id("length");
        } else {
            final boolean finalIsLocalOrPrivateDS = isLocalOrPrivateDS;
            id("array").sbrace(_ -> {
                paren(_ -> id("long")); // is this a cast (long)  maybe cast(_->typeName("long"))?
                paren(_ -> {
                    if (hatPtrOp.strides().size() > 1) {
                        paren(_ -> recurse(((Op.Result) hatPtrOp.operands().get(2)).op()));
                        asterisk().id(hatPtrOp.name());
                        either(finalIsLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow).id(hatPtrOp.strides() != null ? hatPtrOp.strides().getFirst() : "width");
                        add().paren(_ -> recurse( ((Op.Result) hatPtrOp.operands().get(1)).op()));
                    } else {
                        recurse( ((Op.Result) hatPtrOp.operands().get(1)).op());
                    }
                });
            });
        }
        return self();
    }

    /**
     * <code>
     *  float bfloat16Tofloat(ushort bf16) {
     *      b16_t b;
     *      b.s[0] = 0;
     *      b.s[1] = s;
     *      return b.f;
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
                               .returnKeyword(_-> id(b16).dot().id(f)));
    }

    /**
     * <code>
     * ushort floatTobfloat16(float f) {
     *      b16_t b = {f};
     *      uint32_t bits = b.i;
     *      short sign_bit = (short)((bits & 0x8000_0000) >> 16);
     *      int lsb    = bits & 0x1_0000;
     *      int round  = bits & 0x0_8000;
     *      int sticky = bits & 0x0_7FFF;
     *      if (round != 0 && ((lsb | sticky) != 0 )) {
     *          bits += 0x1_0000;
     *      }
     *      return (short) (((bits >> 16 ) | sign_bit) & 0xffff);
     * }
     * </code>
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
                        _ ->  brace( _ -> id(parameterName)).snl())
                        .assign( _ -> u32Type("bits"), _ -> id(idBFloat16).dot().id("i")).snl()
                        .assign( _ -> u16Type("sign_bit"), _ -> cast( _ -> s16Type()).paren( _ -> paren( _ -> id("bits").ampersand().constant("0x80000000")).rightShift(16))).snl()
                        .assign( _ -> s32Type("lsb"), _ -> id("bits").ampersand().constant("0x10000")).snl()
                        .assign( _ -> s32Type("round"), _ -> id("bits").ampersand().constant("0x08000")).snl()
                        .assign( _ -> s32Type("sticky"), _ -> id("bits").ampersand().constant("0x07FFF")).snl()
                        .ifTrueCondition(_ -> id("round").sp().ne().sp().intConstZero().condAnd().paren(_ -> paren(_ -> id("lsb").bitwiseOR().id("sticky")).ne().intConstZero()),
                                _ -> id("bits").sp().plusEquals().sp().constant("0x10000"))
                        .returnKeyword( _ -> cast( _ ->u16Type()).paren( _ -> paren( _ -> paren( _-> id("bits").rightShift(16)).bitwiseOR().id("sign_bit")).ampersand().constant("0xffff"))));
    }

    @Override
    public final T varLoadOp(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp $ -> varName($);
            case null, default -> {}
        }
        return self();
    }

    @Override
    public final T varStoreOp( CoreOp.VarAccessOp.VarStoreOp varStoreOp) {
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
        equals().parenthesisIfNeeded( varStoreOp, ((Op.Result)varStoreOp.operands().get(1)).op());
        return self();
    }

    @Override
    public final  T convOp( JavaOp.ConvOp convOp) {
        // TODO: I think we need to work out how to handle doubles. If I remove this OpenCL on MAC complains (no FP64)
        if (convOp.resultType() == JavaType.DOUBLE) {
            paren(_ -> type(JavaType.FLOAT)); // why double to float?
        } else {
            paren(_ -> type((JavaType)convOp.resultType()));
        }
        parenthesisIfNeeded( convOp, ((Op.Result) convOp.operands().getFirst()).op());
        return self();
    }

    public abstract  T atomicInc( Op.Result instanceResult, String name);

    static Regex atomicIncRegex = Regex.of("(atomic.*)Inc");

    private boolean isAttributeSharedOrPrivate(VarTable.HATOpAttribute attribute, Invoke invoke) {
        if (attribute == VarTable.HATOpAttribute.INIT_SHARED || attribute == VarTable.HATOpAttribute.PRIVATE || attribute == VarTable.HATOpAttribute.SHARED) {
            return true;
        } else return attribute == VarTable.HATOpAttribute.NARROW && !invoke.returnsVoid();
    }

    private boolean isInvokeLoadingFromOnChipMemory(JavaOp.InvokeOp invokeOp) {
        Invoke invoke = invoke(scopedCodeBuilderContext.lookup(), invokeOp);
        if (invoke.refIs(NonMappableIface.class) && invoke.returnsClassType() && !invoke.nameMatchesRegex(OpHelper.RESERVED_METHODS_MEMORY_REGIONS)) {
            SequencedSet<Op.Result> uses = invoke.op().result().uses();
            return uses.stream().filter(use -> use.op() instanceof CoreOp.VarOp)
                    .map(use -> (CoreOp.VarOp) use.op()).anyMatch(_ -> true);
        }
        return false;
    }

    private boolean isVectorOperation(JavaOp.InvokeOp invokeOp) {
        Invoke invoke = invoke(scopedCodeBuilderContext.lookup(), invokeOp);
        return invoke.returns(IfaceValue.Vector.class) && invoke.nameMatchesRegex(OpHelper.RESERVED_METHOD_VECTORS);
    }

    private boolean isSharedOrPrivate(MethodHandles.Lookup lookup, Op op) {
        return isSharedOrPrivate(lookup, op.operands().getFirst());
    }

    public boolean isSharedOrPrivate(MethodHandles.Lookup lookup, Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isSharedOrPrivate(lookup, varLoadOp); //recurse
            case CoreOp.VarOp varOp -> {
                // extra analysis
                Value first = varOp.operands().getFirst();
                if (first instanceof Block.Parameter) {
                    // if the var comes from a parameter, then it is global memory
                    yield false;
                }
                // otherwise we continue traversal
                yield isSharedOrPrivate(lookup, varOp);
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
                    yield NON_MAPPABLE_IFACE.contains(lowerCase);
                }
                yield false;
            }
            default -> false;
        };
    }

    public abstract T generateVectorLoad(Value source, Value index, IfaceValue.Vector.Shape vectorShape,  boolean deviceAllocated);

    // recursive
    public static String findVectorVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorVarNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    public static String findVectorVarNameOrNull(Value v) {
        switch (v) {
            case Op.Result r when r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp -> {
                return findVectorVarNameOrNull(varLoadOp);
            }
            case null, default -> {
                if (v instanceof CoreOp.Result r && r.op() instanceof CoreOp.VarOp varOp) {
                    return varOp.varName();
                }
                return null;
            }
        }
    }

    private boolean isVectorView(JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
        return (invoke.named("storeFloat4View") || invoke.named("storeFloat2View"))
                && varAccess(scopedCodeBuilderContext.lookup(), invoke.opFromOperandNOrNull(1)) instanceof OpHelper.VarAccess varAccess
                && varAccess.isLoad() && varAccess.isTypeAssignable(IfaceValue.Vector.class);
    }

    public abstract T hatVectorStoreOp(JavaOp.InvokeOp invokeOp, IfaceValue.Vector.Shape vectorShape, String name, boolean deviceAllocated);

    public static IfaceValue.Vector.Shape getVectorShapeFromOperandN(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, int idx) {
        if (invokeOp.operands().get(idx) instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            if (varLoadOp.resultType() instanceof VarType varType) {
                return getVectorShape(lookup, varType.valueType());
            } else {
                return getVectorShape(lookup, varLoadOp.resultType());
            }
        }
        return null;
    }

    private boolean findIsSharedOrPrivateSpace(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findIsSharedOrPrivateSpace(varLoadOp.operands().getFirst());
        } else if (v.declaringElement() instanceof CoreOp.VarOp varOp) {
            return findIsSharedOrPrivateSpace(varOp.operands().getFirst());
        } else {
            return !(v instanceof Block.Parameter);
        }
    }

    public String mapLane(int lane) {
        return switch (lane) {
            case 0 -> "x";
            case 1 -> "y";
            case 2 -> "z";
            case 3 -> "w";
            default -> throw new InternalError("Invalid lane: " + lane);
        };
    }

    public abstract T hatSelectStoreOp(OpHelper.Invoke invoke, InvokeVar invokeVar);

    public T hatSelectLoadOp(OpHelper.Invoke invoke, InvokeVar invokeVar) {
        if (invoke.op().operands().getFirst().declaringElement() instanceof JavaOp.ArrayAccessOp.ArrayLoadOp vLoadOp) {
            recurse( vLoadOp);
        } else {
            id(invokeVar.name());
        }
        dot().id(mapLane(invokeVar.laneIdx()));
        return self();
    }

    public record InvokeVar(JavaOp.InvokeOp invokeOp, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        // recursive
        public static String vectorNameOrThrow(Value v) {
            return switch (OpHelper.asOpFromResultOrNull(v)) {
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp ->
                        vectorNameOrThrow(varLoadOp.operands().getFirst()); // recurse
                case CoreOp.VarOp varOp -> varOp.varName();
                case null -> null;
                default -> throw new IllegalStateException("failed to find vector name");
            };
        }
        public String name(){
            return vectorNameOrThrow(varLoadOp.operands().getFirst());
        }
        //recursive
        public CoreOp.VarOp findVarOpOrNull(Value v) {
            return switch (OpHelper.asOpFromResultOrNull(v)){
                case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> findVarOpOrNull(varLoadOp.operands().getFirst()); //recurse
                case CoreOp.VarOp varOp -> varOp;
                case null -> null;
                default ->  null;
            };
        }

        public CoreOp.VarOp varOpFromOperand(int idx){
            return findVarOpOrNull(invokeOp.operands().get(idx));
        }
        public CodeType returnType() {
            return invokeOp.resultType();
        }

        public int laneIdx() {
            return "xyzw".indexOf(invokeOp.invokeReference().name().charAt(0));
        }

        public String resolveName() {
            return varOpFromOperand(1) instanceof CoreOp.VarOp varOp?varOp.varName() : null;
        }
    }

    private static boolean is16BitFloat(OpHelper.Invoke invoke, Regex methodName) {
        return invoke.refIs(S16ImplOfF16.class) && invoke.nameMatchesRegex(methodName);
    }


    public abstract T hatF16ConvOp(JavaOp.InvokeOp invokeOp, Class<?> reducedFloatType);

    private boolean isVectorSelectOperation(Invoke invoke) {
        return invoke.nameMatchesRegex("[xyzw]") && invoke.refIs(IfaceValue.Vector.class) && invoke.opFromFirstOperandOrThrow() instanceof CoreOp.VarAccessOp.VarLoadOp;
    }

    private boolean isS16Conversion(Invoke invoke) {
        return !invoke.returnsVoid() && is16BitFloat(invoke, Regex.of("(of|floatToF16|float2bfloat16)")) && invoke.opFromOnlyUseOrNull() instanceof CoreOp.VarOp;
    }

    private boolean isS16ToFloatConversion(Invoke invoke) {
        return invoke instanceof Invoke.Static && invoke.nameMatchesRegex("(f16ToFloat|bfloat162float)") && invoke.returnsFloat();
    }

    private static boolean isF16Local(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isF16Local(varLoadOp); //recurse
            case CoreOp.VarOp varOp ->
                    !(varOp.operands().getFirst().declaringElement() instanceof JavaOp.InvokeOp invokeOp)
                            || !invokeOp.invokeReference().name().equals("array");
            default -> false;
        };
    }

    //recursive
    private static boolean isF16Local(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isF16Local(varLoadOp.operands().getFirst());
    }

    public abstract T hatF16ToFloatConvOp(Invoke invoke, Class<?> reducedFloatType, boolean wasFloat, boolean isF16Local);

    private boolean isInvokeFromNarrowTypeConversion(JavaOp.InvokeOp invoke) {
        SequencedSet<Op.Result> uses = invoke.result().uses();
        boolean[] result = new boolean[1];
        uses.forEach(usage -> {
            if (usage.declaringElement() instanceof JavaOp.InvokeOp invokeOp2) {
                var invoke2 = invoke(scopedCodeBuilderContext().lookup(), invokeOp2);
                if (invoke2.nameMatchesRegex("(f16ToFloat|bfloat162float)")) {
                    result[0] = true;
                }
            }
        });
        return result[0];
    }

    private boolean isInvokeFromSharedOrPrivate(Op.Result instance, Invoke invoke) {
        boolean isLocalOrPrivateDS = false;
        VarTable varTable = kernelCallGraph.getVarTable();
        if (instance.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                && scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst()) instanceof CoreOp.VarOp varOp
                && varTable.doesVarOpExist(scopedCodeBuilderContext.funcOp().funcName(), varOp)) {
            VarTable.HATOpAttribute attribute = varTable.getAttributeOrThrow(scopedCodeBuilderContext.funcOp().funcName(), varOp);
            isLocalOrPrivateDS = isAttributeSharedOrPrivate(attribute, invoke);
        }
        return isLocalOrPrivateDS;
    }

    private boolean isS16BinaryOp(Invoke invoke) {
        return is16BitFloat(invoke, Regex.of("(add|sub|mul|div)")) && !invoke.returnsVoid();
    }

    private void handleVectorOperations(Invoke invoke) {
        IfaceValue.Vector.Shape vectorShape = getVectorShape(invoke.lookup(), invoke.returnType());
        if (invoke.name().equalsIgnoreCase("float4view")
                || invoke.name().equalsIgnoreCase("float2view")) {
            // could be share or global
            boolean isSharedOrPrivate = isSharedOrPrivate(invoke.lookup(), invoke.op());
            Value source = invoke.op().operands().getFirst();
            Value index = invoke.op().operands().get(1);
            generateVectorLoad(source, index, vectorShape, isSharedOrPrivate);
        } else if (invoke.name().equalsIgnoreCase("of")) {
            generateVectorOf(invoke.op(), vectorShape);
        } else if (invoke.name().equalsIgnoreCase("makeMutable")) {
            var name = findVectorVarNameOrNull(invoke.op().operands().getFirst());
            id(name);
        }  else if (isVectorBinaryOperation(invoke)) {
            handleVectorBinaryOperation(invoke);
        } else {
            throw new IllegalStateException("[CodeGen] Vector Operation not found: " + invoke.name());
        }
    }

    private void handleVectorView(Invoke invoke) {
        IfaceValue.Vector.Shape vectorShape = getVectorShapeFromOperandN(invoke.lookup(), invoke.op(), 1);
        boolean isShared = findIsSharedOrPrivateSpace(invoke.op().operands().getFirst());
        String vectorName = findVectorVarNameOrNull(invoke.op().operands().get(1));
        hatVectorStoreOp(invoke.op(), vectorShape, vectorName, isShared);
    }

    private void handleVectorSelect(Invoke invoke) {
        InvokeVar invokeVar = new InvokeVar(invoke.op(), invoke.varLoadOpFromFirstOperandOrNull());
        if (invoke.returnsVoid()) {
            hatSelectStoreOp(invoke, invokeVar);
        } else {
            hatSelectLoadOp(invoke, invokeVar);
        }
    }

    private void handleS16Conversion(Invoke invoke) {
        // F16
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

    protected boolean isVectorBinaryOperation(Invoke invoke) {
        return (invoke.returns(IfaceValue.Vector.class) && invoke.nameMatchesRegex("(add|sub|mul|div)"));
    }

    public abstract T hatBinaryVectorOp(OpHelper.Invoke binOp);

    private void handleVectorBinaryOperation(Invoke invoke) {
        hatBinaryVectorOp(invoke);
    }

    @Override
    public final T invokeOp(JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(scopedCodeBuilderContext().lookup(), invokeOp);
        if (C99VecAndMatHandler.isVecInvoke(invoke)) { // hacked for vec op calls.
            C99VecAndMatHandler.handleInvoke(self(), invoke);
        } else if (isVectorOperation(invokeOp)) {
            handleVectorOperations(invoke);
        } else if (isVectorView(invokeOp)) {
            handleVectorView(invoke);
        } else if (isVectorSelectOperation(invoke)) {
            handleVectorSelect(invoke);
        } else if (isS16Conversion(invoke)) {
            handleS16Conversion(invoke);
        } else if (isS16ToFloatConversion(invoke)) {
            handleS16ToFloatConversion(invoke);
        } else if (isS16BinaryOp(invoke)) {
            handleS16BinaryOperation(invoke);
        } else if (invoke.refIs(IfaceValue.class)) {
            if (invoke instanceof Invoke.Virtual && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex(atomicIncRegex)) {
                if (invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instanceResult) {
                    atomicInc(instanceResult,
                            ((Regex.Match) atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                    );
                }
            } else if (isInvokeLoadingFromOnChipMemory(invokeOp)) {
                // equivalent to custom ops for private and shared memory
                generateOnChipMemoryLoad(invokeOp);
            } else if (invoke instanceof Invoke.Virtual && invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instance) {
                // Attention: Since F16.toFloat operations are supported, it should be possible to
                // implement a load from global memory from an F16Array and directly use it for a math operation.
                // In this case, we need to add an extra parenthesis.
                boolean narrowTypeCast = isInvokeFromNarrowTypeConversion(invokeOp);
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

                    either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

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
        } else if (!invoke.returnsVoid() && invoke.refIs(HATMath.class)) {
            // codegen for the math operation
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
            if (attribute != null) {
                // If attribute exits, we apply codegen based on attribute since there is a pre-search and
                // categorization about the corresponding OpenCL code to be generated.
                switch (attribute) {
                    case NARROW -> varOpForNarrowType(varOp);
                    case VECTOR -> varOpForVectors(varOp);
                    case INIT_SHARED -> varOpInit(varOp);
                    case SHARED -> varOpLocalMemory(varOp);
                    case PRIVATE -> varOpPrivateMemory(varOp);
                    default -> throw new IllegalStateException("Unexpected HATOpAttribute: " + attribute);
                }
            } else {
                genericVarOp(varOp);
            }
        }
        return self();
    }

    @Override
    public T arrayLoadOp( JavaOp.ArrayAccessOp.ArrayLoadOp arrayLoadOp) {
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

    public abstract T hatVectorStoreOp(JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp, IfaceValue.Vector.Shape vectorShape, boolean isLocal, String name);

    @Override
    public T arrayStoreOp(JavaOp.ArrayAccessOp.ArrayStoreOp arrayStoreOp) {
        if (HATArrayViewPhase.isVectorOp(scopedCodeBuilderContext.lookup(), arrayStoreOp)) {
            Op varOp = opFromFirstOperandOrNull(((Op.Result) arrayStoreOp.operands().getLast()).op());
            String name = HATArrayViewPhase.hatPtrName(varOp);
            var vectorShape = getVectorShape(scopedCodeBuilderContext.lookup(), arrayStoreOp.operands().getLast().type());
            boolean deviceAllocated = HATArrayViewPhase.isLocalSharedOrPrivate(arrayStoreOp);
            hatVectorStoreOp(arrayStoreOp, vectorShape, deviceAllocated , name);
        } else {
            recurse(((Op.Result) arrayStoreOp.operands().get(0)).op());
            sbrace(_ -> recurse(((Op.Result) arrayStoreOp.operands().get(1)).op()));
            sp().equals().sp();
            recurse(((Op.Result) arrayStoreOp.operands().get(2)).op());
        }
        return self();
    }

    protected abstract T varOpForNarrowType(CoreOp.VarOp varOp);
    protected abstract T varOpForVectors(CoreOp.VarOp varOp);
    protected abstract T varOpInit(CoreOp.VarOp varOp);
    protected abstract T varOpLocalMemory(CoreOp.VarOp varOp);
    protected abstract T varOpPrivateMemory(CoreOp.VarOp varOp);

    protected void genFieldAccess(Value operand, boolean isReference) {
        if (isReference) {
            rarrow().id(VALUE);
        } else if (!OpHelper.isPrimitiveResult(operand)) {
            dot().id(VALUE);
        }
    }

    private void generateMathIntrinsicOperation(Invoke invoke) {
        // if the resulting type is a narrowed-type (e.g., bfloat16, or half float)
         if (invoke.returnsClassType() && S16ImplOfF16.codeTypeToFloatClassOrNull(invoke,(ClassType)invoke.returnType()) instanceof Class<? extends S16ImplOfF16> float16Class){
             paren(_ ->
                     f16OrBF16(float16Class))
                     .brace(_-> {
                         id(mapMathIntrinsic(invoke.name()));
                         // For each operand, obtain if it is a reference from global memory or device memory.
                         List<Boolean> referenceList = IntStream.range(0, invoke.op().operands().size())
                                 .mapToObj(i -> isArrayReference( invoke.op().operands().get(i)))
                                 .collect(Collectors.toList());

                         paren(_ -> {
                             int[] counter = {0};
                             commaSpaceSeparated(invoke.op().operands(), op -> {
                                 recurse(OpHelper.asResultOrThrow(op).op());
                                 genFieldAccess(op, referenceList.get(counter[0]++));
                             });
                         });
                     });
         }else {
             id(mapMathIntrinsic(invoke.name()));
             paren(_ ->
                 commaSpaceSeparated(invoke.op().operands(), op ->
                     recurse(OpHelper.asResultOrThrow(op).op())
                 )
             );
         }
    }

    protected abstract String mapMathIntrinsic(String name);
}
