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
import hat.dialect.HATBarrierOp;
import hat.dialect.HATF16Op;
import hat.dialect.HATMemoryDefOp;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATPtrOp;
import hat.dialect.HATTensorOp;
import hat.dialect.HATThreadOp;
import hat.dialect.HATVectorOp;
import hat.types.BF16;
import hat.types.F16;
import hat.types.Tensor;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import hat.types.S16ImplOfF16;
import optkl.IfaceValue;
import jdk.incubator.code.Value;
import optkl.OpHelper;
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
import java.util.SequencedSet;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import static hat.buffer.F16Array.F16Impl;
import static java.lang.invoke.MethodHandles.lookup;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke.invoke;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilder<T> implements HATOpDispatcher<T> {
    protected  final KernelCallGraph kernelCallGraph;

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
                            && invoke(lookup(),varOpResult.op()) instanceof OpHelper.Invoke invoke && invoke.named("array");
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

    protected abstract T hatWarpSize();

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
            case HATThreadOp.HAT_WARP_SIZE  _ -> HAT_WARP_SIZE();
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

    public record LocalArrayDeclaration(ClassType classType, HATMemoryVarOp varOp) {}


    public final T privateDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return suffix_t(localArrayDeclaration.classType()).sp().varName(localArrayDeclaration.varOp());
    }

    public final T localDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return HAT_LOCAL_MEM()
                .sp() // we should be able to compose-call to privateDeclaration?
                .suffix_t(localArrayDeclaration.classType())
                .sp()
                .varName(localArrayDeclaration.varOp());
    }

    @Override
    public final T hatBarrierOp(HATBarrierOp barrierOp) {
        return HAT_BARRIER();
    }

    @Override
    public final T hatLocalVarOp( HATMemoryVarOp.HATLocalVarOp hatLocalVarOp) {
        return   localDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
    }

    @Override
    public final T hatPrivateVarOp( HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp) {
        return privateDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
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
        if (C99VecAndMatHandler.isVecOrMatType(scopedCodeBuilderContext().lookup(),javaType)){
            C99VecAndMatHandler.handleType(self(),javaType);
        }else if (javaType instanceof ClassType classType
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


    @Override
    public final T hatVectorVarLoadOp( HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp) {
        return varName(hatVectorVarLoadOp);
    }

    public final T f16Type() {
        return suffix_t(F16.class);
    }

    public final T bf16Type() {
        return suffix_t(BF16.class);
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

    @Override
    public final T hatF16VarOp( HATF16Op.HATF16VarOp hatF16VarOp) {
        var float16Class = hatF16VarOp.float16Class();
        return f16OrBF16(float16Class).sp().assign(
                _-> id(hatF16VarOp.varName()),
                _->recurse( OpHelper.asResultOrThrow(hatF16VarOp.operands().getFirst()).op()));
    }

    protected boolean isMixedFirstOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16Op.HATF16BinaryOp.FIRST_OP;
    }

    private boolean isMixedSecondOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16Op.HATF16BinaryOp.LAST_OP;
    }

    public final T builtin_float2bfloat16() {
        return id("floatTobfloat16");
    }

    public final T builtin_bfloat16ToFloat() {
        return id("bfloat16Tofloat");
    }

    public static final String VALUE = "value";

    private final T binaryOperationsForBfloat16( HATF16Op.HATF16BinaryOp hatf16BinaryOp) {

        boolean isFirstOperandReference = isArrayReference(hatf16BinaryOp.operands().get(0));
        boolean isSecondOperandReference = isArrayReference(hatf16BinaryOp.operands().get(1));
        final byte f32Mixed;
        if (!isFirstOperandReference && isOperandF32(hatf16BinaryOp.operands().get(0))) {
            f32Mixed = HATF16Op.HATF16BinaryOp.FIRST_OP;
        } else if (!isSecondOperandReference && isOperandF32(hatf16BinaryOp.operands().get(1))) {
            f32Mixed = HATF16Op.HATF16BinaryOp.LAST_OP;
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
                recurse( OpHelper.asResultOrThrow(hatf16BinaryOp.operands().getFirst()).op());

                if (isFirstOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(hatf16BinaryOp.operands().getFirst())) {
                    dot().id(VALUE);
                }

                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    cparen(); //closed
                }
                sp().id(hatf16BinaryOp.binaryOperationType().symbol()).sp();

                if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();
                }

                recurse(OpHelper.asResultOrThrow(hatf16BinaryOp.operands().get(1)).op());
                if (isSecondOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(hatf16BinaryOp.operands().get(1))) {
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

    @Override
    public T hatF16BinaryOp( HATF16Op.HATF16BinaryOp hatF16BinaryOp) {
        var float16Class = hatF16BinaryOp.float16Class();
        if (BF16.class.isAssignableFrom(float16Class)) {
            return binaryOperationsForBfloat16( hatF16BinaryOp);
        }
        paren(_-> f16Type());
        return brace(_->
            paren(_-> {
                recurse( OpHelper.asResultOrThrow(hatF16BinaryOp.operands().getFirst()).op());
                boolean isFirstOperandReference = isArrayReference( hatF16BinaryOp.operands().get(0));
                boolean isSecondOperandReference = isArrayReference(hatF16BinaryOp.operands().get(1));
                if (isFirstOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(hatF16BinaryOp.operands().getFirst())) {
                    dot().id(VALUE);
                } else {
                    blockComment("hatF16BinaryOp not a result !!");
                }
                sp().id(hatF16BinaryOp.binaryOperationType().symbol()).sp();
                recurse( OpHelper.asResultOrThrow(hatF16BinaryOp.operands().get(1)).op());
                if (isSecondOperandReference) {
                    rarrow().id(VALUE);
                } else if (!OpHelper.isPrimitiveResult(hatF16BinaryOp.operands().get(1))) {
                    dot().id(VALUE);
                }else {
                    blockComment("hatF16BinaryOp not a value !!");
                }
            })
        );
    }

    @Override
    public final T hatF16VarLoadOp( HATF16Op.HATF16VarLoadOp hatF16VarLoadOp) {
        return id(hatF16VarLoadOp.varName()).dot().id(VALUE);
    }

    @Override
    public final T hatVectorMakeOf( HATVectorOp.HATVectorMakeOfOp hatVectorMakeOfOp) {
        return id(hatVectorMakeOfOp.varName());
    }

    public abstract T genVectorIdentifier( HATVectorOp.HATVectorOfOp hatVectorOfOp);

    @Override
    public final T hatVectorOfOps( HATVectorOp.HATVectorOfOp hatVectorOp) {
        return genVectorIdentifier( hatVectorOp)
                .paren(_->commaSpaceSeparated(
                        hatVectorOp.operands(),
                        operand -> recurse( OpHelper.asResultOrThrow(operand).op()))
                );
    }

    @Override
    public final T hatPrivateVarInitOp( HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp) {
        return suffix_t(hatPrivateInitVarOp.classType()).sp()
                .assign(
                        _-> id(hatPrivateInitVarOp.varName()),
                        _->recurse(OpHelper.asResultOrThrow(hatPrivateInitVarOp.operands().getFirst()).op()));
    }

    @Override
    public final T hatMemoryLoadOp( HATMemoryDefOp.HATMemoryLoadOp hatMemoryLoadOp) {
        return recurse( OpHelper.asResultOrThrow(hatMemoryLoadOp.operands().getFirst()).op())
                .dot().id(hatMemoryLoadOp.memberName())
                .when(hatMemoryLoadOp.operands().size() > 1,_->// If the hatMemoryLoadOp has more than 1 operand, the second is the index
                   sbrace(_-> recurse( OpHelper.asResultOrThrow(hatMemoryLoadOp.operands().get(1)).op()))
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

    private T ptrAccess(HATPtrOp hatPtrOp) {
        id(hatPtrOp.name());
        boolean isLocalOrPrivateDS = false;
        if (((Op.Result) hatPtrOp.operands().getFirst()).op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
            if (resolve instanceof HATMemoryVarOp) {
                isLocalOrPrivateDS = true;
            }
        }
        either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

        if (hatPtrOp instanceof HATPtrOp.HATPtrLengthOp) {
            id("length");
        } else {
            boolean finalIsLocalOrPrivateDS = isLocalOrPrivateDS;// ?
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
    public final T varLoadOp( CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
        switch (resolve) {
            case CoreOp.VarOp $ -> varName($);
            case HATMemoryVarOp $ -> varName($);
            case HATVectorOp.HATVectorVarOp $ -> varName($);
            case HATVectorOp.HATVectorLoadOp $ -> varName($);
            case HATVectorOp.HATVectorBinaryOp $ -> varName($);
            case HATF16Op.HATF16VarOp $ -> varName($);
            case HATTensorOp.TensorVarOp $ -> varName($);
            case null, default -> {
            }
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
            case HATF16Op.HATF16VarOp hatf16VarOp -> varName(hatf16VarOp);
            case HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp -> varName(hatPrivateInitVarOp);
            case HATMemoryVarOp.HATPrivateVarOp hatPrivateVarOp -> varName(hatPrivateVarOp);
            case HATMemoryVarOp.HATLocalVarOp hatLocalVarOp -> varName(hatLocalVarOp);
            case HATVectorOp.HATVectorVarOp hatVectorVarOp -> varName(hatVectorVarOp);
            case HATTensorOp.TensorVarOp hattensorVarOp -> varName(hattensorVarOp);
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

    @Override
    public final T invokeOp( JavaOp.InvokeOp invokeOp) {
        var invoke = invoke(scopedCodeBuilderContext().lookup(),invokeOp);
        if (C99VecAndMatHandler.isVecInvoke( invoke)){ // hacked for vec op calls.
            C99VecAndMatHandler.handleInvoke(self(),invoke);
        }else if (invoke.refIs(IfaceValue.class)) {
            if (invoke instanceof Invoke.Virtual && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex(atomicIncRegex)) {
                if (invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instanceResult) {
                    atomicInc(instanceResult,
                            ((Regex.Match) atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                    );
                }
            } else if (invoke instanceof Invoke.Virtual && invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instance) {
                // Attention: Since F16.toFloat operations are supported, it should be possible to
                // implement a load from global memory from an F16Array and directly use it for a math operation.
                // In this case, we need to add an extra parenthesis.
                SequencedSet<Op.Result> uses = invokeOp.result().uses();
                boolean narrowTypeCast = uses.stream().anyMatch(node -> node.op() instanceof HATF16Op.HATF16ToFloatConvOp);
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
                    boolean isLocalOrPrivateDS = (instance.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp
                            && scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst()) instanceof HATMemoryVarOp);
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
                funcName(invoke.op()).paren(_ ->
                        commaSpaceSeparated(invoke.op().operands(),
                                op -> {
                                    if (op instanceof Op.Result result) {
                                        recurse(result.op());
                                    }
                                })
                );
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

    protected T recurseValueOrThrough(Value value) {
        if (value instanceof Op.Result r) {
            return recurse(r.op());
        } else {
            throw launchBackendException("OpResult expected, but found: " + value.getClass());
        }
    }

    protected abstract RuntimeException launchBackendException(String message);

}
