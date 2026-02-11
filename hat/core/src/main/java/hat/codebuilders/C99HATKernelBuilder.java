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
import hat.buffer.F16Array;
import hat.dialect.HATBarrierOp;
import hat.dialect.HATF16Op;
import hat.dialect.HATMathLibOp;
import hat.dialect.HATMemoryDefOp;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATPtrOp;
import hat.dialect.HATThreadOp;
import hat.dialect.HATVectorOp;
import hat.dialect.ReducedFloatType;
import hat.phases.HATFP16Phase;
import hat.phases.HATPhaseUtils;
import hat.types.BF16;
import hat.types.F16;
import hat.types._F16;
import optkl.IfaceValue;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.OpHelper;
import optkl.codebuilders.ScopedCodeBuilderContext;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Schema;
import jdk.incubator.code.Op;
import optkl.FuncOpParams;
import optkl.util.Regex;
import optkl.util.Mutable;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.codebuilders.CodeBuilder;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static hat.buffer.F16Array.F16Impl;

import static java.lang.invoke.MethodHandles.lookup;
import static optkl.OpHelper.Invoke;
import static optkl.OpHelper.FieldAccess.fieldAccess;
import static optkl.OpHelper.Invoke.invoke;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilder<T> implements HATOpDispatcher<T> {

    protected C99HATKernelBuilder(ScopedCodeBuilderContext scopedCodeBuilderContext) {
        super(scopedCodeBuilderContext);
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
        return identifier("HAT_GIX");
    }

    public final T HAT_GIY() {
        return identifier("HAT_GIY");
    }

    public final T HAT_GIZ() {
        return identifier("HAT_GIZ");
    }

    public final T HAT_GSX() {
        return identifier("HAT_GSX");
    }

    public final T HAT_GSY() {
        return identifier("HAT_GSY");
    }

    public final T HAT_GSZ() {
        return identifier("HAT_GSZ");
    }

    public final T HAT_LIX() {
        return identifier("HAT_LIX");
    }

    public final T HAT_LIY() {
        return identifier("HAT_LIY");
    }

    public final T HAT_LIZ() {
        return identifier("HAT_LIZ");
    }

    public final T HAT_LSX() {
        return identifier("HAT_LSX");
    }

    public final T HAT_LSY() {
        return identifier("HAT_LSY");
    }

    public final T HAT_LSZ() {
        return identifier("HAT_LSZ");
    }

    public final T HAT_BIX() {
        return identifier("HAT_BIX");
    }

    public final T HAT_BIY() {
        return identifier("HAT_BIY");
    }

    public final T HAT_BIZ() {
        return identifier("HAT_BIZ");
    }

    public final T HAT_BSX() {
        return identifier("HAT_BSX");
    }

    public final T HAT_BSY() {
        return identifier("HAT_BSY");
    }

    public final T HAT_BSZ() {
        return identifier("HAT_BSZ");
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
        return HAT_KERNEL().space().voidType().space().funcName(funcOp);
    }

    public final  T functionDeclaration( JavaType javaType, CoreOp.FuncOp funcOp) {
        return HAT_FUNC().space().type(javaType).space().funcName(funcOp);
    }

    public final boolean isHalfType(Schema.IfaceType ifaceType) {
        return ifaceType.iface.isAssignableFrom(F16.class)
                || ifaceType.iface.isAssignableFrom(F16Array.F16Impl.class);
    }

    public final boolean isbfloat16(Schema.IfaceType ifaceType) {
         return ifaceType.iface.isAssignableFrom(BF16.class)
               || ifaceType.iface.isAssignableFrom(BF16Array.BF16Impl.class);
    }

    public final T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword()
                .space()
                .structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .space()
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
                                typeName("half");
                            } else if (isbfloat16(ifaceType)) {
                                typeName("BFLOAT16");
                            } else {
                                typeName(primitiveField.type.getSimpleName());
                            }
                            space().typeName(primitiveField.name);
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
                            space().typeName(ifaceField.name);
                            if (ifaceField instanceof Schema.FieldNode.IfaceArray array) {
                                if (array instanceof Schema.FieldNode.IfaceFieldControlledArray fieldControlledArray) {
                                    if (isLast && ifaceType.parent == null) {
                                        sbrace(_ -> literal(1));
                                    } else {
                                      //  if (boundSchema != null) {
                                          //  var done = StreamMutable.of(false);
                                            boundSchema.boundArrayFields().stream()
                                                    .filter(a->a.field.equals(ifaceField))
                                                    .findFirst()
                                                    .ifPresentOrElse(
                                                            a-> sbrace(_ -> literal(a.len)),
                                                            ()->{
                                                                throw new IllegalStateException("we need to extract the array size hat kind of array ");
                                                            });
                                        //} else {
                                          //  throw new IllegalStateException("bound schema is null  !");
                                      //  }
                                    }
                                } else if (array instanceof Schema.FieldNode.IfaceFixedArray fixed) {
                                    sbrace(_ -> literal(Math.max(1, fixed.len)));
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                            }
                        } else if (field instanceof Schema.SchemaNode.Padding padding) {
                            u08Type().space().identifierWithRandomSuffix("pad$",5).sbrace(_->intValue((int)(padding.len)));//; emitText(toC99(padding));
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
        StringBuilder sb = new StringBuilder();
        final String LEGAL_CHARS = "_$ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        ThreadLocalRandom.current() //
                .ints(len, 0, LEGAL_CHARS.length()) //
                .mapToObj(LEGAL_CHARS::charAt) //
                .forEach(sb::append);
        identifier(prefix+sb.toString());
        return self();
    }

    public record LocalArrayDeclaration(ClassType classType, HATMemoryVarOp varOp) {}


    public final T privateDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return suffix_t(localArrayDeclaration.classType()).space().varName(localArrayDeclaration.varOp());
    }

    public final T localDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return HAT_LOCAL_MEM()
                .space() // we should be able to compose-call to privateDeclaration?
                .suffix_t(localArrayDeclaration.classType())
                .space()
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
                 typedefKeyword().space().s08Type("byte").semicolonNl()
                .typedefKeyword().space().s08Type("boolean").semicolonNl()
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
        if (javaType instanceof ClassType classType
                && OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, IfaceValue.class)
                && !OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, _F16.class)
        ) {
            HAT_GLOBAL_MEM().space().suffix_t(classType).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType, KernelContext.class)) {
            HAT_GLOBAL_MEM().space().suffix_t(KernelContext.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType,F16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().space().suffix_t(F16Impl.class).asterisk();
        } else if (OpHelper.isAssignable(scopedCodeBuilderContext().lookup(), javaType,BF16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().space().suffix_t(BF16Array.BF16Impl.class).asterisk();
        } else {
            typeName(javaType.toString());
        }
        return self();
    }


    public final  T kernelMethod(ScopedCodeBuilderContext buildContext,CoreOp.FuncOp funcOp) {
          buildContext.funcScope(funcOp, () -> {
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

    public final  T kernelEntrypoint(ScopedCodeBuilderContext buildContext) {
        nl();
        buildContext.funcScope(buildContext.funcOp(), () ->
                kernelDeclaration(buildContext.funcOp())
                .parenNlIndented(_ -> commaNlSeparated(
                    buildContext.paramTable.list(),
                        this::declareParam)
                )
                .braceNlIndented(_ -> nlSeparated(
                    OpHelper.Statement.statements(buildContext.funcOp().bodies().getFirst().entryBlock()),
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

    protected final T genReducedType(ReducedFloatType reducedFloatType) {
        return (switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> f16Type();
            case ReducedFloatType.BFloat16 _ -> bf16Type();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        });
    }

    @Override
    public final T hatF16VarOp( HATF16Op.HATF16VarOp hatF16VarOp) {
        ReducedFloatType reducedFloatType = hatF16VarOp.reducedFloatType();
        return (switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> f16Type();
            case ReducedFloatType.BFloat16 _ ->  bf16Type();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }).space().assign(
                _-> identifier(hatF16VarOp.varName()),
                _->recurse( OpHelper.asResultOrThrow(hatF16VarOp.operands().getFirst()).op()));
    }

    private boolean isMixedFirstOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16Op.HATF16BinaryOp.FIRST_OP;
    }

    private boolean isMixedSecondOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16Op.HATF16BinaryOp.LAST_OP;
    }

    public final T builtin_float2bfloat16() {
        return identifier("floatTobfloat16");
    }

    public final T builtin_bfloat16ToFloat() {
        return identifier("bfloat16Tofloat");
    }

    private final T binaryOperationsForBfloat16( HATF16Op.HATF16BinaryOp hatf16BinaryOp) {
        byte f32Mixed = hatf16BinaryOp.getByteFloatRepresentation();
        paren(_-> bf16Type());
        brace(_-> {
            paren(_-> {
                builtin_float2bfloat16();
                oparen();
                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();// open
                }
                recurse( OpHelper.asResultOrThrow(hatf16BinaryOp.operands().getFirst()).op());

                List<Boolean> references = hatf16BinaryOp.references();
                if (references.getFirst()) {
                    rarrow().identifier("value");
                } else if (!OpHelper.isPrimitiveResult(hatf16BinaryOp.operands().getFirst())) {
                    dot().identifier("value");
                }

                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    cparen(); //closed
                }
                space().identifier(hatf16BinaryOp.binaryOperationType().symbol()).space();

                if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();
                }

                recurse(OpHelper.asResultOrThrow(hatf16BinaryOp.operands().get(1)).op());
                if (references.get(1)) {
                    rarrow().identifier("value");
                } else if (!OpHelper.isPrimitiveResult(hatf16BinaryOp.operands().get(1))) {
                    dot().identifier("value");
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
        ReducedFloatType reducedFloatType = hatF16BinaryOp.reducedFloatType();
        if (reducedFloatType instanceof ReducedFloatType.BFloat16) {
            return binaryOperationsForBfloat16( hatF16BinaryOp);
        }
        paren(_-> f16Type());
        return brace(_->
            paren(_-> {
                recurse( OpHelper.asResultOrThrow(hatF16BinaryOp.operands().getFirst()).op());
                if (hatF16BinaryOp.references().getFirst()) {
                    rarrow().identifier("value");
                } else if (!OpHelper.isPrimitiveResult(hatF16BinaryOp.operands().getFirst())) {
                    dot().identifier("value");
                } else {
                    blockComment("hatF16BinaryOp not a result !!");
                }
                space().identifier(hatF16BinaryOp.binaryOperationType().symbol()).space();
                recurse( OpHelper.asResultOrThrow(hatF16BinaryOp.operands().get(1)).op());
                if (hatF16BinaryOp.references().get(1)) {
                    rarrow().identifier("value");
                } else if (!OpHelper.isPrimitiveResult(hatF16BinaryOp.operands().get(1))) {
                    dot().identifier("value");
                }else {
                    blockComment("hatF16BinaryOp not a value !!");
                }
            })
        );
    }

    @Override
    public final T hatF16VarLoadOp( HATF16Op.HATF16VarLoadOp hatF16VarLoadOp) {
        return identifier(hatF16VarLoadOp.varName()).dot().identifier("value");
    }

    @Override
    public final T hatVectorMakeOf( HATVectorOp.HATVectorMakeOfOp hatVectorMakeOfOp) {
        return identifier(hatVectorMakeOfOp.varName());
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
        return suffix_t(hatPrivateInitVarOp.classType()).space()
                .assign(
                        _-> identifier(hatPrivateInitVarOp.varName()),
                        _->recurse(OpHelper.asResultOrThrow(hatPrivateInitVarOp.operands().getFirst()).op()));
    }

    @Override
    public final T hatMemoryLoadOp( HATMemoryDefOp.HATMemoryLoadOp hatMemoryLoadOp) {
        return recurse( OpHelper.asResultOrThrow(hatMemoryLoadOp.operands().getFirst()).op())
                .dot().identifier(hatMemoryLoadOp.memberName())
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
        identifier(hatPtrOp.name());
        boolean isLocalOrPrivateDS = false;
        if (((Op.Result) hatPtrOp.operands().getFirst()).op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            Op resolve = scopedCodeBuilderContext().resolve(varLoadOp.operands().getFirst());
            if (resolve instanceof HATMemoryVarOp) {
                isLocalOrPrivateDS = true;
            }
        }
        either(isLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow);

        if (hatPtrOp instanceof HATPtrOp.HATPtrLengthOp) {
            identifier("length");
        } else {
            boolean finalIsLocalOrPrivateDS = isLocalOrPrivateDS;// ?
            identifier("array").sbrace(_ -> {
                paren(_ -> identifier("long"));
                paren(_ -> {
                    if (hatPtrOp.strides().size() > 1) {
                        paren(_ -> recurse(((Op.Result) hatPtrOp.operands().get(2)).op()));
                        asterisk().identifier(hatPtrOp.name());
                        either(finalIsLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow).identifier(hatPtrOp.strides() != null ? hatPtrOp.strides().getFirst() : "width");
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
                       _ -> bfloat16Type(b16).semicolonNl()
                               .identifier(b16).dot().identifier(s).sbrace( _ -> intConstZero()).equals().intConstZero().semicolonNl()
                               .identifier(b16).dot().identifier(s).sbrace( _ -> intConstOne()).equals().constant(parameterName).semicolonNl()
                               .returnKeyword(_-> identifier(b16).dot().identifier(f)));
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
                        _ ->  brace( _ -> identifier(parameterName)).semicolonNl())
                        .assign( _ -> u32Type("bits"), _ -> identifier(idBFloat16).dot().identifier("i")).semicolonNl()
                        .assign( _ -> u16Type("sign_bit"), _ -> cast( _ -> s16Type()).paren( _ -> paren( _ -> identifier("bits").ampersand().constant("0x80000000")).rightShift(16))).semicolonNl()
                        .assign( _ -> s32Type("lsb"), _ -> identifier("bits").ampersand().constant("0x10000")).semicolonNl()
                        .assign( _ -> s32Type("round"), _ -> identifier("bits").ampersand().constant("0x08000")).semicolonNl()
                        .assign( _ -> s32Type("sticky"), _ -> identifier("bits").ampersand().constant("0x07FFF")).semicolonNl()
                        .ifTrueCondition(_ -> identifier("round").space().ne().space().intConstZero().condAnd().paren( _ -> paren( _ -> identifier("lsb").bitwiseOR().identifier("sticky")).ne().intConstZero()),
                                _ -> identifier("bits").space().plusEquals().space().constant("0x10000"))
                        .returnKeyword( _ -> cast( _ ->u16Type()).paren( _ -> paren( _ -> paren( _-> identifier("bits").rightShift(16)).bitwiseOR().identifier("sign_bit")).ampersand().constant("0xffff"))));
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
        if ( invoke.refIs(IfaceValue.class )) {
            if (invoke instanceof Invoke.Virtual && invoke.operandCount() == 1 && invoke.returnsInt() && invoke.nameMatchesRegex(atomicIncRegex)) {
                if (invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instanceResult) {
                    atomicInc( instanceResult,
                            ((Regex.Match)atomicIncRegex.is(invoke.name())).stringOf(1) // atomicXXInc -> atomicXX
                    );
                }
            } else if (invoke instanceof Invoke.Virtual && invoke.resultFromOperandNOrThrow(0) instanceof Op.Result instance) {
                parenWhen(
                        invoke.operandCount() > 1
                                && invoke(scopedCodeBuilderContext().lookup(),instance.op()) instanceof Invoke invoke0
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
                        _->{
                            when(invoke.returnsClassType(), _ -> ampersand());
                            recurse( instance.op());
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
                                equals().recurse( op);
                            }
                        }
                        case 3-> {
                            if ( invoke.opFromOperandNOrThrow(1) instanceof Op op1
                                    && invoke.opFromOperandNOrThrow(2) instanceof Op op2) {
                                sbrace(_ -> recurse( op1)).equals().recurse( op2);
                            }
                        }
                        default -> throw new IllegalStateException("How ");
                    }
                } else {
                    if (invoke.opFromOperandNOrNull(1) instanceof Op op) {
                        sbrace(_ -> recurse( op));
                    }else{
                        // this is just call.
                    }
                }
            }
        } else {// General case
            funcName(invoke.op()).paren(_ ->
                    commaSpaceSeparated(invoke.op().operands(),
                            op -> {if (op instanceof Op.Result result) {recurse( result.op());}
                            })
            );
        }
        return self();
    }

    public static final String VALUE = "value";

    protected void genFieldAccess(Value operand, boolean isReference) {
        if (isReference) {
            rarrow().identifier(VALUE);
        } else if (!OpHelper.isPrimitiveResult(operand)) {
            dot().identifier(VALUE);
        }
    }

    @Override
    public T hatMathLibOp(HATMathLibOp hatMathLibOp) {

        // Obtain if the resulting type is a narrowed-type (e.g., bfloat16, or half float)
        ReducedFloatType reducedFloatType = HATFP16Phase.categorizeReducedFloat(hatMathLibOp.resultType().toString());
        if (reducedFloatType != null) {
            // If special type, then we need to build the type
            // For now this applies to F16 and bFloat16
            paren(_ -> genReducedType(reducedFloatType)).obrace();
        }
        identifier(mapMathIntrinsic(reducedFloatType, hatMathLibOp.name()));

        // For each operand, obtain if it is a reference from global memory or device memory.
        // Important: when the code-tree has its own lowering, the way to place this information
        // would be in a C99 lowered tree. Thus, we will avoid code-analysis during code-gen.
        List<Boolean> referenceList = IntStream.range(0, hatMathLibOp.operands().size())
                .mapToObj(i -> HATPhaseUtils.isArrayReference(lookup(), hatMathLibOp.operands().get(i)))
                .collect(Collectors.toList());

        paren( _ -> {
            int numArgs = hatMathLibOp.numArguments();
            IntStream.range(0, numArgs).forEach(i -> {
                recurse(OpHelper.asResultOrThrow(hatMathLibOp.operands().get(i)).op());
                if (reducedFloatType != null) {
                    genFieldAccess(hatMathLibOp.operands().get(i), referenceList.get(i));
                }
                // Don't generate the comma after the last argument
                if (i != numArgs - 1) {
                    comma();
                }
            });
        });
        if (reducedFloatType != null) {
            cbrace();
        }
        return self();
    }

    protected abstract String mapMathIntrinsic(ReducedFloatType reducedFloatType, String name);
}
