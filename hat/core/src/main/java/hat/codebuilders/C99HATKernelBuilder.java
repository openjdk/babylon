/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
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

import hat.buffer.*;
import hat.KernelContext;
import hat.dialect.*;
import hat.types.BF16;
import hat.types.F16;
import optkl.OpTkl;
import optkl.codebuilders.ScopedCodeBuilderContext;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;
import jdk.incubator.code.Op;
import optkl.FuncOpParams;
import optkl.util.StreamMutable;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;
import optkl.codebuilders.CodeBuilder;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import static hat.buffer.F16Array.F16Impl;
import static optkl.FieldAccess.fieldAccessOpHelper;
import static optkl.OpTkl.asResultOrThrow;
import static optkl.OpTkl.isAssignable;
import static optkl.OpTkl.isPrimitiveResult;
import static optkl.OpTkl.statements;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilderContext<T> implements BabylonKernelOpBuilder<T>  {

    public T HAT_KERNEL() {
        return keyword("HAT_KERNEL");
    }

    public T HAT_FUNC() {
        return keyword("HAT_FUNC");
    }

    public T HAT_GLOBAL_MEM() {
        return keyword("HAT_GLOBAL_MEM");
    }

    public T HAT_LOCAL_MEM() {
        return keyword("HAT_LOCAL_MEM");
    }

    public T HAT_BARRIER() {
        return keyword("HAT_BARRIER");
    }

    public T HAT_GIX(){
        return identifier("HAT_GIX");
    }

    public T HAT_GIY(){
        return identifier("HAT_GIY");
    }

    public T HAT_GIZ(){
        return identifier("HAT_GIZ");
    }

    public T HAT_GSX(){
        return identifier("HAT_GSX");
    }

    public T HAT_GSY(){
        return identifier("HAT_GSY");
    }

    public T HAT_GSZ(){
        return identifier("HAT_GSZ");
    }

    public T HAT_LIX(){
        return identifier("HAT_LIX");
    }

    public T HAT_LIY(){
        return identifier("HAT_LIY");
    }

    public T HAT_LIZ(){
        return identifier("HAT_LIZ");
    }

    public T HAT_LSX(){
        return identifier("HAT_LSX");
    }

    public T HAT_LSY(){
        return identifier("HAT_LSY");
    }

    public T HAT_LSZ(){
        return identifier("HAT_LSZ");
    }


    public T HAT_BIX(){
        return identifier("HAT_BIX");
    }

    public T HAT_BIY(){
        return identifier("HAT_BIY");
    }

    public T HAT_BIZ(){
        return identifier("HAT_BIZ");
    }


    @Override
    public T hatGlobalThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATGlobalThreadIdOp globalThreadIdOp) {
        switch (globalThreadIdOp.getDimension()) {
            case 0 -> HAT_GIX();
            case 1 -> HAT_GIY();
            case 2 -> HAT_GIZ();
            default -> throw new RuntimeException("globalId id = " + globalThreadIdOp.getDimension());
        }
        return self();
    }

    @Override
    public T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATGlobalSizeOp globalSizeOp) {
        return (switch (globalSizeOp.getDimension()) {
            case 0 -> HAT_GSX();
            case 1 -> HAT_GSY();
            case 2 -> HAT_GSZ();
            default -> throw new RuntimeException("globalSize id = " + globalSizeOp.getDimension());
        });

    }

    @Override
    public T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATLocalThreadIdOp localThreadIdOp) {
        return (switch (localThreadIdOp.getDimension()) {
            case 0 -> HAT_LIX();
            case 1 -> HAT_LIY();
            case 2 -> HAT_LIZ();
            default -> throw new RuntimeException("localId id = " + localThreadIdOp.getDimension());
        });

    }

    @Override
    public T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATLocalSizeOp hatLocalSizeOp) {
        return (switch (hatLocalSizeOp.getDimension()) {
            case 0 -> HAT_LSX();
            case 1 -> HAT_LSY();
            case 2 -> HAT_LSZ();
            default -> throw new RuntimeException("localSize id = " + hatLocalSizeOp.getDimension());
        });
    }

    @Override
    public T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HATThreadOp.HATBlockThreadIdOp hatBlockThreadIdOp) {
        return (switch (hatBlockThreadIdOp.getDimension()) {
            case 0 -> HAT_BIX();
            case 1 -> HAT_BIY();
            case 2 -> HAT_BIZ();
            default -> throw new RuntimeException("blockId id = " + hatBlockThreadIdOp.getDimension());
        });
    }

    public T kernelDeclaration(CoreOp.FuncOp funcOp) {
        return HAT_KERNEL().space().voidType().space().funcName(funcOp);
    }

    public T functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp) {
        return HAT_FUNC().space().type(codeBuilderContext,javaType).space().funcName(funcOp);
    }

    public final boolean isHalfType(Schema.IfaceType ifaceType) {
        return (ifaceType.iface.getName().equals(F16.class.getName())
                || ifaceType.iface.getName().equals(F16Array.F16Impl.class.getName()));
    }

    public final boolean isbfloat16(Schema.IfaceType ifaceType) {
        return (ifaceType.iface.getName().equals(BF16.class.getName())
                || ifaceType.iface.getName().equals(BF16Array.BF16Impl.class.getName()));
    }

    public final T typedef(BoundSchema<?> boundSchema, Schema.IfaceType ifaceType) {
        typedefKeyword()
                .space()
                .structOrUnion(ifaceType instanceof Schema.IfaceType.Struct)
                .space()
                .suffix_s(ifaceType.iface.getSimpleName())
                .braceNlIndented(_ -> {
                    int fieldCount = ifaceType.fields.size();
                    var fieldIdx = StreamMutable.of(0);
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
    final  T identifierWithRandomSuffix(String prefix, final int len) {
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
    public T hatBarrierOp(ScopedCodeBuilderContext buildContext, HATBarrierOp barrierOp) {
        return HAT_BARRIER();
    }

    @Override
    public final T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATLocalVarOp hatLocalVarOp) {
        return   localDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
    }

    @Override
    public final T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HATMemoryVarOp.HATPrivateVarOp hatLocalVarOp) {
        return privateDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
    }

    public abstract T defines();

    public T types() {
        return
                 typedefKeyword().space().s08Type("byte").semicolonNl()
                .typedefKeyword().space().s08Type("boolean").semicolonNl()
                .typedefStruct(KernelContext.class, _ -> s32Type("dimensions").semicolon()).nl();
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        var fieldAccess = fieldAccessOpHelper(buildContext.lookup,fieldLoadOp);
        if (fieldAccess.operandCount()==0 && fieldAccess.isPrimitive()) {
            literal(fieldAccess.getStaticFinalPrimitiveValue().toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        if (javaType instanceof ClassType classType && isAssignable(buildContext.lookup, javaType, MappableIface.class)) {
            HAT_GLOBAL_MEM().space().suffix_t(classType).asterisk();
        } else if (OpTkl.isAssignable(buildContext.lookup, javaType,KernelContext.class)) {
            HAT_GLOBAL_MEM().space().suffix_t(KernelContext.class).asterisk();
        } else if (OpTkl.isAssignable(buildContext.lookup, javaType,F16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().space().suffix_t(F16Impl.class).asterisk();
        } else if (OpTkl.isAssignable(buildContext.lookup, javaType,BF16.class)) {// TODO: update this with a custom op, to avoid direct use of Impls
            HAT_GLOBAL_MEM().space().suffix_t(BF16Array.BF16Impl.class).asterisk();
        } else {
            typeName(javaType.toString());
        }
        return self();
    }

    public T kernelMethod(ScopedCodeBuilderContext buildContext,CoreOp.FuncOp funcOp) {
          buildContext.funcScope(funcOp, () -> {
              nl();
              functionDeclaration(buildContext,(JavaType) funcOp.body().yieldType(), funcOp);
              parenNlIndented(_ ->
                    commaNlSeparated(
                            new FuncOpParams(funcOp).list(),
                            param -> declareParam(buildContext,param)
                    )
              );

              braceNlIndented(_ ->
                nlSeparated(
                        statements(funcOp.bodies().getFirst().entryBlock()),
                        statement->statement(buildContext,statement)
                )
              );
          });
        return self();
    }

    public T kernelEntrypoint(ScopedCodeBuilderContext buildContext) {
        nl();
        buildContext.funcScope(buildContext.funcOp, () ->
                kernelDeclaration(buildContext.funcOp)
                .parenNlIndented(_ -> commaNlSeparated(
                    buildContext.paramTable.list(),
                    param -> declareParam(buildContext,param))
                )
                .braceNlIndented(_ -> nlSeparated(
                    statements(buildContext.funcOp.bodies().getFirst().entryBlock()),
                    statement ->statement(buildContext,statement)
                )
            )
        );
        return self();
    }


    @Override
    public T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp) {
        return varName(hatVectorVarLoadOp);
    }

    public final T f16Type() {
        return suffix_t(F16.class);
    }

    public final T bf16Type() {
        return suffix_t(BF16.class);
    }

    @Override
    public T hatF16VarOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16VarOp hatF16VarOp) {
        ReducedFloatType reducedFloatType = hatF16VarOp.reducedFloatType();
        return (switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> f16Type();
            case ReducedFloatType.BFloat16 _ ->  bf16Type();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }).space().assign(
                _-> identifier(hatF16VarOp.varName()),
                _->recurse(buildContext, asResultOrThrow(hatF16VarOp.operands().getFirst()).op()));
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

    private T binaryOperationsForBfloat16(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16BinaryOp hatf16BinaryOp) {
        byte f32Mixed = hatf16BinaryOp.getByteFloatRepresentation();
        paren(_-> bf16Type());
        brace(_-> {
            paren(_-> {
                builtin_float2bfloat16();
                oparen();
                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();// open
                }
                recurse(buildContext, asResultOrThrow(hatf16BinaryOp.operands().getFirst()).op());

                List<Boolean> references = hatf16BinaryOp.references();
                if (references.getFirst()) {
                    rarrow().identifier("value");
                } else if (!isPrimitiveResult(hatf16BinaryOp.operands().getFirst())) {
                    dot().identifier("value");
                }else{
                    //throw new IllegalStateException("what happens here 1");
                }

                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    cparen(); //closed
                }
                space().identifier(hatf16BinaryOp.binaryOperationType().symbol()).space();

                if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();
                }

                recurse(buildContext, asResultOrThrow(hatf16BinaryOp.operands().get(1)).op());
                if (references.get(1)) {
                    rarrow().identifier("value");
                } else if (!isPrimitiveResult(hatf16BinaryOp.operands().get(1))) {
                    dot().identifier("value");
                } else{
                      //  throw new IllegalStateException("what happens here 2");
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
    public T hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16BinaryOp hatF16BinaryOp) {
        ReducedFloatType reducedFloatType = hatF16BinaryOp.reducedFloatType();
        if (reducedFloatType instanceof ReducedFloatType.BFloat16) {
            return binaryOperationsForBfloat16(buildContext, hatF16BinaryOp);
        }
        paren(_-> f16Type());
        return brace(_->
            paren(_-> {
                recurse(buildContext, asResultOrThrow(hatF16BinaryOp.operands().getFirst()).op());
                if (hatF16BinaryOp.references().getFirst()) {
                    rarrow().identifier("value");
                } else if (!isPrimitiveResult(hatF16BinaryOp.operands().getFirst())) {
                    dot().identifier("value");
                } else {
                    blockComment("hatF16BinaryOp not a result !!");
                }
                space().identifier(hatF16BinaryOp.binaryOperationType().symbol()).space();
                recurse(buildContext, asResultOrThrow(hatF16BinaryOp.operands().get(1)).op());
                if (hatF16BinaryOp.references().get(1)) {
                    rarrow().identifier("value");
                } else if (!isPrimitiveResult(hatF16BinaryOp.operands().get(1))) {
                    dot().identifier("value");
                }else {
                    blockComment("hatF16BinaryOp not a value !!");
                }
            })
        );
    }

    @Override
    public T hatF16VarLoadOp(ScopedCodeBuilderContext buildContext, HATF16Op.HATF16VarLoadOp hatF16VarLoadOp) {
        return identifier(hatF16VarLoadOp.varName()).dot().identifier("value");
    }

    @Override
    public T hatVectorMakeOf(ScopedCodeBuilderContext builderContext, HATVectorOp.HATVectorMakeOfOp hatVectorMakeOfOp) {
        return identifier(hatVectorMakeOfOp.varName());
    }

    public abstract T genVectorIdentifier(ScopedCodeBuilderContext builderContext, HATVectorOp.HATVectorOfOp hatVectorOfOp);

    @Override
    public T hatVectorOfOps(ScopedCodeBuilderContext buildContext, HATVectorOp.HATVectorOfOp hatVectorOp) {
        return genVectorIdentifier(buildContext, hatVectorOp)
                .paren(_->commaSpaceSeparated(
                        hatVectorOp.operands(),
                        operand -> recurse(buildContext, asResultOrThrow(operand).op()))
                );
    }

    @Override
    public T hatPrivateVarInitOp(ScopedCodeBuilderContext builderContext, HATMemoryVarOp.HATPrivateInitVarOp hatPrivateInitVarOp) {
        return suffix_t(hatPrivateInitVarOp.classType()).space()
                .assign(
                        _-> identifier(hatPrivateInitVarOp.varName()),
                        _->recurse(builderContext,asResultOrThrow(hatPrivateInitVarOp.operands().getFirst()).op()));
    }

    @Override
    public T hatMemoryLoadOp(ScopedCodeBuilderContext builderContext, HATMemoryDefOp.HATMemoryLoadOp hatMemoryLoadOp) {
        return recurse(builderContext, asResultOrThrow(hatMemoryLoadOp.operands().getFirst()).op())
                .dot().identifier(hatMemoryLoadOp.memberName())
                .when(hatMemoryLoadOp.operands().size() > 1,_->// If the hatMemoryLoadOp has more than 1 operand, the second is the index
                   sbrace(_-> recurse(builderContext, asResultOrThrow(hatMemoryLoadOp.operands().get(1)).op()))
                );
    }

    public T hatPtrLoadOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrLoadOp hatPtrLoadOp) {
        ptrAccess(builderContext, hatPtrLoadOp);
        return self();
    }

    @Override
    public T hatPtrStoreOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrStoreOp hatPtrStoreOp) {
        ptrAccess(builderContext, hatPtrStoreOp).equals().recurse(builderContext, ((Op.Result) hatPtrStoreOp.operands().getLast()).op());
        return self();
    }

    @Override
    public T hatPtrLengthOp(ScopedCodeBuilderContext builderContext, HATPtrOp.HATPtrLengthOp hatPtrLengthOp) {
        ptrAccess(builderContext, hatPtrLengthOp);
        return self();
    }

    T ptrAccess(ScopedCodeBuilderContext builderContext, HATPtrOp hatPtrOp) {
        identifier(hatPtrName(hatPtrOp));
        boolean isLocalOrPrivateDS = false;
        if (((Op.Result) hatPtrOp.operands().getFirst()).op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            Op resolve = builderContext.scope.resolve(varLoadOp.operands().getFirst());
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
                        paren(_ -> recurse(builderContext, ((Op.Result) hatPtrOp.operands().get(2)).op()));
                        asterisk().identifier(hatPtrName(hatPtrOp));
                        either(finalIsLocalOrPrivateDS, CodeBuilder::dot, CodeBuilder::rarrow).identifier(hatPtrOp.strides() != null ? hatPtrOp.strides().getFirst() : "width");
                        add().paren(_ -> recurse(builderContext, ((Op.Result) hatPtrOp.operands().get(1)).op()));
                    } else {
                        recurse(builderContext, ((Op.Result) hatPtrOp.operands().get(1)).op());
                    }
                });
            });
        }
        return self();
    }

    public String hatPtrName(HATPtrOp hatPtrOp) {
        Op op = ((Op.Result) ((Op.Result) (hatPtrOp.operands().getFirst())).op().operands().getFirst()).op();
        return switch (op) {
            case CoreOp.VarOp varOp -> varOp.varName();
            case HATMemoryVarOp.HATLocalVarOp hatLocalVarOp -> hatLocalVarOp.varName();
            case HATMemoryVarOp.HATPrivateVarOp hatPrivateVarOp -> hatPrivateVarOp.varName();
            case null, default -> "";
        };
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
     *      b16_t b1 = {f};
     *      return b1.s[1];
     * }
     * </code>
     * @param parameterName
     * @return
     */
    public final T build_builtin_float2bfloat16(String parameterName) {
        String b16 = "b16";
        String s = "s";
        return funcDef(
                _ -> u16Type(),
                _ -> builtin_float2bfloat16(),
                _ -> f32Type(parameterName),
                _ -> assign(_ -> bfloat16Type(b16),
                        _ ->  brace( _ -> identifier(parameterName)).semicolonNl()
                                .returnKeyword(_ ->identifier(b16).dot().identifier(s).sbrace(_ -> intConstOne()))));
    }
}
