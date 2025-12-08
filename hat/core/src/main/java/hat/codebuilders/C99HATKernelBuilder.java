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

import hat.buffer.BF16;
import hat.buffer.BF16Array;
import hat.KernelContext;
import hat.buffer.Buffer;
import hat.buffer.F16;
import hat.buffer.F16Array;
import hat.dialect.HATBarrierOp;
import hat.dialect.HATBlockThreadIdOp;
import hat.dialect.HATF16BinaryOp;
import hat.dialect.HATF16VarLoadOp;
import hat.dialect.HATF16VarOp;
import hat.dialect.HATGlobalSizeOp;
import hat.dialect.HATGlobalThreadIdOp;
import hat.dialect.HATLocalSizeOp;
import hat.dialect.HATLocalThreadIdOp;
import hat.dialect.HATLocalVarOp;
import hat.dialect.HATMemoryLoadOp;
import hat.dialect.HATMemoryOp;
import hat.dialect.HATPrivateInitVarOp;
import hat.dialect.HATPrivateVarOp;
import hat.dialect.HATVectorMakeOfOp;
import hat.dialect.HATVectorOfOp;
import hat.dialect.HATVectorVarLoadOp;
import hat.dialect.ReducedFloatType;
import hat.ifacemapper.BoundSchema;
import hat.ifacemapper.MappableIface;
import hat.ifacemapper.Schema;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import hat.util.StreamMutable;
import jdk.incubator.code.Op;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.List;
import java.util.function.Consumer;

import static hat.buffer.F16Array.F16Impl;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends C99HATCodeBuilderContext<T>implements BabylonKernelOpBuilder<T>  {

    public T kernelDeclaration(CoreOp.FuncOp funcOp) {
        return kernelPrefix().voidType().space().funcName(funcOp);
    }

    public T functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp) {
        return functionPrefix().type(codeBuilderContext,javaType).space().funcName(funcOp);
    }

    public T kernelPrefix() {
        return keyword("HAT_KERNEL").space();
    }

    public T functionPrefix() {
        return keyword("HAT_FUNC").space();
    }

    public T globalPtrPrefix() {
        return keyword("HAT_GLOBAL_MEM").space();
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
                                if (array instanceof Schema.FieldNode.PrimitiveFieldControlledArray fieldControlledArray) {
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
                                        if (boundSchema != null) {
                                            boolean[] done = new boolean[]{false};
                                            boundSchema.boundArrayFields().forEach(a -> {
                                                if (a.field.equals(ifaceField)) {
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
                                } else if (array instanceof Schema.FieldNode.IfaceFixedArray fixed) {
                                    sbrace(_ -> literal(Math.max(1, fixed.len)));
                                } else {
                                    throw new IllegalStateException("what kind of array ");
                                }
                            }
                        } else if (field instanceof Schema.SchemaNode.Padding padding) {
                            emitText(padding.toC99());
                        } else {
                            throw new IllegalStateException("hmm");
                        }
                        fieldIdx.set(fieldIdx.get() + 1);
                    });
                }).suffix_t(ifaceType.iface).semicolon().nl().nl();
        return self();
    }

     public record LocalArrayDeclaration(ClassType classType, HATMemoryOp varOp) {}


    public final T privateDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return suffix_t(localArrayDeclaration.classType()).space().varName(localArrayDeclaration.varOp());
    }
    public T localPtrPrefix() {
        return keyword("HAT_LOCAL_MEM").space();
    }
    public final T localDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        return localPtrPrefix() // we should be able to compose-call to privateDeclaration?
                .suffix_t(localArrayDeclaration.classType())
                .space()
                .varName(localArrayDeclaration.varOp());
    }

    @Override
    public T hatBarrierOp(ScopedCodeBuilderContext buildContext, HATBarrierOp barrierOp) {
        return identifier("HAT_BARRIER");
    }



    @Override
    public final T hatLocalVarOp(ScopedCodeBuilderContext buildContext, HATLocalVarOp hatLocalVarOp) {
        return   localDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
    }

    @Override
    public final T hatPrivateVarOp(ScopedCodeBuilderContext buildContext, HATPrivateVarOp hatLocalVarOp) {
        return privateDeclaration(new LocalArrayDeclaration(hatLocalVarOp.classType(), hatLocalVarOp));
    }
    public abstract T defines();

    public T types() {
        return charTypeDefs("byte", "boolean")
                .nl()
                .typedefStructOrUnion(true, KernelContext.class, _ ->
                    intDeclaration("dimensions").semicolon()
                )
                .nl();
    }
    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) {
            literal(OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp).toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        if (OpTk.isAssignable(buildContext.lookup, javaType, MappableIface.class) && javaType instanceof ClassType classType) {
            globalPtrPrefix().suffix_t(classType).asterisk();
        } else if (javaType instanceof ClassType classType && classType.toClassName().equals(KernelContext.class.getName())) {
            globalPtrPrefix().suffix_t(KernelContext.class).asterisk();
        } else if (javaType instanceof ClassType classType && classType.toClassName().equals(F16.class.getCanonicalName())) {
            // Check for special types (e.g., FP16)
            // TODO: We need to update this with a custom op, so we avoid direct use of Impls
            globalPtrPrefix().suffix_t(F16Impl.class).asterisk();
        } else if (javaType instanceof ClassType classType && classType.toClassName().equals(BF16.class.getCanonicalName())) {
            // Special type: BFLOAT16
            // TODO: We need to update this with a custom op, so we avoid direct use of Impls
            globalPtrPrefix().suffix_t(BF16Array.BF16Impl.class).asterisk();
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
                        OpTk.statements(funcOp.bodies().getFirst().entryBlock()),
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
                    OpTk.statements(buildContext.funcOp.bodies().getFirst().entryBlock()),
                    statement ->statement(buildContext,statement)
                )
            )
        );
        return self();
    }


    @Override
    public T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HATGlobalThreadIdOp globalThreadIdOp) {
        return globalId(globalThreadIdOp.getDimension());
    }

    @Override
    public T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HATGlobalSizeOp globalSizeOp) {
        return globalSize(globalSizeOp.getDimension());
    }

    @Override
    public T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HATLocalThreadIdOp localThreadIdOp) {
        return localId(localThreadIdOp.getDimension());
    }

    @Override
    public T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HATLocalSizeOp hatLocalSizeOp) {
        return localSize(hatLocalSizeOp.getDimension());
    }

    @Override
    public T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HATBlockThreadIdOp hatBlockThreadIdOp) {
        return blockId(hatBlockThreadIdOp.getDimension());
    }

    public T globalId(int id) {
        switch (id) {
            case 0 -> identifier("HAT_GIX");
            case 1 -> identifier("HAT_GIY");
            case 2 -> identifier("HAT_GIZ");
            default -> throw new RuntimeException("globalId id = " + id);
        }
        return self();
    }

    public T localId(int id) {
        return (switch (id) {
            case 0 -> identifier("HAT_LIX");
            case 1 -> identifier("HAT_LIY");
            case 2 -> identifier("HAT_LIZ");
            default -> throw new RuntimeException("localId id = " + id);
        });
    }

    public T globalSize(int id) {
        return (switch (id) {
            case 0 -> identifier("HAT_GSX");
            case 1 -> identifier("HAT_GSY");
            case 2 -> identifier("HAT_GSZ");
            default -> throw new RuntimeException("globalSize id = " + id);
        });
    }

    public T localSize(int id) {
        return (switch (id) {
            case 0 -> identifier("HAT_LSX");
            case 1 -> identifier("HAT_LSY");
            case 2 -> identifier("HAT_LSZ");
            default -> throw new RuntimeException("localSize id = " + id);
        });
    }


    public T blockId(int id) {
        return (switch (id) {
            case 0 -> identifier("HAT_BIX");
            case 1 -> identifier("HAT_BIY");
            case 2 -> identifier("HAT_BIZ");
            default -> throw new RuntimeException("blockId id = " + id);
        });
    }

    @Override
    public T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HATVectorVarLoadOp hatVectorVarLoadOp) {
        return varName(hatVectorVarLoadOp);
    }

    @Override
    public T hatF16VarOp(ScopedCodeBuilderContext buildContext, HATF16VarOp hatF16VarOp) {
        ReducedFloatType reducedFloatType = hatF16VarOp.reducedFloatType();
        return (switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> halfType();
            case ReducedFloatType.BFloat16 _ ->  bfloatType();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        })
                .space()
                .identifier(hatF16VarOp.varName())
                .space()
                .equals()
                .space()
                .recurse(buildContext, OpTk.asResultOrThrow(hatF16VarOp.operands().getFirst()).op());
    }

    private boolean isMixedFirstOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16BinaryOp.FIRST_OP;
    }

    private boolean isMixedSecondOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16BinaryOp.LAST_OP;
    }
    public final T builtin_float2bfloat16() {
        return identifier("floatTobfloat16");
    }

    public final T builtin_bfloat16ToFloat() {
        return identifier("bfloat16Tofloat");
    }
    private T binaryOperationsForBfloat16(ScopedCodeBuilderContext buildContext, HATF16BinaryOp hatf16BinaryOp) {

        byte f32Mixed = hatf16BinaryOp.getF32();

        paren(_->bfloatType());
        brace(_-> {
            paren(_-> {
                builtin_float2bfloat16();
                oparen();
                if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
                    builtin_bfloat16ToFloat().oparen();// open
                }
                recurse(buildContext, OpTk.asResultOrThrow(hatf16BinaryOp.operands().get(0)).op());

                List<Boolean> references = hatf16BinaryOp.references();
                if (references.getFirst()) {
                    rarrow().identifier("value");
                } else if (hatf16BinaryOp.operands().get(0) instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
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

                recurse(buildContext, OpTk.asResultOrThrow(hatf16BinaryOp.operands().get(1)).op());
                if (references.get(1)) {
                    rarrow().identifier("value");
                } else if (hatf16BinaryOp.operands().get(1) instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
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
    public T hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16BinaryOp hatF16BinaryOp) {
        ReducedFloatType reducedFloatType = hatF16BinaryOp.reducedFloatType();
        if (reducedFloatType instanceof ReducedFloatType.BFloat16) {
            return binaryOperationsForBfloat16(buildContext, hatF16BinaryOp);
        }
        paren(_-> halfType());
        return brace(_->
            paren(_-> {
                recurse(buildContext, OpTk.asResultOrThrow(hatF16BinaryOp.operands().getFirst()).op());
                if (hatF16BinaryOp.references().getFirst()) {
                    rarrow().identifier("value");
                } else if (hatF16BinaryOp.operands().getFirst() instanceof Op.Result r2 && !(r2.op().resultType() instanceof PrimitiveType)) {
                    dot().identifier("value");
                } else {
                    blockComment("hatF16BinaryOp not a result !!");
                }
                space().identifier(hatF16BinaryOp.binaryOperationType().symbol()).space();
                recurse(buildContext, OpTk.asResultOrThrow(hatF16BinaryOp.operands().get(1)).op());
                if (hatF16BinaryOp.references().get(1)) {
                    rarrow().identifier("value");
                } else if (hatF16BinaryOp.operands().get(1) instanceof Op.Result r4 && !(r4.op().resultType() instanceof PrimitiveType)) {
                    dot().identifier("value");
                }else {
                    blockComment("hatF16BinaryOp not a value !!");
                }
            })
        );
    }

    @Override
    public T hatF16VarLoadOp(ScopedCodeBuilderContext buildContext, HATF16VarLoadOp hatF16VarLoadOp) {
        return identifier(hatF16VarLoadOp.varName()).dot().identifier("value");
    }

    @Override
    public T hatVectorMakeOf(ScopedCodeBuilderContext builderContext, HATVectorMakeOfOp hatVectorMakeOfOp) {
        return identifier(hatVectorMakeOfOp.varName());
    }

    public abstract T genVectorIdentifier(ScopedCodeBuilderContext builderContext, HATVectorOfOp hatVectorOfOp);

    @Override
    public T hatVectorOfOps(ScopedCodeBuilderContext buildContext, HATVectorOfOp hatVectorOp) {
        return genVectorIdentifier(buildContext, hatVectorOp)
                .paren(_->commaSpaceSeparated(
                        hatVectorOp.operands(),
                        operand -> recurse(buildContext, OpTk.asResultOrThrow(operand).op()))
                );
    }

    @Override
    public T hatPrivateVarInitOp(ScopedCodeBuilderContext builderContext, HATPrivateInitVarOp hatPrivateInitVarOp) {
        return suffix_t(hatPrivateInitVarOp.classType())
                .space()
                .identifier(hatPrivateInitVarOp.varName())
                .space()
                .equals()
                .space()
                .recurse(builderContext,OpTk.asResultOrThrow(hatPrivateInitVarOp.operands().getFirst()).op());
    }

    @Override
    public T hatMemoryLoadOp(ScopedCodeBuilderContext builderContext, HATMemoryLoadOp hatMemoryLoadOp) {
        return recurse(builderContext, OpTk.asResultOrThrow(hatMemoryLoadOp.operands().getFirst()).op())
                .dot().identifier(hatMemoryLoadOp.memberName())
                .when(hatMemoryLoadOp.operands().size() > 1,_->// If the hatMemoryLoadOp has more than 1 operand, the second is the index
                   sbrace(_-> recurse(builderContext, OpTk.asResultOrThrow(hatMemoryLoadOp.operands().get(1)).op()))
                );
    }

    public final T buildStructSingleValueMember(String structName,  String type) {
       return typedefStruct(structName,_-> typeName(type).space().identifier("value").semicolonNl());
    }

    public final T buildForLoopHeader(String loopVar, String init, String loopBound) {
        forKeyword().paren(_ -> intType().space().identifier(loopVar).space().equals().identifier(init).semicolon().space()
                .identifier(loopVar).lt().identifier(loopBound).semicolon().space()
                .identifier(loopVar).plusplus());
        return self();
    }

    //public final T call_builtin_byteCopy(){
      //  return identifier("");
   // }

    public final T funcDef(Consumer<T> type,Consumer<T> name, Consumer<T> args, Consumer<T> body){
         type.accept(self());
         space();
         name.accept(self());
         paren(args);
         braceNlIndented(body);
         return nl();
    }
    public final T assign(Consumer<T> lhs,Consumer<T> rhs){
        lhs.accept(self());
        space().equals().space();
        rhs.accept(self());
        return semicolonNl();
    }
    public final T cast(Consumer<T> type){
        return paren(_-> type.accept(self()));
    }
    public final T returnKeyword(Consumer<T> exp){
        return returnKeyword().space().paren(_-> exp.accept(self())).semicolon();
    }

    public final T call(Consumer<T> name,Consumer<T> ...args){
        name.accept(self());
        return paren(_->commaSpaceSeparated(args));
    }
    public final T call(String name,Consumer<T> ...args){
        return call(_->identifier(name),args);
    }

    /**
     * <code>
     *  void byteCopy(void *dest, const void* src, size_t size) {
     *      unsigned char *c = (unsigned char*)dest;
     *      unsigned char *s = (unsigned char*)src;
     *      for (int i = 0; i < size; i++) {
     *          *c++ = *s++;
     *      }
     *  }
     * </code>
     * @return
     */

    public final T build_builtin_byteCopy() {
        return funcDef(
                _->voidType(),
                _->identifier("byteCopy"),
                _->commaSpaceSeparated(
                        _-> voidPtrType("dest"),
                        _-> voidPtrType("src"),
                        _-> size_t("size")
                ),
                _ ->
                   assign(_->u08PtrType("c"), _->cast(_ -> u08PtrType()).identifier("dest"))
                   .assign(_->u08PtrType("s"), _->cast(_ -> u08PtrType()).identifier("src"))
                   .buildForLoopHeader("i", "0", "size").braceNlIndented(_ ->
                        dereference("c").plusplus().equals().dereference("s").plusplus().semicolon()
                   )
                );
    }

    /**
     * <code>
     *  float bfloat16Tofloat(ushort bf16) {
     *      uint bitsRecovered = bf16 << 16;
     *      float r = bitsRecovered;
     *      byteCopy(&r, &bitsRecovered, sizeof(r));
     *      return r;
     * }
     * </code>
     *
     * @param parameterName
     * @return
     */
    public final T build_builtin_bfloat16ToFloat(String parameterName) {
        return funcDef(_->f32Type(),
                _->builtin_bfloat16ToFloat(),
                _->unsignedShortType(parameterName),
                 _ -> assign(_->u32Type("bits"), _->identifier(parameterName).leftShift(16))
                      .f32Type("r").semicolonNl()
                      .call("byteCopy",_->addressOf("r"),_->addressOf("bits"),_->sizeof("r"))
                      .semicolonNl()
                      .returnKeyword(_->identifier("r"))
                );
    }

    /**
     * <code>
     * ushort floatTobfloat16(float f) {
     *      uint bits;
     *      byteCopy(&bits, &f, sizeof(bits));
     *      short bf16 = bits >> 16;
     *      return bf16;
     * }
     * </code>
     * @param parameterName
     * @return
     */
    public final T build_builtin_float2bfloat16(String parameterName) {
        return funcDef(
                _->shortType(),
                _->builtin_float2bfloat16(),
                _->f32Type(parameterName),
                _->
                   u32Type("bits").semicolonNl()
                   .call("byteCopy", _->addressOf("bits"), _->addressOf(parameterName), _->sizeof("bits")).semicolonNl()
                   .returnKeyword(_->identifier("bits").rightShift(16))
                );
    }

}
