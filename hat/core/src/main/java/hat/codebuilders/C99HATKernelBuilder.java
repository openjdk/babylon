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
import hat.dialect.HATBlockThreadIdOp;
import hat.dialect.HATF16BinaryOp;
import hat.dialect.HATF16VarLoadOp;
import hat.dialect.HATF16VarOp;
import hat.dialect.HATGlobalSizeOp;
import hat.dialect.HATGlobalThreadIdOp;
import hat.dialect.HATLocalSizeOp;
import hat.dialect.HATLocalThreadIdOp;
import hat.dialect.HATMemoryLoadOp;
import hat.dialect.HATPrivateInitVarOp;
import hat.dialect.HATVectorMakeOfOp;
import hat.dialect.HATVectorOfOp;
import hat.dialect.HATVectorVarLoadOp;
import hat.dialect.ReducedFloatType;
import hat.ifacemapper.MappableIface;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
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

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends HATCodeBuilderWithContext<T> {

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

    public T localPtrPrefix() {
        return keyword("HAT_LOCAL_MEM").space();
    }

    public T syncBlockThreads() {
        return identifier("HAT_BARRIER");
    }

    public abstract T defines();

    public T types() {
        return this
                .charTypeDefs("byte", "boolean")
                .typedefStructOrUnion(true, KernelContext.class, _ -> {
                    intDeclaration("dimensions").semicolon().nl();
                });
    }
    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) {
            Object value = OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("What is this field load ?" + fieldLoadOp);
        }
        return self();
    }

    T typedefStructOrUnion(boolean isStruct, Class<?> klass, Consumer<T> consumer) {
        return typedefKeyword()
                .space()
                .structOrUnion(isStruct)
                .space()
                .either(isStruct, _ -> suffix_s(klass), _ -> suffix_u(klass))
                .braceNlIndented(consumer)
                .suffix_t(klass).semicolon().nl();
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
              var paramTable = new FuncOpParams(funcOp);
              parenNlIndented(_ ->
                    separated(paramTable.list(),(_)->comma().nl(), param ->
                        declareParam(buildContext,param)
                    )
              );

              braceNlIndented(_ ->
                separated(OpTk.statements(funcOp.bodies().getFirst().entryBlock()),(_)->nl(),
                        statement->statement(buildContext,statement)
                )
              );
          });
        return self();
    }

    public T kernelEntrypoint(ScopedCodeBuilderContext buildContext, Object... args) {
        nl();
        buildContext.funcScope(buildContext.funcOp, () -> {
            kernelDeclaration(buildContext.funcOp);
            // We skip the first arg which was KernelContext.
            var list = buildContext.paramTable.list();
            for (int arg = 0; arg < args.length; arg++) {
                if (args[arg] instanceof Buffer) {
                    list.get(arg).setClass(args[arg].getClass());  // de we have to do this?
                }
            }
            parenNlIndented(_ -> separated(list.stream(),(_)->comma().nl(),param -> declareParam(buildContext,param)));

            braceNlIndented(_ -> {
                 separated(OpTk.statements(buildContext.funcOp.bodies().getFirst().entryBlock()), (_)->nl(),
                        statement ->statement(buildContext,statement)
                );
            });
        });
        return self();
    }

    public T privateDeclaration(HATCodeBuilderWithContext.LocalArrayDeclaration localArrayDeclaration) {
        return suffix_t(localArrayDeclaration.classType()).space().varName(localArrayDeclaration.varOp()).nl();
    }

    public T localDeclaration(HATCodeBuilderWithContext.LocalArrayDeclaration localArrayDeclaration) {
        return localPtrPrefix() // we should be able to compose-call to privateDeclaration?
                .suffix_t(localArrayDeclaration.classType()).space().varName(localArrayDeclaration.varOp());
    }

    @Override
    public T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HATGlobalThreadIdOp globalThreadIdOp) {
        globalId(globalThreadIdOp.getDimension());
        return self();
    }

    @Override
    public T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HATGlobalSizeOp globalSizeOp) {
        globalSize(globalSizeOp.getDimension());
        return self();
    }

    @Override
    public T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HATLocalThreadIdOp localThreadIdOp) {
        localId(localThreadIdOp.getDimension());
        return self();
    }

    @Override
    public T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HATLocalSizeOp hatLocalSizeOp) {
        localSize(hatLocalSizeOp.getDimension());
        return self();
    }

    @Override
    public T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HATBlockThreadIdOp hatBlockThreadIdOp) {
        blockId(hatBlockThreadIdOp.getDimension());
        return self();
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
        switch (id) {
            case 0 -> identifier("HAT_LIX");
            case 1 -> identifier("HAT_LIY");
            case 2 -> identifier("HAT_LIZ");
            default -> throw new RuntimeException("localId id = " + id);
        }
        return self();
    }

    public T globalSize(int id) {
        switch (id) {
            case 0 -> identifier("HAT_GSX");
            case 1 -> identifier("HAT_GSY");
            case 2 -> identifier("HAT_GSZ");
            default -> throw new RuntimeException("globalSize id = " + id);
        }
        return self();
    }

    public T localSize(int id) {
        switch (id) {
            case 0 -> identifier("HAT_LSX");
            case 1 -> identifier("HAT_LSY");
            case 2 -> identifier("HAT_LSZ");
            default -> throw new RuntimeException("localSize id = " + id);
        }
        return self();
    }


    public T blockId(int id) {
        switch (id) {
            case 0 -> identifier("HAT_BIX");
            case 1 -> identifier("HAT_BIY");
            case 2 -> identifier("HAT_BIZ");
            default -> throw new RuntimeException("blockId id = " + id);
        }
        return self();
    }

    @Override
    public T hatVectorVarLoadOp(ScopedCodeBuilderContext buildContext, HATVectorVarLoadOp hatVectorVarLoadOp) {
        varName(hatVectorVarLoadOp);
        return self();
    }

    @Override
    public T hatF16VarOp(ScopedCodeBuilderContext buildContext, HATF16VarOp hatF16VarOp) {

        ReducedFloatType reducedFloatType = hatF16VarOp.reducedFloatType();
        switch (reducedFloatType) {
            case ReducedFloatType.HalfFloat _ -> halfType();
            case ReducedFloatType.BFloat16 _ ->  bfloatType();
            default -> throw new IllegalStateException("Unexpected value: " + reducedFloatType);
        }

        space().identifier(hatF16VarOp.varName())
                .space().equals().space();
        Value operand = hatF16VarOp.operands().getFirst();
        if (operand instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        return self();
    }

    private boolean isMixedFirstOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16BinaryOp.FIRST_OP;
    }

    private boolean isMixedSecondOperand(byte f32Mixed) {
        return f32Mixed != 0 && f32Mixed != HATF16BinaryOp.LAST_OP;
    }

    private T binaryOperationsForBfloat16(ScopedCodeBuilderContext buildContext, HATF16BinaryOp hatf16BinaryOp) {
        Value op1 = hatf16BinaryOp.operands().get(0);
        Value op2 = hatf16BinaryOp.operands().get(1);
        List<Boolean> references = hatf16BinaryOp.references();
        byte f32Mixed = hatf16BinaryOp.getF32();

        oparen().bfloatType()
                .cparen().obrace().oparen();

        builtin_float2bfloat16()
                .oparen();

        if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
            builtin_bfloat162float().oparen();
        }


        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        if (references.getFirst()) {
            rarrow().identifier("value");
        } else if (op1 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }

        if (isMixedFirstOperand(f32Mixed) || f32Mixed == 0) {
            cparen();
        }
        space().identifier(hatf16BinaryOp.binaryOperationType().symbol()).space();

        if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
            builtin_bfloat162float().oparen();
        }

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        if (references.get(1)) {
            rarrow().identifier("value");
        } else if (op2 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }

        if (isMixedSecondOperand(f32Mixed) || f32Mixed == 0) {
            cparen();
        }
        cparen().cparen().cbrace();
        return self();
    }

    @Override
    public T hatF16BinaryOp(ScopedCodeBuilderContext buildContext, HATF16BinaryOp hatF16BinaryOp) {

        ReducedFloatType reducedFloatType = hatF16BinaryOp.reducedFloatType();
        if (reducedFloatType instanceof ReducedFloatType.BFloat16) {
            return binaryOperationsForBfloat16(buildContext, hatF16BinaryOp);
        }

        Value op1 = hatF16BinaryOp.operands().get(0);
        Value op2 = hatF16BinaryOp.operands().get(1);
        List<Boolean> references = hatF16BinaryOp.references();

        oparen().halfType();

        cparen().obrace().oparen();
        if (op1 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }
        if (references.getFirst()) {
            rarrow().identifier("value");
        } else if (op1 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }
        space().identifier(hatF16BinaryOp.binaryOperationType().symbol()).space();

        if (op2 instanceof Op.Result r) {
            recurse(buildContext, r.op());
        }

        if (references.get(1)) {
            rarrow().identifier("value");
        } else if (op2 instanceof Op.Result r && !(r.op().resultType() instanceof PrimitiveType)) {
            dot().identifier("value");
        }

        cparen().cbrace();
        return self();
    }

    @Override
    public T hatF16VarLoadOp(ScopedCodeBuilderContext buildContext, HATF16VarLoadOp hatF16VarLoadOp) {
        identifier(hatF16VarLoadOp.varName());
        dot().identifier("value");
        return self();
    }

    @Override
    public T hatVectorMakeOf(ScopedCodeBuilderContext builderContext, HATVectorMakeOfOp hatVectorMakeOfOp) {
        identifier(hatVectorMakeOfOp.varName());
        return self();
    }

    public abstract T genVectorIdentifier(ScopedCodeBuilderContext builderContext, HATVectorOfOp hatVectorOfOp);

    @Override
    public T hatVectorOfOps(ScopedCodeBuilderContext buildContext, HATVectorOfOp hatVectorOp) {
        genVectorIdentifier(buildContext, hatVectorOp);

        List<Value> inputOperands = hatVectorOp.operands();
        int i;
        for (i = 0; i < (inputOperands.size() - 1); i++) {
            var operand = inputOperands.get(i);
            if ((operand instanceof Op.Result r)) {
                recurse(buildContext, r.op());
            }
            comma().space();
        }
        // Last parameter
        var operand = inputOperands.get(i);
        if ((operand instanceof Op.Result r)) {
            recurse(buildContext, r.op());
        }
        cparen();
        return self();
    }

    @Override
    public T hatPrivateVarInitOp(ScopedCodeBuilderContext builderContext, HATPrivateInitVarOp hatPrivateInitVarOp) {
        suffix_t(hatPrivateInitVarOp.classType()).space().identifier(hatPrivateInitVarOp.varName());
        space().equals().space();
        Value operand = hatPrivateInitVarOp.operands().getFirst();
        if (operand instanceof Op.Result r) {
            recurse(builderContext, r.op());
        }
        return self();
    }

    @Override
    public T hatMemoryLoadOp(ScopedCodeBuilderContext builderContext, HATMemoryLoadOp hatMemoryLoadOp) {
        List<Value> operands = hatMemoryLoadOp.operands();
        Value base = operands.get(0);
        if (base instanceof Op.Result r) {
           recurse(builderContext, r.op());
        }
        dot().identifier(hatMemoryLoadOp.memberName());

        if (operands.size() > 1) {
            // If the hatMemoryLoadOp has more than 1 operand,
            // then we know that the second operand represents
            // an index to access an array, since members, otherwise,
            // will be accessed via structVarName.member1.member2.member3...,  etc.

            // The following code generates [ indexValue ]
            osbrace();
            Value index = operands.get(1);
            if (index instanceof Op.Result r) {
                recurse(builderContext, r.op());
            }
            csbrace();
        }
        return self();
    }
}
