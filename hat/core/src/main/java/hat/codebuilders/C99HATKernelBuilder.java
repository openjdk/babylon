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

import hat.buffer.Buffer;
import hat.dialect.HatBlockThreadIdOp;
import hat.dialect.HatGlobalThreadIdOp;
import hat.dialect.HatGlobalSizeOp;
import hat.dialect.HatLocalSizeOp;
import hat.dialect.HatLocalThreadIdOp;
import hat.ifacemapper.MappableIface;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.function.Consumer;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends HATCodeBuilderWithContext<T> {
    public T types() {
        return this
                .charTypeDefs("byte", "boolean")
                .typedefStructOrUnion(true, "KernelContext", _ -> {
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

    T typedefStructOrUnion(boolean isStruct, String name, Consumer<T> consumer) {
        return typedefKeyword()
                .space()
                .structOrUnion(isStruct)
                .space()
                .either(isStruct, _ -> suffix_s(name), _ -> suffix_u(name))
                .braceNlIndented(consumer)
                .suffix_t(name).semicolon().nl();
    }

    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        if (OpTk.isAssignable(buildContext.lookup, javaType, MappableIface.class) && javaType instanceof ClassType classType) {
            globalPtrPrefix().space().suffix_t(classType).asterisk();
        }else if (javaType instanceof ClassType classType && classType.toClassName().equals("hat.KernelContext")){
            globalPtrPrefix().space().suffix_t("KernelContext").asterisk();
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
        return localPtrPrefix().space() // we should be able to compose-call to privateDeclaration?
                .suffix_t(localArrayDeclaration.classType()).space().varName(localArrayDeclaration.varOp());
    }

    @Override
    public T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HatGlobalThreadIdOp globalThreadIdOp) {
        globalId(globalThreadIdOp.getDimension());
        return self();
    }

    @Override
    public T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HatGlobalSizeOp globalSizeOp) {
        globalSize(globalSizeOp.getDimension());
        return self();
    }

    @Override
    public T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HatLocalThreadIdOp localThreadIdOp) {
        localId(localThreadIdOp.getDimension());
        return self();
    }

    @Override
    public T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HatLocalSizeOp hatLocalSizeOp) {
        localSize(hatLocalSizeOp.getDimension());
        return self();
    }

    @Override
    public T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HatBlockThreadIdOp hatBlockThreadIdOp) {
        blockId(hatBlockThreadIdOp.getDimension());
        return self();
    }

    public abstract T globalPtrPrefix();

    public abstract T localPtrPrefix();

    public abstract T defines();

    public abstract T kernelDeclaration(CoreOp.FuncOp funcOp);

    public abstract T functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp);

    public abstract T globalId(int id);

    public abstract T localId(int id);

    public abstract T globalSize(int id);

    public abstract T localSize(int id);

    public abstract T blockId(int id);

}
