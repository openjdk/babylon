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
import jdk.incubator.code.Op;
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
            globalPtrPrefix().suffix_t(classType).asterisk();
        }else if (javaType instanceof ClassType classType && classType.toClassName().equals("hat.KernelContext")){
            globalPtrPrefix().suffix_t("KernelContext").asterisk();
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



    public T globalId(int id) {
        switch (id) {
            case 0 -> identifier("_gix()");
            case 1 -> identifier("_giy()");
            case 2 -> identifier("_giz()");
            default -> throw new RuntimeException("globalId id = " + id);
        }
        return self();
    }

    public T localId(int id) {
        switch (id) {
            case 0 -> identifier("_lix()");
            case 1 -> identifier("_liy()");
            case 2 -> identifier("_liz()");
            default -> throw new RuntimeException("localId id = " + id);
        }
        return self();
    }

    public T globalSize(int id) {
        switch (id) {
            case 0 -> identifier("_gsx()");
            case 1 -> identifier("_gsy()");
            case 2 -> identifier("_gsz()");
            default -> throw new RuntimeException("globalSize id = " + id);
        }
        return self();
    }

    public T localSize(int id) {
        switch (id) {
            case 0 -> identifier("_lsx()");
            case 1 -> identifier("_lsy()");
            case 2 -> identifier("_lsz()");
            default -> throw new RuntimeException("localSize id = " + id);
        }
        return self();
    }


    public T blockId(int id) {
        switch (id) {
            case 0 -> identifier("_bix()");
            case 1 -> identifier("_biy()");
            case 2 -> identifier("_biz()");
            default -> throw new RuntimeException("blockId id = " + id);
        }
        return self();
    }
 //   public abstract T kernelPrefix();


    public T kernelDeclaration(CoreOp.FuncOp funcOp) {
        return kernelPrefix().voidType().space().funcName(funcOp);
    }
   // public abstract  T functionPrefix();


    public T functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp) {
        return functionPrefix().type(codeBuilderContext,javaType).space().funcName(funcOp);
    }

    public T kernelPrefix() {
        return keyword("_KERNEL").space();
    }

    public T functionPrefix() {
        return keyword("_FUNC").space();
    }

    public T globalPtrPrefix() {
        return keyword("_GLOBAL_MEM").space();
    }

    public T localPtrPrefix() {
        return keyword("_LOCAL_MEM").space();
    }

    public T syncBlockThreads() {
        return identifier("_barrier").ocparen();
    }


  //  public abstract T globalPtrPrefix();

  //  public abstract T localPtrPrefix();

    public abstract T defines();


}
