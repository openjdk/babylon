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

import hat.NDRange;
import hat.buffer.Buffer;
import hat.ifacemapper.MappableIface;
import hat.optools.FuncOpParams;
import hat.optools.OpTk;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.function.Consumer;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends HATCodeBuilderWithContext<T> {
    protected final NDRange ndRange; // Should be in the context ?
    public C99HATKernelBuilder(NDRange ndRange) {
        this.ndRange = ndRange;
    }
    public T types() {
        return this
                .charTypeDefs("byte", "boolean")
                .typedefStructOrUnion(true, "KernelContext", _ -> {

                    intDeclaration("x").semicolonNl();
                    intDeclaration("maxX").semicolonNl();
                    intDeclaration("y").semicolonNl();
                    intDeclaration("maxY").semicolon().nl();
                    intDeclaration("z").semicolonNl();
                    intDeclaration("maxZ").semicolon().nl();
                    intDeclaration("dimensions").semicolonNl();

                    // Because of order of serialization, we need to put
                    // these new members at the end.
                    intDeclaration("gix").semicolonNl();
                    intDeclaration("giy").semicolonNl();
                    intDeclaration("giz").semicolonNl();

                    intDeclaration("gsx").semicolonNl();
                    intDeclaration("gsy").semicolonNl();
                    intDeclaration("gsz").semicolonNl();

                    intDeclaration("lix").semicolonNl();
                    intDeclaration("liy").semicolonNl();
                    intDeclaration("liz").semicolonNl();

                    intDeclaration("lsx").semicolonNl();
                    intDeclaration("lsy").semicolonNl();
                    intDeclaration("lsz").semicolonNl();

                    intDeclaration("bix").semicolonNl();
                    intDeclaration("biy").semicolonNl();
                    intDeclaration("biz").semicolonNl();
                });
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


    public final T scope() {
        identifier("KernelContext_t").space().identifier("mine").semicolon().nl();
        identifier("KernelContext_t").asterisk().space().identifier("kc").equals().ampersand().identifier("mine").semicolon().nl();
        identifier("kc").rarrow().identifier("x").equals().globalId(0).semicolon().nl();
        identifier("kc").rarrow().identifier("maxX").equals().identifier("global_kc").rarrow().identifier("maxX").semicolon().nl();

        //
        identifier("kc").rarrow().identifier("gix").equals().globalId(0).semicolon().nl();
        identifier("kc").rarrow().identifier("gsx").equals().globalSize(0).semicolon().nl();
        identifier("kc").rarrow().identifier("lix").equals().localId(0).semicolon().nl();
        identifier("kc").rarrow().identifier("lsx").equals().localSize(0).semicolon().nl();
        identifier("kc").rarrow().identifier("bix").equals().blockId(0).semicolon().nl();


        if (ndRange.kid.getDimensions() > 1) { // do we need to guard this?
            identifier("kc").rarrow().identifier("y").equals().globalId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("maxY").equals().identifier("global_kc").rarrow().identifier("maxY").semicolon().nl();

            identifier("kc").rarrow().identifier("giy").equals().globalId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("gsy").equals().globalSize(1).semicolon().nl();
            identifier("kc").rarrow().identifier("liy").equals().localId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("lsy").equals().localSize(1).semicolon().nl();
            identifier("kc").rarrow().identifier("biy").equals().blockId(1).semicolon().nl();
        }

        if (ndRange.kid.getDimensions() > 2) { // do we need to guard this
            identifier("kc").rarrow().identifier("z").equals().globalId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("maxZ").equals().identifier("global_kc").rarrow().identifier("maxZ").semicolon().nl();

            identifier("kc").rarrow().identifier("giz").equals().globalId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("gsz").equals().globalSize(1).semicolon().nl();
            identifier("kc").rarrow().identifier("liz").equals().localId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("lsz").equals().localSize(2).semicolon().nl();
            identifier("kc").rarrow().identifier("biz").equals().blockId(2).semicolon().nl();
        }
        return self();
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
                    separated(paramTable.list(),(_)->comma().nl(), info -> {
                        type(buildContext, info.javaType).space().varName(info.varOp);}
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

    public T kernelEntrypoint(ScopedCodeBuilderContext buildContext,Object... args) {
        nl();
             buildContext.funcScope(buildContext.funcOp, () -> {
            kernelDeclaration(buildContext.funcOp);
            // We skip the first arg which was KernelContext.
            var list = buildContext.paramTable.list();
            for (int arg = 1; arg < args.length; arg++) {
                if (args[arg] instanceof Buffer) {
                    list.get(arg).setClass(args[arg].getClass());
                }
            }
            parenNlIndented(_ -> {
                        globalPtrPrefix().space().suffix_t("KernelContext").space().asterisk().identifier("global_kc");
                        list.stream().skip(1).forEach(info ->
                                comma().nl().type(buildContext,info.javaType).space().varName(info.varOp)
                        );
                    }
            );

            braceNlIndented(_ -> {
                scope();
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

    public abstract T globalPtrPrefix();

    public abstract T localPtrPrefix();

    public abstract T defines();

    public abstract T pragmas();

    public abstract T kernelDeclaration(CoreOp.FuncOp funcOp);

    public abstract T functionDeclaration(ScopedCodeBuilderContext codeBuilderContext, JavaType javaType, CoreOp.FuncOp funcOp);

    public abstract T globalId(int id);

    public abstract T localId(int id);

    public abstract T globalSize(int id);

    public abstract T localSize(int id);

    public abstract T blockId(int id);

}
