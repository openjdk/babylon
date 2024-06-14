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
package hat.backend.c99codebuilders;


import hat.buffer.Buffer;
import hat.buffer.KernelContext;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import hat.optools.FuncOpWrapper;
import hat.optools.StructuralOpWrapper;
import hat.util.StreamCounter;

import java.lang.foreign.GroupLayout;
import java.lang.reflect.code.type.ClassType;
import java.lang.reflect.code.type.JavaType;
import java.util.function.Consumer;

public abstract class C99HatKernelBuilder<T extends C99HatKernelBuilder<T>> extends C99HatBuilder<T> {
    public C99HatKernelBuilder() {

    }

    public T types() {
        return this
                .charTypeDefs("s8_t", "byte", "boolean")
                .unsignedCharTypeDefs("u8_t")
                .shortTypeDefs("s16_t")
                .unsignedShortTypeDefs("u16_t")
                .unsignedIntTypeDefs("u32_t")
                .intTypeDefs("s32_t")
                .floatTypeDefs("f32_t")
                .longTypeDefs("s64_t")
                .unsignedLongTypeDefs("u64_t")
                .typedefStructOrUnion(true, "KernelContext", _ -> {
                    intDeclaration("x").semicolonNl();
                    intDeclaration("maxX").semicolon();
                });

    }

    T typedefStructOrUnion(boolean isStruct, String name, Consumer<T> consumer) {
        return
                typedefKeyword().space().structOrUnion(isStruct).space()
                        .either(isStruct, _ -> suffix_s(name), _ -> suffix_u(name)).braceNlIndented(consumer)
                        .suffix_t(name).semicolon().nl();

    }


    public final T scope() {
        return
                identifier("kc").rarrow().identifier("x").equals().globalId().semicolon().nl();
                //.identifier("kc").rarrow().identifier("maxX").equals().globalSize().semicolon().nl();
    }

    public abstract T globalPtrPrefix();

    @Override
    public T type(JavaType javaType) {
        if (FuncOpWrapper.ParamTable.Info.isIfaceBuffer(javaType) && javaType instanceof ClassType classType) {
            globalPtrPrefix().space();
            String name = classType.toClassName();
            int dotIdx = name.lastIndexOf('.');
            int dollarIdx = name.lastIndexOf('$');
            int idx = Math.max(dotIdx, dollarIdx);
            if (idx > 0) {
                name = name.substring(idx + 1);
            }
            suffix_t(name).asterisk();
        } else {
            typeName(javaType.toBasicType().toString());
        }

        return self();
    }

    public T kernelMethod(KernelCallGraph.KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {
        C99HatBuildContext buildContext = new C99HatBuildContext(kernelReachableResolvedMethodCall.funcOpWrapper());
        buildContext.scope(buildContext.funcOpWrapper, () -> {
            nl();
            functionDeclaration(buildContext.funcOpWrapper.getReturnType(), buildContext.funcOpWrapper.functionName());

            var list = buildContext.funcOpWrapper.paramTable.list();
            parenNlIndented(_ ->
                    commaSeparated(list, (info) -> type(info.javaType).space().varName(info.varOp))
            );

            braceNlIndented(_ -> {
                //scope();
                StreamCounter.of(buildContext.funcOpWrapper.wrappedRootOpStream(), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                );
            });
        });
        return self();
    }

    public T kernelEntrypoint(KernelEntrypoint kernelEntrypoint, Object[] args) {

        nl();
        C99HatBuildContext buildContext = new C99HatBuildContext(kernelEntrypoint.funcOpWrapper());
        //  System.out.print(kernelReachableResolvedMethodCall.funcOpWrapper().toText());
        buildContext.scope(buildContext.funcOpWrapper, () -> {

            kernelDeclaration(buildContext.funcOpWrapper.functionName());
            // We skip the first arg which was KernelContext.
            var list = buildContext.funcOpWrapper.paramTable.list();
            for (int arg = 1; arg < args.length; arg++) {
                if (args[arg] instanceof Buffer buffer) {
                    FuncOpWrapper.ParamTable.Info info = list.get(arg);
                    info.setLayout((GroupLayout) Buffer.getLayout(buffer));
                    info.setClass(args[arg].getClass());
                }
            }
            parenNlIndented(_ -> {
                        globalPtrPrefix().space().suffix_t("KernelContext").space().asterisk().identifier("kc");
                        list.stream().skip(1).forEach(info ->
                                comma().space().type(info.javaType).space().varName(info.varOp)
                        );
                    }
            );

            braceNlIndented(_ -> {
                scope();
                StreamCounter.of(buildContext.funcOpWrapper.wrappedRootOpStream(), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                );
            });
        });
        return self();
    }


    public abstract T defines();

    public abstract T pragmas();

    public abstract T kernelDeclaration(String name);

    public abstract T functionDeclaration(JavaType javaType, String name);

    public abstract T globalId();

    public abstract T globalSize();

}
