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


import hat.KernelContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.callgraph.KernelCallGraph;
import hat.callgraph.KernelEntrypoint;
import hat.optools.FuncOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.StructuralOpWrapper;
import hat.util.StreamCounter;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaType;

import javax.annotation.processing.SupportedSourceVersion;
import java.lang.foreign.GroupLayout;

import java.util.function.Consumer;

public abstract class C99HATKernelBuilder<T extends C99HATKernelBuilder<T>> extends HATCodeBuilderWithContext<T> {

    protected final NDRange ndRange;

    public C99HATKernelBuilder(NDRange ndRange) {
        this.ndRange = ndRange;
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

                    // The context is customized depending on the NDRange of the application:
                    // 1D, 2D or 3D.
                    // An alternative is to always generate the 3D range for OpenCL.

                    // Kernels are, at least, 1D
                    intDeclaration("x").semicolonNl();
                    intDeclaration("maxX").semicolonNl();

                    if (ndRange.kid.getDimensions() > 1) {
                        // The code builder needs the NDRange
                        intDeclaration("y").semicolonNl();
                        intDeclaration("maxY").semicolon().nl();
                    }

                    if (ndRange.kid.getDimensions() > 2) {
                        // The code builder needs the NDRange
                        intDeclaration("z").semicolonNl();
                        intDeclaration("maxZ").semicolon().nl();
                    }

                    // Because of order of serialization, we need to put
                    // these new members at the end.
                    intDeclaration("gx").semicolonNl();
                    intDeclaration("lx").semicolonNl();
                    intDeclaration("lsx").semicolonNl();
                    intDeclaration("bsx").semicolonNl();

                    if (ndRange.kid.getDimensions() > 1) {
                        intDeclaration("gy").semicolon().nl();
                        intDeclaration("ly").semicolonNl();
                        intDeclaration("lsy").semicolonNl();
                        intDeclaration("bsy").semicolonNl();
                    }

                    if (ndRange.kid.getDimensions() > 2) {
                        intDeclaration("gz").semicolon().nl();
                        intDeclaration("lz").semicolonNl();
                        intDeclaration("lsz").semicolonNl();
                        intDeclaration("bsz").semicolonNl();
                    }
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
        identifier("kc").rarrow().identifier("gx").equals().globalId(0).semicolon().nl();
        identifier("kc").rarrow().identifier("lx").equals().localId(0).semicolon().nl();
        identifier("kc").rarrow().identifier("lsx").equals().localSize(0).semicolon().nl();
        identifier("kc").rarrow().identifier("bsx").equals().blockSize(0).semicolon().nl();

        if (ndRange.kid.getDimensions() > 1) {
            identifier("kc").rarrow().identifier("y").equals().globalId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("maxY").equals().identifier("global_kc").rarrow().identifier("maxY").semicolon().nl();

            identifier("kc").rarrow().identifier("gy").equals().globalId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("ly").equals().localId(1).semicolon().nl();
            identifier("kc").rarrow().identifier("lsy").equals().localSize(1).semicolon().nl();
            identifier("kc").rarrow().identifier("bsy").equals().blockSize(1).semicolon().nl();
        }
        if (ndRange.kid.getDimensions() > 2) {
            identifier("kc").rarrow().identifier("z").equals().globalId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("maxZ").equals().identifier("global_kc").rarrow().identifier("maxZ").semicolon().nl();

            identifier("kc").rarrow().identifier("gz").equals().globalId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("lz").equals().localId(2).semicolon().nl();
            identifier("kc").rarrow().identifier("lsz").equals().localSize(2).semicolon().nl();
            identifier("kc").rarrow().identifier("bsz").equals().blockSize(2).semicolon().nl();
        }
        return self();
    }

    public abstract T globalPtrPrefix();

    @Override
    public T type(CodeBuilderContext buildContext, JavaType javaType) {
        if (InvokeOpWrapper.isIfaceUsingLookup(buildContext.lookup(),javaType) && javaType instanceof ClassType classType) {
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
            // In the case we call a new invoke method and pass the kernel context around, t
            // then we need to do the mapping between the Java type and its low level interface
            // TODO: Check if there is a better way to obtain the type information using
            // the code reflection APIs and avoid string comparisons.
            String kernelContextFullClassName = KernelContext.class.getCanonicalName();
            if (javaType.toString().equals(kernelContextFullClassName)) {
                typeName("KernelContext_t *");
            } else {
                typeName(javaType.toString());
            }
        }

        return self();
    }

    public T kernelMethod(KernelCallGraph.KernelReachableResolvedMethodCall kernelReachableResolvedMethodCall) {
        CodeBuilderContext buildContext = new CodeBuilderContext(kernelReachableResolvedMethodCall.funcOpWrapper());
        buildContext.scope(buildContext.funcOpWrapper, () -> {
            nl();
            functionDeclaration(buildContext,buildContext.funcOpWrapper.getReturnType(), buildContext.funcOpWrapper.functionName());

            var list = buildContext.funcOpWrapper.paramTable.list();
            parenNlIndented(_ ->
                    commaSeparated(list, (info) -> type(buildContext,info.javaType).space().varName(info.varOp))
            );

            braceNlIndented(_ -> {
                StreamCounter.of(buildContext.funcOpWrapper.wrappedRootOpStream(), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                );
            });
        });
        return self();
    }

    public T kernelMethod(FuncOpWrapper funcOpWrapper) {

        CodeBuilderContext buildContext = new CodeBuilderContext(funcOpWrapper);
        buildContext.scope(buildContext.funcOpWrapper, () -> {
            nl();
            functionDeclaration(buildContext,buildContext.funcOpWrapper.getReturnType(), buildContext.funcOpWrapper.functionName());

            var list = buildContext.funcOpWrapper.paramTable.list();
            parenNlIndented(_ ->
                    commaSeparated(list, (info) -> type(buildContext,info.javaType).space().varName(info.varOp))
            );

            braceNlIndented(_ -> {
                StreamCounter.of(buildContext.funcOpWrapper.wrappedRootOpStream(), (c, root) ->
                        nlIf(c.isNotFirst()).recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>))
                );
            });
        });
        return self();
    }

    public T kernelEntrypoint(KernelEntrypoint kernelEntrypoint,Object... args) {

        nl();
        CodeBuilderContext buildContext = new CodeBuilderContext(kernelEntrypoint.funcOpWrapper());
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
                        globalPtrPrefix().space().suffix_t("KernelContext").space().asterisk().identifier("global_kc");
                        list.stream().skip(1).forEach(info ->
                                comma().space().type(buildContext,info.javaType).space().varName(info.varOp)
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

    public abstract T functionDeclaration(CodeBuilderContext codeBuilderContext,JavaType javaType, String name);

    public abstract T globalId(int id);

    public abstract T localId(int id);

    public abstract T globalSize(int id);

    public abstract T localSize(int id);

    public abstract T blockSize(int id);

}
