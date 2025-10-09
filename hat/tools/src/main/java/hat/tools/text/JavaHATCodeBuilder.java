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
package hat.tools.text;

import hat.codebuilders.ScopedCodeBuilderContext;
import hat.codebuilders.HATCodeBuilderWithContext;
import hat.dialect.HatBlockThreadIdOp;
import hat.dialect.HatVSelectLoadOp;
import hat.dialect.HatVSelectStoreOp;
import hat.dialect.HatVectorBinaryOp;
import hat.dialect.HatVectorLoadOp;
import hat.dialect.HatVectorStoreView;
import hat.dialect.HatGlobalThreadIdOp;
import hat.dialect.HatGlobalSizeOp;
import hat.dialect.HatLocalSizeOp;
import hat.dialect.HatLocalThreadIdOp;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

public class JavaHATCodeBuilder<T extends JavaHATCodeBuilder<T>> extends HATCodeBuilderWithContext<T> {

    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
            try {
                typeName(javaType.resolve(buildContext.lookup).getTypeName());
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        return self();
    }


    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (OpTk.isKernelContextAccess(fieldLoadOp)) {
            identifier("kc").dot().fieldName(fieldLoadOp);
        } else if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) { // only primitve fields
            var value = OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("An instance field? I guess - we dont get those in HAT " +fieldLoadOp);
        }
        return self();
    }

    @Override
    public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        if (!invokeOp.operands().isEmpty() && invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
            recurse(buildContext, instanceResult.op());
        }
        dot().identifier(invokeOp.invokeDescriptor().name());
        paren(_ ->
                // why the sublist? is this static vs instance?
            separated(  invokeOp.operands().subList(0,invokeOp.operands().size()-1), (_)->commaSpace(),o->
                    recurse(buildContext,  ((Op.Result) o).op())
            )
        );
        return self();
    }

    @Override
    public T privateDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        blockComment("/* private declaration !! */");
        return self();
    }

    @Override
    public T localDeclaration(LocalArrayDeclaration localArrayDeclaration) {
        blockComment("/* local declaration !! */");
        return self();
    }

    @Override
    public T syncBlockThreads() {
        blockComment("/* group wide barrier!! */");
        return self();
    }

    @Override
    public T generateVectorStore(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        blockComment("Store Vector Not Implemented");
        return self();
    }

    @Override
    public T generateVectorBinary(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        blockComment("Binary Vector Not Implemented");
        return self();
    }

    @Override
    public T generateVectorLoad(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        blockComment("Load Vector Not Implemented");
        return self();
    }

    @Override
    public T generateVectorSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }

    @Override
    public T generateVectorSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }

    @Override
    public T hatGlobalThreadOp(ScopedCodeBuilderContext buildContext, HatGlobalThreadIdOp globalThreadIdOp) {
        blockInlineComment("Thread ID access");
        return self();
    }

    @Override
    public T hatGlobalSizeOp(ScopedCodeBuilderContext buildContext, HatGlobalSizeOp hatGlobalThreadIdOp) {
        blockInlineComment("GlobalSize");
        return self();
    }

    @Override
    public T hatLocalThreadIdOp(ScopedCodeBuilderContext buildContext, HatLocalThreadIdOp hatLocalThreadIdOp) {
        blockInlineComment("Local Thread ID");
        return self();
    }

    @Override
    public T hatLocalSizeOp(ScopedCodeBuilderContext buildContext, HatLocalSizeOp hatLocalSizeOp) {
        blockInlineComment("Local Size");
        return self();
    }

    @Override
    public T hatBlockThreadIdOp(ScopedCodeBuilderContext buildContext, HatBlockThreadIdOp hatBlockThreadIdOp) {
        blockInlineComment("Block ID ");
        return self();
    }

    @Override
    public T hatVectorStoreOp(ScopedCodeBuilderContext buildContext, HatVectorStoreView hatVectorStoreView) {
        blockComment("Store Vector Not Implemented");
        return self();
    }

    @Override
    public T hatBinaryVectorOp(ScopedCodeBuilderContext buildContext, HatVectorBinaryOp hatVectorBinaryOp) {
        blockComment("Binary Vector Not Implemented");
        return self();
    }

    @Override
    public T hatVectorLoadOp(ScopedCodeBuilderContext buildContext, HatVectorLoadOp hatVectorLoadOp) {
        blockComment("Load Vector Not Implemented");
        return self();
    }

    @Override
    public T hatSelectLoadOp(ScopedCodeBuilderContext buildContext, HatVSelectLoadOp hatVSelectLoadOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }

    @Override
    public T hatSelectStoreOp(ScopedCodeBuilderContext buildContext, HatVSelectStoreOp hatVSelectStoreOp) {
        blockComment("Select Vector Not Implemented");
        return self();
    }

    public T createJava(ScopedCodeBuilderContext buildContext) {
        buildContext.funcScope(buildContext.funcOp, () -> {
            typeName(buildContext.funcOp.resultType().toString()).space().funcName(buildContext.funcOp);
            parenNlIndented(_ -> separated(buildContext.paramTable.list(), (_) -> comma().nl(),
                    param -> declareParam(buildContext, param)));
            braceNlIndented(_ -> separated(OpTk.statements(buildContext.funcOp.bodies().getFirst().entryBlock()), (_) -> nl(),
                    statement -> statement(buildContext, statement))
            );
        });
        return self();
    }

}
