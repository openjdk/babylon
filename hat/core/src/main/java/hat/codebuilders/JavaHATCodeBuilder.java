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
import jdk.incubator.code.dialect.core.CoreOp;
import optkl.OpHelper;
import optkl.codebuilders.BabylonCoreOpBuilder;
import optkl.codebuilders.ScopedCodeBuilderContext;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.invoke.MethodHandles;

import static optkl.OpHelper.NamedOpHelper.FieldAccess.fieldAccessOpHelper;
import static optkl.OpHelper.NamedOpHelper.Invoke.invokeOpHelper;

public class JavaHATCodeBuilder<T extends JavaHATCodeBuilder<T>> extends C99HATCodeBuilderContext<T> implements BabylonCoreOpBuilder<T,ScopedCodeBuilderContext> {


    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        return typeName(javaType.toString());
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp1) {
        var fieldAccess = fieldAccessOpHelper(buildContext.lookup,fieldLoadOp1);
        if ( fieldAccess.refType(KernelContext.class)) {
            identifier("kc").dot().fieldName(fieldAccess.op());
        } else if (fieldAccess.op().operands().isEmpty() && fieldAccess.op().result().type() instanceof PrimitiveType primitiveType) { // only primitve fields
            var value = fieldAccess.getStaticFinalPrimitiveValue();
            literal(primitiveType,value.toString());
        } else {
            throw new IllegalStateException("An instance field? I guess - we dont get those in HAT " + fieldAccess.op());
        }
        return self();
    }

    @Override
    public T atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        return null;
    }

    @Override
     public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        var invoke = invokeOpHelper(buildContext.lookup,invokeOp);
        identifier(invoke.refType().toString()).dot().identifier(invoke.name());
        paren(_ ->
            commaSpaceSeparated(invoke.op().operands(), o-> {
                if (o instanceof Op.Result result) {
                    recurse(buildContext, result.op());
                } else if (o instanceof jdk.incubator.code.Block.Parameter parameter) {
                    identifier("param$"+parameter.index());
                }else {
                    throw new IllegalStateException("What have we here ");
                }
            })
        );
        return self();
    }

    public T createJava(ScopedCodeBuilderContext buildContext) {
        buildContext.funcScope(buildContext.funcOp, () -> {
            typeName(buildContext.funcOp.resultType().toString()).space().funcName(buildContext.funcOp);
            parenNlIndented(_ ->
                    commaNlSeparated(
                            buildContext.paramTable.list(),
                            param -> declareParam(buildContext, param)
                    )
            );
            braceNlIndented(_ -> nlSeparated(
                    OpHelper.Statement.statements(buildContext.funcOp.bodies().getFirst().entryBlock()),
                    statement -> statement(buildContext, statement)
                    )
            );
        });
        return nl();
    }
    private final ScopedCodeBuilderContext scopedCodeBuilderContext;
    public JavaHATCodeBuilder(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp){
        super();
        scopedCodeBuilderContext= new ScopedCodeBuilderContext(lookup,funcOp);
    }

    public String toText() {
        return createJava(scopedCodeBuilderContext).getText();
    }
}
