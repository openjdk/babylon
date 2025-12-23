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

import optkl.OpTkl;
import optkl.codebuilders.BabylonCoreOpBuilder;
import hat.codebuilders.C99HATCodeBuilderContext;
import optkl.codebuilders.ScopedCodeBuilderContext;

import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import static optkl.OpTkl.getStaticFinalPrimitiveValue;

public class JavaHATCodeBuilder<T extends JavaHATCodeBuilder<T>> extends C99HATCodeBuilderContext<T> implements BabylonCoreOpBuilder<T,ScopedCodeBuilderContext> {


    @Override
    public T type(ScopedCodeBuilderContext buildContext, JavaType javaType) {
        return typeName(javaType.toString());
    }

    @Override
    public T fieldLoadOp(ScopedCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (OpTk.isKernelContextFieldAccessOp(buildContext.lookup,fieldLoadOp,_->true)) {
            identifier("kc").dot().fieldName(fieldLoadOp);
        } else if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) { // only primitve fields
            var value = getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("An instance field? I guess - we dont get those in HAT " +fieldLoadOp);
        }
        return self();
    }

    @Override
    public T atomicInc(ScopedCodeBuilderContext buildContext, Op.Result instanceResult, String name) {
        return null;
    }

    @Override
    public T invokeOp(ScopedCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        if (!invokeOp.operands().isEmpty() && invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
            recurse(buildContext, instanceResult.op());
        }
        dot().identifier(invokeOp.invokeDescriptor().name());
        paren(_ ->
                // why the sublist? is this static vs instance?
            commaSpaceSeparated(  invokeOp.operands().subList(0,invokeOp.operands().size()-1), o->
                    recurse(buildContext,  ((Op.Result) o).op())
            )
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
                    OpTkl.statements(buildContext.funcOp.bodies().getFirst().entryBlock()),
                    statement -> statement(buildContext, statement)
                    )
            );
        });
        return nl();
    }
}
