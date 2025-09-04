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

import hat.codebuilders.HATCodeBuilderContext;
import hat.codebuilders.HATCodeBuilderWithContext;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.lang.invoke.MethodHandles;

public  class JavaHATCodeBuilder<T extends JavaHATCodeBuilder<T>> extends HATCodeBuilderWithContext<T> {
    @Override
    public T type(HATCodeBuilderContext buildContext, JavaType javaType) {
            try {
                typeName(javaType.resolve(buildContext.lookup).getTypeName());
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }
        return self();
    }

    @Override
    public T fieldLoad(HATCodeBuilderContext buildContext, JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
        if (OpTk.isKernelContextAccess(fieldLoadOp)) {
            identifier("kc").dot().identifier(OpTk.fieldName(fieldLoadOp));
        } else if (fieldLoadOp.operands().isEmpty() && fieldLoadOp.result().type() instanceof PrimitiveType) { // only primitve fields
            var value = OpTk.getStaticFinalPrimitiveValue(buildContext.lookup,fieldLoadOp);
            literal(value.toString());
        } else {
            throw new IllegalStateException("An instance field? I guess - we dont get those in HAT " +fieldLoadOp);
        }
        return self();
    }

    @Override
    public T methodCall(HATCodeBuilderContext buildContext, JavaOp.InvokeOp invokeOp) {
        if (!invokeOp.operands().isEmpty() && invokeOp.operands().getFirst() instanceof Op.Result instanceResult) {
            recurse(buildContext, instanceResult.op());
        }
        dot().identifier(invokeOp.invokeDescriptor().name());
        paren(_ ->
            commaSeparated(  invokeOp.operands().subList(0,invokeOp.operands().size()-1), o->
                    recurse(buildContext,  ((Op.Result) o).op())
            )
        );
        return self();
    }

    @Override
    public T emitPrivateDeclaration(String typeName, String varName) {
        // TODO: What would emit a Java backend
        return self();
    }

    @Override
    public T emitLocalDeclaration(String typeName, String varName) {
        // TODO: What would emit a Java backend
        return self();
    }

    @Override
    public T syncBlockThreads() {
        // TODO: What would emit a Java backend?
        return self();
    }

    public T compute(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        HATCodeBuilderContext buildContext = new HATCodeBuilderContext(lookup,funcOp);
        typeName(funcOp.resultType().toString()).space().identifier(funcOp.funcName());
        parenNlIndented(_ ->
                commaSeparated(buildContext.paramTable.list(), (info) -> type(buildContext,(JavaType) info.parameter.type()).space().varName(info.varOp))
        );
        braceNlIndented(_ ->
                OpTk.rootOpStream(funcOp)
                        .forEach(root ->
                                recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root)).nl()
                        )
        );
        return self();
    }

}
