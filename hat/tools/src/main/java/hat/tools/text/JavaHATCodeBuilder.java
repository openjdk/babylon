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
import hat.optools.FieldLoadOpWrapper;
import hat.optools.FuncOpWrapper;
import hat.optools.InvokeOpWrapper;
import hat.optools.OpWrapper;
import hat.optools.StructuralOpWrapper;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.invoke.MethodHandle;
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
    public T fieldLoad(HATCodeBuilderContext buildContext, FieldLoadOpWrapper fieldLoadOpWrapper) {
        if (fieldLoadOpWrapper.isKernelContextAccess()) {
            identifier("kc").dot().identifier(fieldLoadOpWrapper.fieldName());
        } else if (fieldLoadOpWrapper.isStaticFinalPrimitive()) {
            literal(fieldLoadOpWrapper.getStaticFinalPrimitiveValue().toString());
        } else {
            throw new IllegalStateException("An instance field? I guess - we dont get those in HAT " + fieldLoadOpWrapper.fieldRef());
        }
        return self();
    }
    @Override
    public T methodCall(HATCodeBuilderContext buildContext, InvokeOpWrapper invokeOpWrapper) {
        if (!invokeOpWrapper.op.operands().isEmpty() && invokeOpWrapper.op.operands().getFirst() instanceof Op.Result instanceResult) {
            recurse(buildContext, OpWrapper.wrap(buildContext.lookup, instanceResult.op()));
        }
        dot().identifier(invokeOpWrapper.name());
        paren(_ ->
            commaSeparated(  invokeOpWrapper.op.operands().subList(0,invokeOpWrapper.op.operands().size()-1), o->
                    recurse(buildContext, OpWrapper.wrap(buildContext.lookup, ((Op.Result) o).op()))
            )
        );
        return self();
    }

    public T compute(MethodHandles.Lookup lookup,FuncOpWrapper funcOpWrapper) {
        HATCodeBuilderContext buildContext = new HATCodeBuilderContext(lookup,funcOpWrapper);
        typeName(funcOpWrapper.functionReturnTypeDesc().toString()).space().identifier(funcOpWrapper.functionName());
        parenNlIndented(_ ->
                commaSeparated(funcOpWrapper.paramTable.list(), (info) -> type(buildContext,(JavaType) info.parameter.type()).space().varName(info.varOp))
        );
        braceNlIndented(_ ->
                funcOpWrapper.wrappedRootOpStream(funcOpWrapper.op.bodies().getFirst().entryBlock()).forEach(root ->
                        recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>)).nl()
                ));
        return self();
    }

   /* public T compute(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        HATCodeBuilderContext buildContext = new HATCodeBuilderContext(funcOpWrapper);
        typeName(funcOpWrapper.functionReturnTypeDesc().toString()).space().identifier(funcOpWrapper.functionName());
        parenNlIndented(_ ->
                commaSeparated(funcOpWrapper.paramTable.list(), (info) -> type(buildContext,(JavaType) info.parameter.type()).space().varName(info.varOp))
        );
        braceNlIndented(_ ->
                funcOpWrapper.wrappedRootOpStream(funcOpWrapper.firstBlockOfFirstBody()).forEach(root ->
                        recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>)).nl()
                ));
        return self();
    }*/
}
