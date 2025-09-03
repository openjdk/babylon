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

import hat.optools.OpTk;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.invoke.MethodHandles;


public  class C99HATComputeBuilder<T extends C99HATComputeBuilder<T>> extends HATCodeBuilderWithContext<T> {

    public T computeDeclaration(TypeElement typeElement, String name) {
        return typeName(typeElement.toString()).space().identifier(name);
    }

    public T compute(MethodHandles.Lookup lookup,CoreOp.FuncOp funcOp) {
        HATCodeBuilderContext buildContext = new HATCodeBuilderContext(lookup,funcOp);
        computeDeclaration(funcOp.resultType(), funcOp.funcName());
        parenNlIndented(_ ->
                commaSeparated(buildContext.paramTable.list(), (info) -> type(buildContext,(JavaType) info.parameter.type()).space().varName(info.varOp))
        );

        braceNlIndented(_ ->
                OpTk.rootOpStream(buildContext.lookup,funcOp)
                        .forEach(root ->
                                recurse(buildContext, root).semicolonIf(!OpTk.isStructural(root)).nl()
                        )
        );

        return self();
    }

    @Override
    public T emitCastToLocal(String typeName, String varName,  String localVarS, boolean isAPISimplified) {
        // TODO: What would emit a pure C99 backend?
        return self();
    }

    @Override
    public T emitlocalArrayWithSize(String localVarS, int size, JavaType type) {
        // TODO: What would emit a pure C99 backend?
        return self();
    }

    @Override
    public T syncBlockThreads() {
        // TODO: What would emit a pure C99 backend?
        return self();
    }
}
