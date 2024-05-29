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


import hat.optools.FuncOpWrapper;
import hat.optools.StructuralOpWrapper;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.JavaType;


public abstract class C99HatComputeBuilder<T extends C99HatComputeBuilder<T>> extends C99HatBuilder<T> {

    public T computeDeclaration(TypeElement typeElement, String name) {
        return typeName(typeElement.toString()).space().identifier(name);
    }

    public T compute(FuncOpWrapper funcOpWrapper) {
        computeDeclaration(funcOpWrapper.functionReturnTypeDesc(), funcOpWrapper.functionName());
        parenNlIndented(_ ->
                commaSeparated(funcOpWrapper.paramTable.list(), (info) -> type((JavaType) info.parameter.type()).space().varName(info.varOp))
        );
        C99HatBuildContext buildContext = new C99HatBuildContext(funcOpWrapper);
        braceNlIndented(_ ->
                funcOpWrapper.wrappedRootOpStream(funcOpWrapper.firstBlockOfFirstBody()).forEach(root ->
                        recurse(buildContext, root).semicolonIf(!(root instanceof StructuralOpWrapper<?>)).nl()
                ));

        return self();
    }
}
