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
package optkl.codebuilders;

import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpHelper;

import java.lang.invoke.MethodHandles;

public class JavaCodeBuilder<T extends JavaCodeBuilder<T>> extends ScopeAwareJavaOrC99StyleCodeBuilder<T> {
    @Override
    public T type( JavaType javaType) {
        // lets do equiv of SimpleName
        String longName = javaType.toString();
        int lastIdx = Math.max(longName.lastIndexOf('$'),longName.lastIndexOf('.'));
        String shortName  = lastIdx>0?longName.substring(lastIdx+1):longName;
        return typeName(shortName);
    }

    public T createJava(ScopedCodeBuilderContext buildContext) {
        buildContext.funcScope(buildContext.funcOp(), () -> {
            typeName(buildContext.funcOp().resultType().toString()).space().funcName(buildContext.funcOp());
            parenNlIndented(_ ->
                    commaNlSeparated(
                            buildContext.paramTable.list(),
                            param -> declareParam( param)
                    )
            );
            braceNlIndented(_ -> nlSeparated(
                    OpHelper.Statement.statements(buildContext.funcOp().bodies().getFirst().entryBlock()),
                    statement -> statement( statement)
                    )
            );
        });
        return nl();
    }
    MethodHandles.Lookup lookup;
    CoreOp.FuncOp funcOp;
    public JavaCodeBuilder(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp){
        super(new ScopedCodeBuilderContext(lookup,funcOp));
        this.lookup=lookup;
        this.funcOp = funcOp;
    }

    public String toText() {
        ScopedCodeBuilderContext scopedCodeBuilderContext= new ScopedCodeBuilderContext(lookup,funcOp);
        return createJava(scopedCodeBuilderContext).getText();
    }

    public static JavaCodeBuilder of(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        return new JavaCodeBuilder(lookup,funcOp);
    }

    public static String toText(MethodHandles.Lookup lookup, CoreOp.FuncOp funcOp) {
        return of(lookup,funcOp).toText();
    }
}
