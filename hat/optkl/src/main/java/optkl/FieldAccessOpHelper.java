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
package optkl;

import jdk.incubator.code.CodeElement;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.util.Regex;
import optkl.util.carriers.LookupCarrier;

import java.lang.invoke.MethodHandles;

public record FieldAccessOpHelper(MethodHandles.Lookup lookup, JavaOp.FieldAccessOp op) implements LookupCarrier,OpHelper<JavaOp.FieldAccessOp> {
    @Override
    public boolean isStatic(){
        return true;//fieldAccessOp.invokeKind().equals(JavaOp.FieldAccessOp.InvokeKind.STATIC);
    }
    @Override
    public boolean isInstance(){
        return true;//fieldAccessOp.invokeKind().equals(JavaOp.FieldAccessOp.InvokeKind.INSTANCE);
    }
    @Override
    public String name(){
        return op.fieldDescriptor().name();
    }

    public <T>boolean of(Class<T> clazz){
        return isAssignable(clazz,(JavaType) op.resultType());
    }

    public static FieldAccessOpHelper fieldAccessOpHelper(MethodHandles.Lookup lookup, CodeElement<?,?> codeElement){
        return codeElement instanceof JavaOp.FieldAccessOp fieldAccessOp? new FieldAccessOpHelper(lookup,fieldAccessOp): null;

    }
}
