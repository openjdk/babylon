/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.dialect;

import jdk.incubator.code.CodeContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;

import java.util.List;
import java.util.Map;

public final class HATVectorStoreView extends HATVectorOp {

    private final boolean isSharedOrPrivate;

    public HATVectorStoreView(String varName, TypeElement resultType, int storeN, TypeElement vectorElementType, boolean isSharedOrPrivate, List<Value> operands) {
        super(varName, resultType, vectorElementType, storeN, operands);
        this.isSharedOrPrivate = isSharedOrPrivate;
    }

    public HATVectorStoreView(HATVectorStoreView op, CodeContext copyContext) {
        super(op, copyContext);
        this.isSharedOrPrivate = op.isSharedOrPrivate;
    }

    @Override
    public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
        return new HATVectorStoreView(this, copyContext);
    }

   // @Override
   // public TypeElement resultType() {
     //   return super.typeElement;
   // }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect." + vectorElementType().toString() + vectorN() + "StoreView." + varName(), resultType());
    }

    public boolean isSharedOrPrivate() {
        return this.isSharedOrPrivate;
    }

}
