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

import jdk.incubator.code.CopyContext;
import jdk.incubator.code.Op;
import jdk.incubator.code.OpTransformer;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;

import java.util.List;
import java.util.Map;

public final class HATVectorStoreView extends HATVectorViewOp {

    private final TypeElement elementType;
    private final int storeN;
    private final boolean isSharedOrPrivate;
    private final VectorType vectorType;

    public HATVectorStoreView(String varName, TypeElement elementType, int storeN, VectorType vectorType, boolean isSharedOrPrivate, List<Value> operands) {
        super(varName, operands);
        this.elementType = elementType;
        this.storeN = storeN;
        this.isSharedOrPrivate = isSharedOrPrivate;
        this.vectorType = vectorType;
    }

    public HATVectorStoreView(HATVectorStoreView op, CopyContext copyContext) {
        super(op, copyContext);
        this.elementType = op.elementType;
        this.storeN = op.storeN;
        this.isSharedOrPrivate = op.isSharedOrPrivate;
        this.vectorType = op.vectorType;
    }

    @Override
    public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
        return new HATVectorStoreView(this, copyContext);
    }

    @Override
    public TypeElement resultType() {
        return elementType;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.floatNStoreView." + varName(), elementType);
    }

    public int storeN() {
        return storeN;
    }

    public boolean isSharedOrPrivate() {
        return this.isSharedOrPrivate;
    }

    public String buildType() {
        return vectorType.type();
    }
}
