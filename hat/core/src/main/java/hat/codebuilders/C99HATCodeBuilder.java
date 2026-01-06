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
import hat.dialect.HATF16Op;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import optkl.codebuilders.C99CodeBuilder;

public  class C99HATCodeBuilder<T extends C99HATCodeBuilder<T>> extends C99CodeBuilder<T> {

    public final T varName(HATMemoryVarOp hatLocalVarOp) {
        identifier(hatLocalVarOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorVarOp hatVectorVarOp) {
        identifier(hatVectorVarOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorLoadOp vectorLoadOp) {
        identifier(vectorLoadOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorStoreView hatVectorStoreView) {
        identifier(hatVectorStoreView.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorBinaryOp hatVectorBinaryOp) {
        identifier(hatVectorBinaryOp.varName());
        return self();
    }

    public final T varName(HATVectorOp.HATVectorVarLoadOp hatVectorVarLoadOp) {
        identifier(hatVectorVarLoadOp.varName());
        return self();
    }

    public final T varName(HATF16Op.HATF16VarOp hatF16VarOp) {
        identifier(hatF16VarOp.varName());
        return self();
    }
}
