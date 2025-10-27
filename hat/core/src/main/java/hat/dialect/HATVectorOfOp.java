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

public class HATVectorOfOp extends HATVectorOp {

    private final TypeElement typeElement;
    private final int loadN;

    public HATVectorOfOp(TypeElement typeElement, int loadN, List<Value> operands) {
        super("", typeElement, loadN, operands);
        this.typeElement = typeElement;
        this.loadN = loadN;
    }

    public HATVectorOfOp(HATVectorOfOp op, CopyContext copyContext) {
        super(op, copyContext);
        this.typeElement = op.typeElement;
        this.loadN = op.loadN;
    }

    @Override
    public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
        return new HATVectorOfOp(this, copyContext);
    }

    @Override
    public TypeElement resultType() {
        return typeElement;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.vectorOf." + varName(), typeElement);
    }


}
