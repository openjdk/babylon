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
import jdk.incubator.code.dialect.core.CoreOp;

import java.util.List;
import java.util.Map;

public class HATVSelectStoreOp extends HATVectorViewOp {

    private final TypeElement elementType;
    private final int lane;
    private final CoreOp.VarOp resultVarOp;

    public HATVSelectStoreOp(String varName, TypeElement typeElement, int lane, CoreOp.VarOp resultVarOp, List<Value> operands) {
        super(varName, operands);
        this.elementType = typeElement;
        this.lane = lane;
        this.resultVarOp = resultVarOp;
    }

    public HATVSelectStoreOp(HATVSelectStoreOp that, CopyContext cc) {
        super(that, cc);
        this.elementType = that.elementType;
        this.lane = that.lane;
        this.resultVarOp = that.resultVarOp;
    }

    @Override
    public Op transform(CopyContext copyContext, OpTransformer opTransformer) {
        return new HATVSelectStoreOp(this, copyContext);
    }

    @Override
    public TypeElement resultType() {
        return elementType;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.vselect.store." + lane, elementType);
    }

    public String mapLane() {
        return switch (lane) {
            case 0 -> "x";
            case 1 -> "y";
            case 2 -> "z";
            case 3 -> "w";
            default -> throw new InternalError("Invalid lane: " + lane);
        };
    }

    public CoreOp.VarOp resultValue() {
        return resultVarOp;
    }
}
