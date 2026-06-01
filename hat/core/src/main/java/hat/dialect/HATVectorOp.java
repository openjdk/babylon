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

import optkl.IfaceValue.Vector;
import jdk.incubator.code.CodeContext;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;

public abstract sealed class HATVectorOp extends HATOp implements VarLikeOp {

    // TODO all these fields should be final
    private  String varName;
    private final CodeType resultType;
    private final Vector.Shape vectorShape;

    protected HATVectorOp(String varName, CodeType resultType, Vector.Shape vectorShape, List<Value> operands) {
        super(operands);
        this.varName = varName;
        this.resultType = resultType;
        this.vectorShape = vectorShape;
    }

    protected HATVectorOp(HATVectorOp that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
        this.resultType = that.resultType;
        this.vectorShape = that.vectorShape;
    }

    @Override
    public final String varName() {
        return varName;
    }

    public final Vector.Shape vectorShape() {
        return vectorShape;
    }

    //TODO all these fields should be final why do we allow this to be mutated.
    public void varName(String varName) {
        this.varName = varName;
    }

    public String mapLane(int lane) {
        return switch (lane) {
            case 0 -> "x";
            case 1 -> "y";
            case 2 -> "z";
            case 3 -> "w";
            default -> throw new InternalError("Invalid lane: " + lane);
        };
    }
    @Override
    public final CodeType resultType() {
        return resultType;
    }
    public String buildType() {
        return vectorShape.codeType().toString() + vectorShape.lanes();
    }

    public static non-sealed class HATVectorBinaryOp extends HATVectorOp {

        private final BinaryOpEnum operationType;
        public HATVectorBinaryOp(String varName,  BinaryOpEnum operationType, CodeType codeType, Vector.Shape vectorShape, List<Value> operands) {
            super(  varName /* this is clearly wrong binary ops have no name */,
                    codeType,
                    vectorShape,
                    operands);
            this.operationType = operationType;
        }

        public HATVectorBinaryOp(HATVectorBinaryOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.operationType = op.operationType;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.HATVectorBinaryOp." + operationType() + "--" + varName(), resultType());
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVectorBinaryOp(this, copyContext);
        }

        public BinaryOpEnum operationType() {
            return operationType;
        }
    }
}