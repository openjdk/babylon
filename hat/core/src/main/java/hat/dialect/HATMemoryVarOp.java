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
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.ClassType;
import optkl.IfaceValue;
import optkl.util.ops.StatementLikeOp;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;

public abstract sealed class HATMemoryVarOp extends HATOp implements VarLikeOp, StatementLikeOp {

    protected final String varName;

    protected HATMemoryVarOp(String varName, List<Value> operands) {
        super(operands);
        this.varName = varName;
    }

    protected HATMemoryVarOp(HATMemoryVarOp that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
    }

    @Override
    public String varName() {
        return varName;
    }

    public abstract ClassType classType();

    public static final class HATVarOp extends HATMemoryVarOp  {

        private final VarType codeType;
        private final Class<?> float16Class;
        private final IfaceValue.Vector.Shape vectorShape;
        private final ClassType klassType;
        private final DeviceRegion deviceRegion;

        // Seems we need to add attributes in the form of( {"Attrib" -> object })
        // float16Class is only needed for F16
        // Vector Shape is only needed in the case of Vectors

        public HATVarOp(String varName, VarType codeType, DeviceRegion deviceRegion, List<Value> operands) {
            super(varName, operands);
            this.codeType = codeType;
            this.float16Class = null;
            this.vectorShape = null;
            this.klassType = null;
            this.deviceRegion = deviceRegion;
        }

        public HATVarOp(String varName, Class<?> float16Class, VarType varType, List<Value> operands) {
            super(varName, operands);
            this.codeType = varType;
            this.float16Class = float16Class;
            this.vectorShape = null;
            this.klassType = null;
            this.deviceRegion = DeviceRegion.NARROW; // if float16Class -> NARROW

        }

        public HATVarOp(String varName, VarType resultType, IfaceValue.Vector.Shape vectorShape, List<Value> operand) {
            super(varName, operand);
            this.codeType = resultType;
            this.vectorShape = vectorShape;
            this.float16Class = null;

            // local
            this.klassType = null;
            this.deviceRegion = DeviceRegion.VECTOR;  // we can infer Vector category because it has a vector shape
        }

        public enum DeviceRegion {
            UNKNOWN,
            PRIVATE,
            SHARED,
            INIT,
            TENSOR,
            NARROW,
            VECTOR,
        }

        // Local/Private Memory Types
        public HATVarOp(String varName, ClassType javaType, VarType varType, DeviceRegion deviceRegion, List<Value> operands) {
            super(varName, operands);
            this.klassType = javaType;
            this.codeType = varType;
            this.deviceRegion = deviceRegion;

            this.float16Class = null;
            this.vectorShape = null;
        }

        public HATVarOp(HATVarOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
            this.float16Class = op.float16Class;
            this.vectorShape = op.vectorShape;

            this.klassType = op.klassType;
            this.deviceRegion = op.deviceRegion;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATVarOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.HATVarOp." + varName, codeType);
        }

        @Override
        public ClassType classType() {
            return klassType;
        }

        public Class<?> float16Class() {
            return float16Class;
        }

        public boolean hasVectorShape() {
            return vectorShape != null;
        }

        public DeviceRegion deviceRegion() {
            return this.deviceRegion;
        }

        public String buildVectorType() {
            return vectorShape.codeType().toString() + vectorShape.lanes();
        }
    }
}
