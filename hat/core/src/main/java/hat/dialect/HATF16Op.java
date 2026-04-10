/*
 * Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.VarType;
import optkl.util.ops.Precedence;
import optkl.util.ops.StatementLikeOp;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;

public abstract sealed class HATF16Op extends HATOp implements VarLikeOp {

    private String varName;

    protected HATF16Op(String varName, List<Value> operands) {
        super(operands);
        this.varName = varName;
    }

    protected HATF16Op(HATF16Op that, CodeContext cc) {
        super(that, cc);
        this.varName = that.varName;
    }
    @Override
    public String varName() {
        return varName;
    }

    public void  varName(String varName) {
        this.varName = varName;
    }

    public static final class HATF16VarOp extends HATF16Op implements StatementLikeOp {

        private final VarType varType;
        private final Class<?> float16Class;

        public HATF16VarOp(String varName, Class<?> float16Class, VarType varType, List<Value> operands) {
            super(varName, operands);
            this.varType = varType;
            this.float16Class = float16Class;
        }

        public HATF16VarOp(HATF16VarOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.varType = op.varType;
            this.float16Class = op.float16Class;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATF16VarOp(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return varType;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.fp16varop." + varName(), varType);
        }

        public Class<?> float16Class() {
            return float16Class;
        }

    }

    public static final class HATF16VarLoadOp extends HATF16Op implements Precedence.LoadOrConv {

        private final VarType typeElement;

        public HATF16VarLoadOp(String varName, VarType typeElement, List<Value> operands) {
            super(varName, operands);
            this.typeElement = typeElement;
        }

        public HATF16VarLoadOp(HATF16VarLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.typeElement = op.typeElement;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATF16VarLoadOp(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeElement;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.fp16VarOp." + varName(), typeElement);
        }

    }

    public static final class HATF16ToFloatConvOp extends HATF16Op implements Precedence.LoadOrConv {

        private final TypeElement typeElement;
        private final boolean isLocal;
        private final boolean wasFloat;
        private final Class<?> float16Class;

        public HATF16ToFloatConvOp(TypeElement typeElement, Class<?> float16Class, boolean isLocal, boolean wasFloat, List<Value> operands) {
            super("", operands);
            this.typeElement = typeElement;
            this.isLocal = isLocal;
            this.wasFloat = wasFloat;
            this.float16Class = float16Class;
        }

        public HATF16ToFloatConvOp(HATF16ToFloatConvOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.typeElement = op.typeElement;
            this.isLocal = op.isLocal;
            this.wasFloat = op.wasFloat;
            this.float16Class = op.float16Class;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATF16ToFloatConvOp(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeElement;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.f16ToFloat", typeElement);
        }

        public boolean isLocal() {
            return isLocal;
        }

        public boolean wasFloat() {
            return wasFloat;
        }

        public Class<?> float16Class() {
            return float16Class;
        }

    }

    public static final class HATF16ConvOp extends HATF16Op {

        private final TypeElement typeElement;
        private final Class<?> float16Class;

        public HATF16ConvOp(TypeElement typeElement, Class<?> float16Class, List<Value> operands) {
            super("", operands);
            this.typeElement = typeElement;
            this.float16Class = float16Class;
        }

        public HATF16ConvOp(HATF16ConvOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.typeElement = op.typeElement;
            this.float16Class = op.float16Class;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATF16ConvOp(this, copyContext);
        }

        @Override
        public TypeElement resultType() {
            return typeElement;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.f16Conv", typeElement);
        }

        public Class<?> float16Class() {
            return float16Class;
        }

    }

    public abstract static sealed class HATF16BinaryOp extends HATF16Op {

        protected final TypeElement typeElement;
        protected final BinaryOpEnum operationType;
        protected final Class<?> float16Class;

        public static final byte FIRST_OP = 0x01;
        public static final byte LAST_OP = 0x10;

        protected HATF16BinaryOp(TypeElement typeElement, Class<?> float16Class, BinaryOpEnum operationType, List<Value> operands) {
            super("", operands);
            this.typeElement = typeElement;
            this.operationType = operationType;
            this.float16Class = float16Class;
        }

        protected HATF16BinaryOp(HATF16BinaryOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.typeElement = op.typeElement;
            this.operationType = op.operationType;
            this.float16Class = op.float16Class;
        }

        @Override
        public TypeElement resultType() {
            return this.typeElement;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.fp16." + varName(), operationType.symbol());
        }

        public BinaryOpEnum binaryOperationType() {
            return operationType;
        }

        public Class<?> float16Class() {
            return float16Class;
        }

        public static final class HATF16AddOp extends HATF16BinaryOp implements Precedence.Additive {

            public HATF16AddOp(TypeElement typeElement, Class<?> float16Class, List<Value> operands) {
                super(typeElement, float16Class, BinaryOpEnum.ADD, operands);
            }

            public HATF16AddOp(HATF16AddOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATF16AddOp(this, copyContext);
            }
        }

        public static final class HATF16DivOp extends HATF16BinaryOp implements Precedence.Multiplicative {

            public HATF16DivOp(TypeElement typeElement, Class<?> float16Class, List<Value> operands) {
                super(typeElement, float16Class, BinaryOpEnum.DIV, operands);
            }

            public HATF16DivOp(HATF16DivOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATF16DivOp(this, copyContext);
            }
        }

        public static final class HATF16MulOp extends HATF16BinaryOp implements Precedence.Multiplicative {

            public HATF16MulOp(TypeElement typeElement, Class<?> float16Class, List<Value> operands) {
                super(typeElement, float16Class, BinaryOpEnum.MUL, operands);
            }

            public HATF16MulOp(HATF16MulOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATF16MulOp(this, copyContext);
            }
        }

        public static final class HATF16SubOp extends HATF16BinaryOp implements Precedence.Additive {

            public HATF16SubOp(TypeElement typeElement, Class<?> float16Class, List<Value> operands) {
                super(typeElement, float16Class, BinaryOpEnum.SUB, operands);
            }

            public HATF16SubOp(HATF16SubOp op, CodeContext copyContext) {
                super(op, copyContext);
            }

            @Override
            public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
                return new HATF16SubOp(this, copyContext);
            }
        }
    }
}