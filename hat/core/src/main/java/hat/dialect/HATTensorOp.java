/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
import optkl.util.ops.Precedence;
import optkl.util.ops.StatementLikeOp;
import optkl.util.ops.VarLikeOp;

import java.util.List;
import java.util.Map;

public abstract sealed class HATTensorOp extends HATOp {

    protected HATTensorOp(List<Value> operands) {
        super(operands);
    }

    protected HATTensorOp(Op that, CodeContext cc) {
        super(that, cc);
    }

    public static final class TensorVarOp extends HATTensorOp implements VarLikeOp, StatementLikeOp {

        private final VarType codeType;
        private final String varName;

        public TensorVarOp(String varName, VarType codeType, List<Value> operands) {
            super(operands);
            this.varName = varName;
            this.codeType = codeType;
        }

        public TensorVarOp(TensorVarOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.varName = op.varName;
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorVarOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect.TensorVarOp." + varName, codeType);
        }

        @Override
        public String varName() {
            return varName;
        }
    }

    public static final class TensorCreateOp extends HATTensorOp implements Precedence.Invoke {

        private final CodeType codeType;

        public TensorCreateOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorCreateOp(TensorCreateOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorCreateOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.Tensor.CreateOp";
        }
    }

    public static final class TensorVarLoadOp extends HATTensorOp implements Precedence.LoadOrConv  {

        private final CodeType codeType;

        public TensorVarLoadOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;

        }

        public TensorVarLoadOp(TensorVarLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorVarLoadOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.TensorVarLoadOp";
        }
    }

    public static final class TensorFillOp extends HATTensorOp implements Precedence.Invoke {

        private final CodeType codeType;

        public TensorFillOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorFillOp(TensorFillOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorFillOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.Tensor.fill";
        }
    }

    public static final class TensorMMAOp extends HATTensorOp implements Precedence.Invoke {

        private final CodeType codeType;

        public TensorMMAOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorMMAOp(TensorMMAOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorMMAOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.Tensor.MMA";
        }
    }

    public static final class TensorStoreLoadOp extends HATTensorOp implements Precedence.Store  {

        private final CodeType codeType;

        public TensorStoreLoadOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;

        }

        public TensorStoreLoadOp(TensorStoreLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorStoreLoadOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.TensorStoreLoadOp";
        }
    }

    public static final class TensorLoadOp extends HATTensorOp implements Precedence.LoadOrConv {

        private final CodeType codeType;

        public TensorLoadOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorLoadOp(TensorLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorLoadOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.TensorLoadOp";
        }
    }

    public static final class TensorStoreOp extends HATTensorOp implements Precedence.LoadOrConv {

        private final CodeType codeType;

        public TensorStoreOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorStoreOp(TensorStoreOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorStoreOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.TensorStoreOp";
        }
    }


}
