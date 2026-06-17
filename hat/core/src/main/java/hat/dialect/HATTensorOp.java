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
        private final String loadVariance;

        public TensorLoadOp(CodeType codeType, String loadVariance,List<Value> operands) {
            super(operands);
            this.codeType = codeType;
            this.loadVariance = loadVariance;
        }

        public TensorLoadOp(TensorLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
            this.loadVariance = op.loadVariance;
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

        public String getLoadVariance() {
            return this.loadVariance;
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

    public static final class TensorShapeOp extends HATTensorOp implements Precedence.Invoke {

        private final CodeType codeType;

        public TensorShapeOp(CodeType codeType, List<Value> operands) {
            super(operands);
            this.codeType = codeType;
        }

        public TensorShapeOp(TensorShapeOp op, CodeContext copyContext) {
            super(op, copyContext);
            this.codeType = op.codeType;
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new TensorShapeOp(this, copyContext);
        }

        @Override
        public CodeType resultType() {
            return codeType;
        }

        @Override
        public String externalizeOpName() {
            return "hat.dialect.Tensor.TensorShape";
        }
    }
}
