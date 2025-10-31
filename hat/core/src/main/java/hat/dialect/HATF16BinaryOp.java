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
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;

import java.util.List;
import java.util.Map;

public abstract class HATF16BinaryOp extends HATF16Op {

    protected final TypeElement elementType;
    protected final OpType operationType;
    protected final List<Boolean> references;

    public enum OpType {
        ADD("+"),
        SUB("-"),
        MUL("*"),
        DIV("/");

        String symbol;

        OpType(String symbol) {
            this.symbol = symbol;
        }

        public String symbol() {
            return symbol;
        }
    }

    public HATF16BinaryOp(TypeElement typeElement, OpType operationType, List<Boolean> references, List<Value> operands) {
        super("", operands);
        this.elementType = typeElement;
        this.operationType = operationType;
        this.references = references;
    }

    public HATF16BinaryOp(HATF16BinaryOp op, CopyContext copyContext) {
        super(op, copyContext);
        this.elementType = op.elementType;
        this.operationType = op.operationType;
        this.references = op.references;
    }

    @Override
    public TypeElement resultType() {
        return this.elementType;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect.fp16." + varName(), operationType.symbol());
    }

    public OpType operationType() {
        return operationType;
    }

    public List<Boolean> references() {
        return references;
    }

}
