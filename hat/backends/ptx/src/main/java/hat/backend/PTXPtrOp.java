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
package hat.backend;

import hat.ifacemapper.Schema;

import java.lang.reflect.code.*;
import java.lang.reflect.code.op.ExternalizableOp;
import java.util.List;

public class PTXPtrOp extends ExternalizableOp {
    public String fieldName;
    public static final String NAME = "ptxPtr";
    final TypeElement resultType;
    public Schema<?> schema;

    PTXPtrOp(TypeElement resultType, String fieldName, List<Value> operands, Schema<?> schema) {
        super(NAME, operands);
        this.resultType = resultType;
        this.fieldName = fieldName;
        this.schema = schema;
    }

    PTXPtrOp(PTXPtrOp that, CopyContext cc) {
        super(that, cc);
        this.resultType = that.resultType;
        this.fieldName = that.fieldName;
        this.schema = that.schema;
    }

    @Override
    public PTXPtrOp transform(CopyContext cc, OpTransformer ot) {
        return new PTXPtrOp(this, cc);
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }
}
