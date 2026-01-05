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

import optkl.ifacemapper.Schema;
import jdk.incubator.code.*;
import optkl.util.ops.Precedence;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;

public abstract sealed class HATPtrOp extends HATOp
        permits HATPtrOp.HATPtrLengthOp, HATPtrOp.HATPtrLoadOp, HATPtrOp.HATPtrStoreOp {

    private final TypeElement resultType;
    private final Class<?> bufferClass;
    private final List<String> strides;

    private static final String NAME = "HATPtrOp";

    public static List<String> getFieldsOfBuffer(Class<?> clazz) {
        List<String> retValue = List.of();
        if (Modifier.isPublic(clazz.getModifiers())) {
            try {
                if (clazz.getField("schema").get(null/* we expect static */) instanceof Schema<?> schema) {
                    retValue = schema.rootIfaceType.fields
                            .stream()
                            .map(fieldNode -> fieldNode.name)
                            .toList();
                    retValue = retValue.isEmpty()
                            ?retValue
                            :retValue.subList(0, retValue.size() - 1); // is this intended to drop the last one?
                }
            } catch (IllegalAccessException | NoSuchFieldException e) {
                throw new RuntimeException("No schema field ",e);
            }
        }
        return retValue;
    }
    public HATPtrOp(TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
        super(operands);
        this.resultType = resultType;
        this.bufferClass = bufferClass;
        this.strides = getFieldsOfBuffer(bufferClass);
    }

    public HATPtrOp(HATPtrOp op, CodeContext copyContext) {
        super(op, copyContext);
        this.resultType = op.resultType;
        this.bufferClass = op.bufferClass;
        this.strides = op.strides;
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }

    public List<String> strides() {
        return strides;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect." + NAME, this.resultType());
    }

    public static final class HATPtrStoreOp extends HATPtrOp implements Precedence.Store {

        private static final String NAME = "HATPtrStoreOp";

        public HATPtrStoreOp(TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(resultType, bufferClass, operands);
        }

        public HATPtrStoreOp(HATPtrStoreOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATPtrStoreOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect." + NAME, this.resultType());
        }

    }

    public static final class HATPtrLoadOp extends HATPtrOp implements Precedence.LoadOrConv {

        private static final String NAME = "HATPtrLoadOp";

        public HATPtrLoadOp(TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(resultType, bufferClass, operands);
        }

        public HATPtrLoadOp(HATPtrLoadOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATPtrLoadOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect." + NAME, this.resultType());
        }
    }

    public static final class HATPtrLengthOp extends HATPtrOp implements Precedence.LoadOrConv {

        private static final String NAME = "HATPtrLengthOp";

        public HATPtrLengthOp(TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(resultType, bufferClass, operands);
        }

        public HATPtrLengthOp(HATPtrLengthOp op, CodeContext copyContext) {
            super(op, copyContext);
        }

        @Override
        public Op transform(CodeContext copyContext, CodeTransformer opTransformer) {
            return new HATPtrLengthOp(this, copyContext);
        }

        @Override
        public Map<String, Object> externalize() {
            return Map.of("hat.dialect." + NAME, this.resultType());
        }
    }
}