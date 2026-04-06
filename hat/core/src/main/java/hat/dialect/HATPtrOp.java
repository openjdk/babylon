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

import hat.device.DeviceSchema;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;
import jdk.incubator.code.*;
import optkl.util.ops.Precedence;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;

public abstract sealed class HATPtrOp extends HATOp
        permits HATPtrOp.HATPtrLengthOp, HATPtrOp.HATPtrLoadOp, HATPtrOp.HATPtrStoreOp {

    private TypeElement resultType;
    private List<String> strides;
    private String name;

    private static final String NAME = "HATPtrOp";

    public HATPtrOp(String name, TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
        this(operands);
        this.resultType = resultType;
        List<String> retValue = List.of();
        if (Modifier.isPublic(bufferClass.getModifiers())) {
            try {
                if (bufferClass.getField("schema").get(null) instanceof Schema<?> schema) {
                    retValue = schema.rootIfaceType.fields
                            .stream()
                            .map(fieldNode -> fieldNode.name)
                            .toList();

                    if (!retValue.isEmpty()) {
                        retValue = retValue.subList(0, retValue.size() - 1);// remove the "array" field from the fields
                    }
                }
            } catch (IllegalAccessException | NoSuchFieldException e) {
                try {
                    if (bufferClass.getField("deviceSchema").get(null) instanceof DeviceSchema deviceSchema) {
                        // We did find a device schema !  I think we should not be getting here with device schemas
                    } else {
                        throw new RuntimeException("No schema or deviceSchema field ");
                    }
                } catch (IllegalAccessException | NoSuchFieldException e2) {
                    throw new RuntimeException("No schema field ", e2);
                }
            }
        }
        this.strides = retValue;
        this.name = name;
    }

    public HATPtrOp(List<Value> operands) {
        super(operands);
    }

    public HATPtrOp(HATPtrOp op, CodeContext copyContext) {
        super(op, copyContext);
        this.resultType = op.resultType;
        this.strides = op.strides;
        this.name = op.name;
    }

    @Override
    public TypeElement resultType() {
        return resultType;
    }

    public List<String> strides() {
        return strides;
    }

    public String name() {
        return name;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect." + NAME, this.resultType());
    }

    public static final class HATPtrStoreOp extends HATPtrOp implements Precedence.Store {

        private static final String NAME = "HATPtrStoreOp";

        public HATPtrStoreOp(String name, TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(name, resultType, bufferClass, operands);
        }

        public HATPtrStoreOp(List<Value> operands) {
            super(operands);
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

        public HATPtrLoadOp(String name, TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(name, resultType, bufferClass, operands);
        }

        public HATPtrLoadOp(List<Value> operands) {
            super(operands);
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

        public HATPtrLengthOp(String name, TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
            super(name, resultType, bufferClass, operands);
        }

        public HATPtrLengthOp(List<Value> operands) {
            super(operands);
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