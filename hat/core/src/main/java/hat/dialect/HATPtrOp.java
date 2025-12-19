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
import optkl.ifacemapper.Schema;
import jdk.incubator.code.*;

import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Map;

public abstract class HATPtrOp extends HATOp {

    private final TypeElement resultType;
    private final Class<?> bufferClass;
    private final List<String> strides;

    private static final String NAME = "HATPtrOp";

    public HATPtrOp(TypeElement resultType, Class<?> bufferClass, List<Value> operands) {
        super(operands);
        this.resultType = resultType;
        this.bufferClass = bufferClass;
        List<String> fields = getFieldsOfBuffer(bufferClass);
        this.strides = (fields == null) ? List.of() : fields.subList(0, fields.size() - 1);
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

    public Class<?> bufferClass() {
        return bufferClass;
    }


    public List<String> strides() {
        return strides;
    }

    @Override
    public Map<String, Object> externalize() {
        return Map.of("hat.dialect." + NAME, this.resultType());
    }

    public static List<String> getFieldsOfBuffer(Class<?> clazz) {
        try {
            if (!Modifier.isPublic(clazz.getModifiers())) return null;
            Object obj = clazz.getField("schema").get(null);
            if (obj instanceof DeviceSchema<?> deviceSchema) {
                return null;
            } else if (obj instanceof Schema<?> schema) {
                return schema.rootIfaceType.fields.stream().map(fieldNode -> fieldNode.name).toList();
            }
            return null;
        } catch (IllegalAccessException | NoSuchFieldException e) {
            throw new RuntimeException(e);
        }
        // return Schema.of(clazz).rootIfaceType.fields;
    }
}