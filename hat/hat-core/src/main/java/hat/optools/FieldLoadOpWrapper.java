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
package hat.optools;

import jdk.incubator.code.TypeElement;
import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.ClassType;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;

public class FieldLoadOpWrapper extends FieldAccessOpWrapper<CoreOp.FieldAccessOp.FieldLoadOp> implements LoadOpWrapper {
    FieldLoadOpWrapper( MethodHandles.Lookup lookup,CoreOp.FieldAccessOp.FieldLoadOp op) {
        super(lookup,op);
    }

    public Object getStaticFinalPrimitiveValue() {
        if (fieldType() instanceof ClassType classType) {
            Class<?> clazz = (Class<?>) classTypeToType(classType);
            try {
                Field field = clazz.getField(fieldName());
                field.setAccessible(true);
                return field.get(null);
            } catch (NoSuchFieldException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }
        throw new RuntimeException("Could not find field value" + fieldName() + " in " + fieldType());
    }
}
