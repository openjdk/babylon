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
package oracle.code.onnx.metadata;

import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.RecordComponent;
import java.util.HashMap;

public final class ModelMetadataResolver {

    private ModelMetadataResolver() {}

    public static ModelMetadata from(Method method) {
        var parameterShapes = new HashMap<Integer, TensorMetadata>();
        Parameter[] parameters = method.getParameters();
        for (int i = 0; i < parameters.length; i++) {
            TensorMetadata tensorMetadata = toMetadata(parameters[i].getAnnotation(Shape.class), parameters[i].getAnnotation(ElementShape.class));
            if (tensorMetadata != null) {
                parameterShapes.put(i, tensorMetadata);
            }
        }

        var valueShapes = new HashMap<String, TensorMetadata>();
        Class<?> returnType = method.getReturnType();
        if (returnType.isRecord()) {
            for (RecordComponent component : returnType.getRecordComponents()) {
                TensorMetadata tensorMetadata = toMetadata(component.getAnnotation(Shape.class), component.getAnnotation(ElementShape.class));
                if (tensorMetadata != null) {
                    valueShapes.put(component.getName(), tensorMetadata);
                }
            }
        }
        return new ModelMetadata(parameterShapes, valueShapes);
    }

    private static TensorMetadata toMetadata(Shape shape, ElementShape elementShape) {
        if (elementShape != null )
            return new TensorMetadata(elementShape.value(), elementShape.count());
        if (shape != null)
            return new TensorMetadata(shape.value(), TensorMetadata.NO_ELEMENT_COUNT);
        return null;
    }


}
