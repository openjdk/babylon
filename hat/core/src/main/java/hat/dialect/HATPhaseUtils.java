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

import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpTkl;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

public class HATPhaseUtils {

    public record VectorMetaData(TypeElement vectorTypeElement, int lanes) {
    }

    public static VectorMetaData getVectorTypeInfo(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, int param) {
        Value varValue = invokeOp.operands().get(param);
        if (varValue instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorTypeInfoWithCodeReflection(lookup,varLoadOp.resultType());
        }
        return null;
    }

    public static VectorMetaData getVectorTypeInfo(MethodHandles.Lookup lookup,JavaOp.InvokeOp invokeOp) {
        return getVectorTypeInfoWithCodeReflection(lookup,invokeOp.resultType());
    }
    public static TypeElement getVectorElementType(String primitive) {
        return switch (primitive) {
            case "float" -> JavaType.FLOAT;
            case "double" -> JavaType.DOUBLE;
            case "int" -> JavaType.INT;
            case "long" -> JavaType.LONG;
            case "short" -> JavaType.SHORT;
            case "byte" -> JavaType.BYTE;
            case "char" -> JavaType.CHAR;
            case "boolean" -> JavaType.BOOLEAN;
            default -> null;
        };
    }

    /**
     * This method inspects the Vector Type Methods to obtain two methods for code-model:
     * 1) Method `type` to obtain the primitive base type of the vector type.
     * 2) Method `width` to obtain the number of lanes.
     *
     * @param typeElement
     *  {@link TypeElement}
     * @return
     * {@link VectorMetaData}
     */
    public static VectorMetaData getVectorTypeInfoWithCodeReflection(MethodHandles.Lookup lookup,TypeElement typeElement) {
        Class<?> clazz = (Class<?>)OpTkl.classTypeToTypeOrThrow(lookup, (ClassType) typeElement);
        CoreOp.FuncOp codeModelType = buildCodeModelFor(clazz, "type");
        AtomicReference<TypeElement> vectorElement = new AtomicReference<>();
        codeModelType.elements().forEach(codeElement -> {
            if (codeElement instanceof CoreOp.ReturnOp returnOp) {
                Value v = returnOp.operands().getFirst();
                if (v instanceof Op.Result r && r.op() instanceof JavaOp.FieldAccessOp.FieldLoadOp fieldLoadOp) {
                    String primitiveTypeName = fieldLoadOp.fieldDescriptor().name();
                    vectorElement.set(getVectorElementType(primitiveTypeName.toLowerCase()));
                }
            }
        });

        AtomicInteger lanes = new AtomicInteger(1);
        CoreOp.FuncOp codeModelWidth = buildCodeModelFor(clazz, "width");
        codeModelWidth.elements().forEach(codeElement -> {
            if (codeElement instanceof CoreOp.ReturnOp returnOp) {
                Value v = returnOp.operands().getFirst();
                if (v instanceof Op.Result r && r.op() instanceof CoreOp.ConstantOp constantOp) {
                    lanes.set((Integer) constantOp.value());
                }
            }
        });
        return new VectorMetaData(vectorElement.get(), lanes.get());
    }


    private static CoreOp.FuncOp buildCodeModelFor(Class<?> klass, String methodName) {
        Optional<Method> methodFunction = Stream.of(klass.getMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();
        return Op.ofMethod(methodFunction.get()).get();
    }

}
