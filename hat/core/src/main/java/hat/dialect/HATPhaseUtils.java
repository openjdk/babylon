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
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;

import java.lang.reflect.Method;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

public class HATPhaseUtils {

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

    public record VectorMetaData(TypeElement vectorTypeElement, int lanes) {
    }

    public static VectorMetaData getVectorTypeInfo(JavaOp.InvokeOp invokeOp, int param) {
        Value varValue = invokeOp.operands().get(param);
        if (varValue instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorTypeInfoWithCodeReflection(varLoadOp.resultType());
        }
        return null;
    }

    public static VectorMetaData getVectorTypeInfo(JavaOp.InvokeOp invokeOp) {
        return getVectorTypeInfoWithCodeReflection(invokeOp.resultType());
    }

    private static CoreOp.FuncOp buildCodeModelFor(Class<?> klass, String methodName) {
        Optional<Method> methodFunction = Stream.of(klass.getMethods())
                .filter(m -> m.getName().equals(methodName))
                .findFirst();
        return Op.ofMethod(methodFunction.get()).get();
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
    public static VectorMetaData getVectorTypeInfoWithCodeReflection(TypeElement typeElement) {
        Class<?> aClass;
        try {
            aClass = Class.forName(typeElement.toString());
        } catch (ClassNotFoundException e) {
            // TODO: Add control for exceptions in HAT (HATExceptions Handler)
            throw new RuntimeException(e);
        }
        CoreOp.FuncOp codeModelType = buildCodeModelFor(aClass, "type");
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
        CoreOp.FuncOp codeModelWidth = buildCodeModelFor(aClass, "width");
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

    public static int getWitdh(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getWitdh(varLoadOp.operands().getFirst());
    }

    public static int getWitdh(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getWitdh(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorN();
            }
            return -1;
        }
    }

    public static TypeElement findVectorTypeElement(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorTypeElement(varLoadOp.operands().getFirst());
    }

    public static TypeElement findVectorTypeElement(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorTypeElement(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorElementType;
            }
            return null;
        }
    }

    public static String findNameVector(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findNameVector(varLoadOp.operands().getFirst());
    }

    public static String findNameVector(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findNameVector(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.varName();
            }
            return null;
        }
    }

    public static boolean findF16IsLocal(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findF16IsLocal(varLoadOp.operands().getFirst());
    }

    public static boolean findF16IsLocal(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findF16IsLocal(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATF16VarOp hatf16VarOp) {
                return true;
            }
            return false;
        }
    }

}
