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
package hat.phases;

import hat.dialect.HATF16Op;
import hat.dialect.HATMemoryVarOp;
import hat.dialect.HATVectorOp;
import hat.types.BF16;
import hat.types.F16;
import hat.types._V;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpHelper;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

import static optkl.OpHelper.Invoke.invoke;
import static optkl.OpHelper.resultFromFirstOperandOrNull;

public class HATPhaseUtils {

    static HATArrayViewPhase.ArrayAccessInfo arrayAccessInfo(Value value, Map<Op.Result, Op.Result> replaced) {
        return expressionGraph(value).getInfo(replaced);
    }

    static HATArrayViewPhase.Node<Value> expressionGraph(Value value) {
        return expressionGraph(new HashMap<>(), value);
    }

    static HATArrayViewPhase.Node<Value> expressionGraph(Map<Value, HATArrayViewPhase.Node<Value>> visited, Value value) {
        // If value has already been visited return its node
        if (visited.containsKey(value)) {
            return visited.get(value);
        }

        // Find the expression graphs for each operand
        List<HATArrayViewPhase.Node<Value>> edges = new ArrayList<>();

        // looks like
        for (Value operand : value.dependsOn()) {
            if (operand instanceof Op.Result res &&
                    res.op() instanceof JavaOp.InvokeOp iop
                    && iop.invokeDescriptor().name().toLowerCase().contains("arrayview")){ // We need to find a better way
                continue;
            }
            edges.add(expressionGraph(operand));
        }
        HATArrayViewPhase.Node<Value> node = new HATArrayViewPhase.Node<>(value, edges);
        visited.put(value, node);
        return node;
    }

    static HATVectorOp.HATVectorBinaryOp buildVectorBinaryOp(MethodHandles.Lookup lookup, String opType, String varName, TypeElement resultType, List<Value> outputOperands) {
        VectorMetaData md = getVectorTypeInfoWithCodeReflection(lookup,resultType);
        return switch (opType) {
            case "add" -> new HATVectorOp.HATVectorBinaryOp.HATVectorAddOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "sub" -> new HATVectorOp.HATVectorBinaryOp.HATVectorSubOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "mul" -> new HATVectorOp.HATVectorBinaryOp.HATVectorMulOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            case "div" -> new HATVectorOp.HATVectorBinaryOp.HATVectorDivOp(varName, resultType, md.vectorTypeElement(), md.lanes(), outputOperands);
            default -> throw new IllegalStateException("Unexpected value: " + opType);
        };
    }

    static public boolean isVectorOp(MethodHandles.Lookup lookup, Op op) {
        if (!op.operands().isEmpty()) {
           TypeElement type = OpHelper.firstOperandOrThrow(op).type();
           if (type instanceof ArrayType at) {
               type = at.componentType();
           }
           if (type instanceof ClassType ct) {
               try {
                   return _V.class.isAssignableFrom((Class<?>) ct.resolve(lookup));
               } catch (ReflectiveOperationException e) {
                   throw new RuntimeException(e);
              }
           }
        }
        return false;
    }

    static public boolean isBufferArray(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrThrow(op, JavaOp.InvokeOp.class);
        return iop.invokeDescriptor().name().toLowerCase().contains("arrayview"); // we need a better way
    }

    static public boolean isLocalSharedOrPrivate(Op op) {
        JavaOp.InvokeOp iop = (JavaOp.InvokeOp) findOpInResultFromFirstOperandsOrThrow(op, JavaOp.InvokeOp.class);
        return iop.invokeDescriptor().name().toLowerCase().contains("local") || // we need a better way
                iop.invokeDescriptor().name().toLowerCase().contains("shared") || // also
                iop.invokeDescriptor().name().toLowerCase().contains("private"); // also
    }

    static  public Op findOpInResultFromFirstOperandsOrNull(Op op, Class<?> ...classes) {
        Set<Class<?>> set =Set.of(classes);
        while (!(set.contains(op.getClass()))) {
            if (resultFromFirstOperandOrNull(op) instanceof Op.Result result) {
                op = result.op();
            } else {
                return null;
            }
        }
        return op;
    }

    static public Op findOpInResultFromFirstOperandsOrThrow(Op op, Class<?> ...classes) {
          if (findOpInResultFromFirstOperandsOrNull(op,classes) instanceof Op found){
              return found;
          }else{
              throw new RuntimeException("Expecting to find one of "+List.of(classes));
          }
    }

    static public boolean isBufferInitialize(Op op) {
        // first check if the return is an array type
        if (op instanceof CoreOp.VarOp vop) {
            if (!(vop.varValueType() instanceof ArrayType)){
                return false;
            }
        } else if (!(op instanceof JavaOp.ArrayAccessOp)) {
            if (!(op.resultType() instanceof ArrayType)) {
                return false;
            }
        }
        return isBufferArray(op);
    }

    //recursive
    public static boolean isF16Local(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isF16Local(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> true;
            default -> false;
        };
    }

    //recursive
    public static boolean isF16Local(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isF16Local(varLoadOp.operands().getFirst());
    }

    //recursive
    static boolean isArrayReference(MethodHandles.Lookup lookup, Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isArrayReference(lookup,varLoadOp); // recurse
            case CoreOp.VarOp varOp ->
                    varOp.operands().getFirst() instanceof Op.Result varOpResult
                            && invoke(lookup,varOpResult.op()) instanceof OpHelper.Invoke invoke && invoke.named("array");
            default -> false;
        };
    }

    //recursive
    private static boolean isArrayReference(MethodHandles.Lookup lookup, CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isArrayReference(lookup,varLoadOp.operands().getFirst());
    }

    static boolean isOperandF32(Value v) {
        return v instanceof Op.Result r && switch (r.op()) {
            case CoreOp.VarAccessOp varLoadOp -> varLoadOp.varType().valueType() == JavaType.FLOAT; //recurse
            case CoreOp.VarOp varOp -> varOp.resultType().valueType() == JavaType.FLOAT;
            default -> false;
        };
    }

    // recursive
    private static String findVarNameOrNull(Value v) {
        return  (v instanceof Op.Result r) ? switch (r.op()){
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp-> findVarNameOrNull(varLoadOp); //recurse
            case HATF16Op.HATF16VarOp hatf16VarOp -> hatf16VarOp.varName();
            default -> null;
        }:null;
    }

    // recursive
    static String findVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVarNameOrNull(varLoadOp.operands().getFirst());
    }

    static public boolean is16BitFloat(OpHelper.Invoke invoke, Regex methodName) {
        return invoke.refIs(F16.class,BF16.class) && invoke.named(methodName);
    }

    // recursive
    public static TypeElement findVectorTypeElement(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorTypeElement(varLoadOp); // recurse
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorElementType();
            }
            return null;
        }
    }

    // recursive
    public static TypeElement findVectorTypeElement(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorTypeElement(varLoadOp.operands().getFirst());
    }

    //recursive
    public static int getVectorWidth(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return getVectorWidth(varLoadOp.operands().getFirst());
    }

    //recursive
    private static int getVectorWidth(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return getVectorWidth(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.vectorN();
            }
            return -1;
        }
    }

    //recursive
    public static boolean isSharedOrPrivate(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return isSharedOrPrivate(varLoadOp.operands().getFirst());
    }

    //recursive
    public static boolean isSharedOrPrivate(Value v) {
        return v instanceof Op.Result result && switch (result.op()) {
            case CoreOp.VarAccessOp.VarLoadOp varLoadOp -> isSharedOrPrivate(varLoadOp); //recurse
            case HATMemoryVarOp.HATLocalVarOp _, HATMemoryVarOp.HATPrivateVarOp _ -> true;
            default -> false;
        };
    }

    // recursive
    public static String findVectorVarNameOrNull(CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
        return findVectorVarNameOrNull(varLoadOp.operands().getFirst());
    }

    // recursive
    public static String findVectorVarNameOrNull(Value v) {
        if (v instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            return findVectorVarNameOrNull(varLoadOp);
        } else {
            // Leaf of tree -
            if (v instanceof CoreOp.Result r && r.op() instanceof HATVectorOp hatVectorOp) {
                return hatVectorOp.varName();
            }
            return null;
        }
    }

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
        Class<?> clazz = (Class<?>) OpHelper.classTypeToTypeOrThrow(lookup, (ClassType) typeElement);
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
