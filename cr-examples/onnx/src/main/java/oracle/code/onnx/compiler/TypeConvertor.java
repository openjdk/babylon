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

package oracle.code.onnx.compiler;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.lang.reflect.TypeVariable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.TupleType;
import jdk.incubator.code.dialect.java.ArrayType;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import oracle.code.onnx.Tensor;
import oracle.code.onnx.ir.OnnxType;

public class TypeConvertor {

    static final JavaType TENSOR_CLASS = JavaType.type(Tensor.class);

    final MethodHandles.Lookup l;
    final Map<String, Integer> constantArraySizeMap; // RecordComponent cannot be used as a key!

    TypeConvertor(MethodHandles.Lookup l) {
        this.l = l;
        this.constantArraySizeMap = new HashMap<>(); // @@@ initialize
    }

    void detectConstantArrays(CoreOp.FuncOp f) {
        f.traverse(null, (_, ce) -> {
            if (ce instanceof JavaOp.NewOp no && no.resultType() instanceof ClassType recordType && isRecord(recordType)) {
                Class<?> recordClass;
                try {
                    recordClass = (Class<?>) recordType.rawType().resolve(l);
                } catch (ReflectiveOperationException e) {
                    throw new RuntimeException(e);
                }
                var rcs = recordClass.getRecordComponents();
                var ops = no.operands();
                for (int i = 0; i < rcs.length; i++) {
                    RecordComponent rc  = rcs[i];
                    Type type = rc.getGenericType();
                    if (type instanceof ParameterizedType pt && pt.getRawType().equals(Optional.class)) {
                        type = pt.getActualTypeArguments()[0];
                    }
                    if (type instanceof GenericArrayType) {
                        Value arr = OnnxTransformer.skipVars(ops.get(i));
                        if (arr instanceof Op.Result newArrayResult
                                && newArrayResult.op() instanceof JavaOp.NewOp newArrayOp
                                && newArrayOp.operands().getFirst() instanceof Op.Result constantResult
                                && constantResult.op() instanceof CoreOp.ConstantOp cop) {

                            // explicit constant array construction
                            constantArraySizeMap.put(rc.toString(), (Integer)cop.value());
                        } else {
                            // search for the highest array access index
                            scanUse(arr, rc.toString());
                        }
                    }
                }
            }
            return null;
        });
    }

    void scanUse(Value array, String rcKey) {
        for (var use : array.uses()) {
            if (use instanceof Op.Result or) {
                switch (or.op()) {
                    case CoreOp.VarOp vo ->
                        scanUse(vo.result(), rcKey);
                    case CoreOp.VarAccessOp.VarLoadOp vlo ->
                        scanUse(vlo.result(), rcKey);
                    case JavaOp.ArrayAccessOp aao when aao.operands().get(1) instanceof Op.Result constR
                                                    && constR.op() instanceof CoreOp.ConstantOp cop ->
                        constantArraySizeMap.compute(rcKey, (_, i) -> Math.max((Integer)cop.value(), i == null ? 0 : i));
                    default -> {}
                }
            }
        }
    }

    TupleType recordTypeToTupleType(ClassType recordType) {
        Class<?> recordClass;
        try {
            recordClass = (Class<?>) recordType.rawType().resolve(l);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
        assert recordClass.isRecord();

        List<TypeElement> tupleComponentTypes = new ArrayList<>();
        for (RecordComponent rc : recordClass.getRecordComponents()) {
            Type type = rc.getGenericType();
            if (type instanceof ParameterizedType pt && pt.getRawType().equals(Optional.class)) {
                type = pt.getActualTypeArguments()[0];
            }
            switch (type) {
                case ParameterizedType pt -> {
                    Type elementType = pt.getActualTypeArguments()[0];
                    switch (elementType) {
                        case Class<?> _ -> {
                            tupleComponentTypes.add(convertType(JavaType.type(pt)));
                        }
                        case TypeVariable<?> tv -> {
                            // Resolve type variable
                            JavaType e = null;
                            for (int j = 0; j < recordClass.getTypeParameters().length; j++) {
                                if (recordClass.getTypeParameters()[j].getName().equals(tv.getName())) {
                                    e = recordType.typeArguments().get(j);
                                    break;
                                }
                            }
                            tupleComponentTypes.add(convertType(JavaType.parameterized(JavaType.type(Tensor.class), e)));
                        }
                        default -> throw new IllegalStateException("Unexpected value: " + elementType);
                    }
                }
                case TypeVariable tv -> {
                    // Resolve type variable
                    JavaType e = null;
                    for (int j = 0; j < recordClass.getTypeParameters().length; j++) {
                        if (recordClass.getTypeParameters()[j].getName().equals(tv.getName())) {
                            e = recordType.typeArguments().get(j);
                            break;
                        }
                    }
                    tupleComponentTypes.add(convertType(e));
                }
                case GenericArrayType gat -> {
                    var cType = convertType(JavaType.type(gat.getGenericComponentType()));
                    Integer size = constantArraySizeMap.get(rc.toString());
                    var tContent = new TypeElement[size];
                    Arrays.fill(tContent, cType);
                    tupleComponentTypes.add(CoreType.tupleType(tContent));
                }
                default -> throw new IllegalStateException("Unexpected value: " + rc.getGenericType());
            }
        }

        return CoreType.tupleType(tupleComponentTypes);
    }

    boolean isRecord(TypeElement type) {
        try {
            return type instanceof ClassType ct &&
                    ct.erasure().resolve(l) instanceof Class c &&
                    c.isRecord();
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }


    Integer recordComponentAccessToTupleIndex(MethodRef ref) {
        if (ref.refType() instanceof ClassType ct) {
            Class<?> refClass;
            try {
                refClass = (Class<?>) ct.resolve(l);
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException(e);
            }

            if (refClass.isRecord()) {
                RecordComponent[] recordComponents = refClass.getRecordComponents();
                for (int i = 0; i < recordComponents.length; i++) {
                    if (recordComponents[i].getName().equals(ref.name())) {
                        return i;
                    }
                }
                throw new InternalError();
            }
        }
        return null;
    }

    FunctionType convertType(FunctionType t) {
        return CoreType.functionType(convertType(t.returnType()), t.parameterTypes().stream().map(this::convertType).toList());
    }

    FunctionType convertType(CoreOp.FuncOp fo) {
        return CoreType.functionType(convertType(fo.body().entryBlock().terminatingOp().operands().getFirst()), fo.parameters().stream().map(this::convertType).toList());
    }

    TypeElement convertType(Value value) {
        // convert 1-dimensional constantly accessed constant arrays into tuples
        if (value.type() instanceof ArrayType at && at.dimensions() == 1) {
            int size = countConstantArraySize(value.uses());
            if (size >= 0) {
                var targs = new TypeElement[size];
                Arrays.fill(targs, convertType(at.componentType()));
                return CoreType.tupleType(targs);
            }
        }
        return convertType(value.type());
    }

    static int countConstantArraySize(Set<Op.Result> uses) {
        int size = 0;
        for (var use : uses) {
            int s = switch (use.op()) {
                case JavaOp.ArrayAccessOp aao when aao.operands().get(1) instanceof Op.Result or && or.op() instanceof CoreOp.ConstantOp co ->
                    (Integer)co.value() + 1;
                case CoreOp.VarOp _, CoreOp.VarAccessOp.VarLoadOp _ ->
                    countConstantArraySize(use.op().result().uses());
                default -> -1;
            };
            if (s < 0) return -1;
            size = Integer.max(size, s);
        }
        return size;
    }

    // @@@ Map of Java tensor types to ONNX tensor types
    // @@@ Shape??
    TypeElement convertType(TypeElement type) {
        if (type instanceof ClassType ct) {
            if (ct.rawType().equals(TENSOR_CLASS)) {
                JavaType elementType = ct.typeArguments().getFirst();
                if (elementType.equals(JavaType.J_L_INTEGER)) {
                    return OnnxType.TENSOR_INT32;
                } else if (elementType.equals(JavaType.J_L_FLOAT)) {
                    return OnnxType.TENSOR_FLOAT32;
                } else if (elementType.equals(JavaType.J_L_LONG)) {
                    return OnnxType.TENSOR_INT64;
                } else if (elementType.equals(JavaType.J_L_BYTE)) {
                    return OnnxType.TENSOR_UINT8;
                } else if (elementType.equals(JavaType.J_L_BOOLEAN)) {
                    return OnnxType.TENSOR_BOOL;
                }
            } else if (isRecord(type)) {
                return recordTypeToTupleType(ct);
            }
        }
        return type;
    }
}
