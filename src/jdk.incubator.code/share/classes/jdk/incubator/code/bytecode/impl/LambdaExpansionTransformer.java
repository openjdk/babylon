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
package jdk.incubator.code.bytecode.impl;

import java.lang.constant.ClassDesc;
import java.lang.constant.DirectMethodHandleDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.SequencedMap;
import java.util.Set;
import java.util.stream.Stream;

import jdk.incubator.code.Block;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.CodeType;
import jdk.incubator.code.Op;
import jdk.incubator.code.Quoted;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.core.FunctionType;
import jdk.incubator.code.dialect.core.VarType;
import jdk.incubator.code.dialect.java.FieldRef;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;
import jdk.incubator.code.extern.DialectFactory;
import jdk.incubator.code.internal.OpBuilder;
import jdk.incubator.code.runtime.ReflectableLambdaMetafactory;

import static java.lang.constant.ConstantDescs.*;

/**
 * Lambda expansion transformer generates a module with lambda operations replaced
 * by dynamic function calls and with synthetic functions for lambda bodies and
 * reflectable lambda model builders.
 */
final class LambdaExpansionTransformer implements CodeTransformer {

    private static final DirectMethodHandleDesc DMHD_LAMBDA_METAFACTORY = ofCallsiteBootstrap(
            LambdaMetafactory.class.describeConstable().orElseThrow(),
            "metafactory",
            CD_CallSite, CD_MethodType, CD_MethodHandle, CD_MethodType);

    private static final DirectMethodHandleDesc DMHD_REFLECTABLE_LAMBDA_METAFACTORY = ofCallsiteBootstrap(
            ReflectableLambdaMetafactory.class.describeConstable().orElseThrow(),
            "metafactory",
            CD_CallSite, CD_MethodType, CD_MethodHandle, CD_MethodType);

    private final MethodHandles.Lookup lookup;
    private final Set<String> names;
    private final List<FuncOp> functions = new ArrayList<>();
    private final LinkedHashMap<String, FuncOp> modelsToBuild = new LinkedHashMap<>();
    private int nextLambdaIndex;

    private LambdaExpansionTransformer(MethodHandles.Lookup lookup, Set<String> names) {
        this.lookup = lookup;
        this.names = new LinkedHashSet<>(names);
    }

    static <O extends Op & Op.Invokable> CoreOp.ModuleOp transform(MethodHandles.Lookup lookup,
                                                                   SequencedMap<String, ? extends O> ops) {
        return new LambdaExpansionTransformer(lookup, ops.sequencedKeySet()).transform(ops);
    }

    private <O extends Op & Op.Invokable> CoreOp.ModuleOp transform(SequencedMap<String, ? extends O> ops) {
        for (var e : ops.sequencedEntrySet()) {
            functions.add(switch (e.getValue()) {
                case FuncOp fop -> fop.transform(e.getKey(), this);
                case JavaOp.LambdaOp lop -> lambdaToFuncOp(e.getKey(), lop).transform(this);
                default -> throw new IllegalArgumentException("Unsupported invokable operation: " + e.getValue());
            });
        }
        if (!modelsToBuild.isEmpty()) {
            functions.addAll(OpBuilder.createBuilderFunctions(
                    modelsToBuild,
                    b -> b.add(JavaOp.fieldLoad(
                            FieldRef.field(JavaOp.class, "JAVA_DIALECT_FACTORY", DialectFactory.class))))
                    .functionTable().sequencedValues());
        }
        return CoreOp.module(functions);
    }

    // LambdaMetafactory implementation methods take captures before lambda parameters.
    private static FuncOp lambdaToFuncOp(String name, JavaOp.LambdaOp lop) {
        List<Value> captures = lop.capturedValues();
        FunctionType lambdaType = lop.invokableSignature();
        ArrayList<CodeType> parameterTypes = new ArrayList<>(captures.size() + lambdaType.parameterTypes().size());
        for (Value v : captures) {
            parameterTypes.add(v.type() instanceof VarType vt ? vt.valueType() : v.type());
        }
        parameterTypes.addAll(lambdaType.parameterTypes());
        return CoreOp.func(name, CoreType.functionType(lambdaType.returnType(), parameterTypes)).body(b -> {
            int i = 0;
            for (Value cv : captures) {
                Value v = b.parameters().get(i++);
                if (cv.type() instanceof VarType) {
                    v = b.add(CoreOp.var(v));
                }
                b.context().mapValue(cv, v);
            }
            b.transformBody(lop.body(), b.parameters().subList(i, b.parameters().size()),
                    CodeTransformer.COPYING_TRANSFORMER);
        });
    }

    private static String uniqueName(Set<String> names, String name) {
        if (names.add(name)) {
            return name;
        }
        for (int i = 0; ; i++) {
            String n = name + "$" + i;
            if (names.add(n)) {
                return n;
            }
        }
    }

    @Override
    public Block.Builder acceptOp(Block.Builder block, Op op) {
        if (!(op instanceof JavaOp.LambdaOp lop)) {
            block.add(op);
            return block;
        }
        JavaType intfType = (JavaType) lop.functionalInterface();
        MethodTypeDesc mtd = MethodRef.toNominalDescriptor(lop.invokableSignature());
        try {
            Class<?> intfClass = (Class<?>) intfType.erasure().resolve(lookup);
            Method intfMethod = funcIntfMethod(intfClass, mtd);
            List<Value> captures = lop.capturedValues();
            int i = nextLambdaIndex++;
            String implName = uniqueName(names, "lambda$" + i);
            String intfMethodName = intfMethod.getName();
            DirectMethodHandleDesc lambdaMetafactory = DMHD_LAMBDA_METAFACTORY;
            if (lop.isReflectable()) {
                String modelName = uniqueName(names, "op$lambda$" + i);
                modelsToBuild.put(modelName, Quoted.embedOp(lop));
                lambdaMetafactory = DMHD_REFLECTABLE_LAMBDA_METAFACTORY;
                intfMethodName = intfMethodName + "=" + modelName;
            }
            functions.add(lambdaToFuncOp(implName, lop).transform(this));

            ClassDesc[] captureTypes = captures.stream()
                    .map(Value::type).map(LambdaExpansionTransformer::toClassDesc).toArray(ClassDesc[]::new);
            Op.Result r = block.add(new DynamicFuncCallOp(
                    lop.functionalInterface(),
                    block.context().getValues(captures),
                    implName,
                    lambdaMetafactory,
                    intfMethodName,
                    MethodTypeDesc.of(intfType.toNominalDescriptor(), captureTypes),
                    MethodTypeDesc.of(
                            intfMethod.getReturnType().describeConstable().get(),
                            Stream.of(intfMethod.getParameterTypes()).map(t -> t.describeConstable().get()).toList()),
                    mtd));
            block.context().mapValue(lop.result(), r);
            return block;
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException(e);
        }
    }

    private static ClassDesc toClassDesc(CodeType t) {
        return switch (t) {
            case VarType vt -> toClassDesc(vt.valueType());
            case JavaType jt -> jt.toNominalDescriptor();
            default -> throw new IllegalArgumentException("Bad type: " + t);
        };
    }

    private static Method funcIntfMethod(Class<?> intfc, MethodTypeDesc mtd) {
        Method intfM = null;
        for (Method m : intfc.getMethods()) {
            String methodName = m.getName();
            if (Modifier.isAbstract(m.getModifiers())
                    && (m.getReturnType() != String.class
                    || m.getParameterCount() != 0
                    || !methodName.equals("toString"))
                    && (m.getReturnType() != int.class
                    || m.getParameterCount() != 0
                    || !methodName.equals("hashCode"))
                    && (m.getReturnType() != boolean.class
                    || m.getParameterCount() != 1
                    || m.getParameterTypes()[0] != Object.class
                    || !methodName.equals("equals"))) {
                if (intfM == null) {
                    intfM = m;
                } else if (!intfM.getName().equals(methodName)) {
                    throw new IllegalArgumentException("Not a single-method interface: " + intfc.getName());
                }
            }
        }
        if (intfM == null) {
            throw new IllegalArgumentException("No method in: " + intfc.getName() + " matching: " + mtd);
        }
        return intfM;
    }
}
