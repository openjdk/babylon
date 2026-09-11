/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

package jdk.incubator.code.runtime;

import java.lang.invoke.CallSite;
import java.lang.invoke.ConstantCallSite;
import java.lang.invoke.LambdaConversionException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import jdk.incubator.code.CodeTransformer;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;

/**
 * Provides runtime support for creating methods and lambdas from stored code models.
 * @see ReflectableLambdaMetafactory
 */
public final class UnreflectMetafactory {

    private UnreflectMetafactory() {
    }

    /**
     * Creates a constant call site whose target is generated from the stored
     * code model of a matching method declared by the lookup class.
     *
     * @param caller the call-site lookup.
     * @param methodName the source method name.
     * @param methodType the call-site type.
     * @return a constant call site for the generated implementation.
     * @throws NoSuchMethodException if the source method cannot be identified.
     */
    public static CallSite unreflect(MethodHandles.Lookup caller,
                                     String methodName,
                                     MethodType methodType) throws NoSuchMethodException {
        String className = caller.lookupClass().getName();
        for (Method m : caller.lookupClass().getDeclaredMethods()) {
            boolean isStatic = Modifier.isStatic(m.getModifiers());
            int firstParam = isStatic ? 0 : 1;
            if (m.getName().equals(methodName)
                    && m.getReturnType() == methodType.returnType()
                    && m.getParameterCount() == methodType.parameterCount() - firstParam
                    && (isStatic || methodType.parameterType(0) == caller.lookupClass())
                    && Arrays.equals(m.getParameterTypes(), 0, m.getParameterCount(),
                                     methodType.parameterArray(), firstParam, methodType.parameterCount())) {
                return new ConstantCallSite(BytecodeGenerator.generate(caller, Op.ofMethod(m).orElseThrow()));
            }
        }
        throw new NoSuchMethodException(className + "." + methodName + methodType);
    }

    /**
     * Metafactory used to create a reflectable lambda with implementation
     * generated from its stored code model.
     * <p>
     * The functionality provided by this metafactory is identical to that in
     * {@link ReflectableLambdaMetafactory#metafactory(Lookup, String,
     *        MethodType, MethodType, MethodHandle, MethodType)}
     * with one important difference: this metafactory generates the lambda
     * implementation method from the stored code model.
     *
     * @param caller The lookup
     * @param interfaceMethodName The name of the method to implement.
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param interfaceMethodType Signature and return type of method to be
     *                            implemented by the function object.
     * @param implementation Ignored, retained for compatibility with the
     *                       standard metafactory bootstrap signature
     * @param dynamicMethodType The signature and return type that should
     *                          be enforced dynamically at invocation time.
     * @return a call site whose target creates reflectable lambda instances of
     *         the functional interface specified by the return type of
     *         {@code factoryType}; each instance can be inspected using
     *         {@link Op#ofLambda(Object)}
     *
     * @throws LambdaConversionException If, after the lambda name is decoded,
     *         the parameters of the call are invalid for
     *         {@link ReflectableLambdaMetafactory#metafactory(Lookup, String,
     *                MethodType, MethodType, MethodHandle, MethodType)}
     * @throws NullPointerException If any argument is {@code null}.
     *
     * @see ReflectableLambdaMetafactory#metafactory(Lookup, String, MethodType,
     *      MethodType, MethodHandle, MethodType)
     * @see Op#ofLambda(Object)
     */
    public static CallSite metafactory(MethodHandles.Lookup caller,
                                       String interfaceMethodName,
                                       MethodType factoryType,
                                       MethodType interfaceMethodType,
                                       MethodHandle implementation,
                                       MethodType dynamicMethodType) throws LambdaConversionException {
        return ReflectableLambdaMetafactory.metafactory(caller,
                                                        interfaceMethodName,
                                                        factoryType,
                                                        interfaceMethodType,
                                                        unreflectLambdaImplementation(caller, interfaceMethodName),
                                                        dynamicMethodType);
    }

    /**
     * Metafactory used to create a reflectable lambda with implementation
     * generated from its stored code model.
     * <p>
     * The functionality provided by this metafactory is identical to that in
     * {@link ReflectableLambdaMetafactory#altMetafactory(Lookup, String,
     *        MethodType, Object...)}
     * with one important difference: this metafactory generates the lambda
     * implementation method from the stored code model.
     *
     * @param caller The lookup
     * @param interfaceMethodName The name of the method to implement.
     *                            This is encoded in the format described above.
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param args An array of {@code Object} containing the required
     *              arguments {@code interfaceMethodType}, {@code implementation},
     *              {@code dynamicMethodType}, {@code flags}, and any
     *              optional arguments, as required by
     *              {@link ReflectableLambdaMetafactory#altMetafactory(Lookup,
     *                     String, MethodType, Object...)}
     * @return a CallSite whose target can be used to perform capture, generating
     *         a reflectable lambda instance implementing the interface named by
     *         {@code factoryType}. The code model for such instance can be
     *         inspected using {@link Op#ofLambda(Object)}.
     *
     * @throws LambdaConversionException If, after the lambda name is decoded,
     *         the parameters of the call are invalid for
     *         {@link ReflectableLambdaMetafactory#altMetafactory(Lookup, String,
     *                MethodType, Object...)}
     * @throws NullPointerException If any argument, or any component of {@code args},
     *         is {@code null}.
     * @throws IllegalArgumentException If {@code args} are invalid for
     *         {@link ReflectableLambdaMetafactory#altMetafactory(Lookup, String,
     *                MethodType, Object...)}
     *
     * @see ReflectableLambdaMetafactory#altMetafactory(Lookup, String, MethodType,
     *      Object...)
     * @see Op#ofLambda(Object)
     */
    public static CallSite altMetafactory(MethodHandles.Lookup caller,
                                          String interfaceMethodName,
                                          MethodType factoryType,
                                          Object... args) throws LambdaConversionException {
        args[1] = unreflectLambdaImplementation(caller, interfaceMethodName);
        return ReflectableLambdaMetafactory.altMetafactory(caller,
                                                           interfaceMethodName,
                                                           factoryType,
                                                           args);
    }

    private static MethodHandle unreflectLambdaImplementation(MethodHandles.Lookup caller,
                                                              String interfaceMethodName)
            throws LambdaConversionException {
        String modelMethodName = interfaceMethodName.split("=")[1];
        try {
            MethodHandle opHandle = caller.findStatic(caller.lookupClass(),
                                                      modelMethodName,
                                                      MethodType.methodType(Op.class));
            MethodHandle methodHandle = BytecodeGenerator.generate(caller,
                                                                   unquoteLambda((CoreOp.FuncOp)opHandle.invoke()));
            return methodHandle;
        } catch (Throwable t) {
            throw new LambdaConversionException(t);
        }
    }

    // flatten the quoted lambda into the enclosing function model
    private static CoreOp.FuncOp unquoteLambda(CoreOp.FuncOp funcOp) {
        int capturedValues = funcOp.parameters().size();
        List<Op> ops = funcOp.body().entryBlock().ops();
        JavaOp.LambdaOp lambda = (JavaOp.LambdaOp)((CoreOp.QuotedOp)ops.get(ops.size() - 2)).quotedOp();
        return CoreOp.func(funcOp.funcName(), CoreType.functionType(
                lambda.body().yieldType(),
                Stream.of(funcOp.invokableSignature().parameterTypes(),
                          lambda.invokableSignature().parameterTypes()).flatMap(List::stream).toList())).body(bb -> {
            bb.context().mapBlock(funcOp.body().entryBlock(), bb.entryBlock());
            bb.context().mapValues(funcOp.parameters(), bb.parameters().subList(0, capturedValues));
            for (int i = 0; i < ops.size() - 2; i++) {
                Op o = ops.get(i);
                bb.add(o);
            }
            bb.transformBody(lambda.body(),
                             bb.parameters().subList(capturedValues, bb.parameters().size()),
                             bb.context(),
                             CodeTransformer.COPYING_TRANSFORMER);
        });
    }
}
