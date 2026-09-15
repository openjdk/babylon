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
import java.lang.invoke.MethodHandles.Lookup;
import java.lang.invoke.MethodType;
import java.lang.reflect.Member;
import java.lang.reflect.Method;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;
import jdk.incubator.code.Op;
import jdk.incubator.code.bytecode.BytecodeGenerator;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.core.CoreType;
import jdk.incubator.code.dialect.java.JavaOp;

/**
 * Bootstrap methods for linking {@code invokedynamic} call sites that execute
 * method code models or create lambda instances implemented by code models.
 *
 * @see ReflectableLambdaMetafactory
 */
public final class CodeModelBootstraps {

    private CodeModelBootstraps() {
    }

    /**
     * Bootstrap method for linking an {@code invokedynamic} call site that
     * implements execution of a method's code model.
     * <p>
     * The method's code model is obtained for the method referenced by the
     * given method handle. If the method does not have a code model then an
     * {@code IllegalArgumentException} is thrown.
     * <p>
     * Execution of the code model is implemented by transforming the code model
     * to bytecode and linking it as the target method handle of the returned
     * {@code CallSite}.
     * <p>
     * If model retrieval or transformation fails, the resulting exception or
     * error is propagated.
     *
     * @param lookup   Represents a lookup context with the accessibility
     *                 privileges of the caller. Specifically, the lookup
     *                 context must have
     *                 {@linkplain MethodHandles.Lookup#hasFullPrivilegeAccess()
     *                 full privilege access}.
     *                 When used with {@code invokedynamic}, this is stacked
     *                 automatically by the VM.
     * @param name     The name of the method to implement. This name is
     *                 arbitrary, and has no meaning for this linkage method.
     *                 When used with {@code invokedynamic}, this is provided by
     *                 the {@code NameAndType} of the {@code InvokeDynamic}
     *                 structure and is stacked automatically by the VM.
     * @param methodType The expected signature of the {@code CallSite}. When
     *                   used with {@code invokedynamic}, this is provided by the
     *                   {@code NameAndType} of the {@code InvokeDynamic}
     *                   structure and is stacked automatically by the VM.
     * @param method a direct method handle referencing the original method
     * @return a constant call site whose target implements the behavior
     *         represented by the stored code model
     * @throws NullPointerException If any of the incoming arguments is null.
     *                              This will never happen when a bootstrap method
     *                              is called with {@code invokedynamic}.
     * @throws IllegalArgumentException if the handle cannot be revealed by
     *         {@code lookup}, does not reference an accessible method, the
     *         method has no code model, or the generated target type differs
     *         from {@code methodType}
     */
    public static CallSite linkMethod(MethodHandles.Lookup lookup,
                                      String name,
                                      MethodType methodType,
                                      MethodHandle method) {
        Objects.requireNonNull(name);
        Objects.requireNonNull(methodType);
        Member member = lookup.revealDirect(method).reflectAs(Member.class, lookup);
        if (!(member instanceof Method m)) {
            throw new IllegalArgumentException("Handle does not reference a method");
        }
        CoreOp.FuncOp model = Op.ofMethod(m).orElseThrow(() ->
                new IllegalArgumentException("Method has no code model: " + m));
        MethodHandle target = BytecodeGenerator.generate(lookup, model);
        if (!target.type().equals(methodType)) {
            throw new IllegalArgumentException("Code model target type " + target.type()
                    + " differs from call-site type " + methodType);
        }
        return new ConstantCallSite(target);
    }

    /**
     * Bootstrap method for linking an {@code invokedynamic} call site whose
     * target creates reflectable lambda instances, using the lambda's stored
     * code model as its implementation.
     * <p>
     * Except for the implementation method handle, this method follows the
     * contract and encoded-name convention of
     * {@link ReflectableLambdaMetafactory#metafactory(Lookup, String,
     *        MethodType, MethodType, MethodHandle, MethodType)}.
     * <p>
     * The supplied implementation method handle is ignored and replaced
     * with one linked from the stored code model.
     *
     * @param caller the lookup
     * @param interfaceMethodName the encoded interface-method and code-model
     *                            accessor names, as specified by
     *                            {@link ReflectableLambdaMetafactory#metafactory(Lookup,
     *                                   String, MethodType, MethodType, MethodHandle,
     *                                   MethodType)}
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param interfaceMethodType Signature and return type of method to be
     *                            implemented by the function object.
     * @param implementation ignored, retained for compatibility with the
     *                       standard metafactory bootstrap signature
     * @param dynamicMethodType The signature and return type that should
     *                          be enforced dynamically at invocation time.
     * @return a call site whose target creates reflectable lambda instances of
     *         the functional interface specified by the return type of
     *         {@code factoryType}; each instance can be inspected using
     *         {@link Op#ofLambda(Object)}
     *
     * @throws LambdaConversionException if the implementation cannot be linked
     *         from its stored code model, or if, after the lambda name is
     *         decoded, the parameters of the call are invalid for
     *         {@link ReflectableLambdaMetafactory#metafactory(Lookup, String,
     *                MethodType, MethodType, MethodHandle, MethodType)}
     * @throws NullPointerException if {@code interfaceMethodName},
     *         {@code factoryType}, {@code interfaceMethodType}, or
     *         {@code dynamicMethodType} is {@code null}
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
        MethodHandle generatedImpl = linkLambdaImplementation(caller, interfaceMethodName);
        CallSite site = ReflectableLambdaMetafactory.metafactory(caller,
                                                                 interfaceMethodName,
                                                                 generatedFactoryType(factoryType, generatedImpl),
                                                                 interfaceMethodType,
                                                                 generatedImpl,
                                                                 dynamicMethodType);
        return new ConstantCallSite(site.getTarget().asType(factoryType));
    }

    /**
     * Bootstrap method for linking an {@code invokedynamic} call site whose
     * target creates reflectable lambda instances, using the lambda's stored
     * code model as its implementation.
     * <p>
     * Except for the implementation method handle, this method follows the
     * contract and encoded-name convention of
     * {@link ReflectableLambdaMetafactory#altMetafactory(Lookup, String,
     *        MethodType, Object...)}.
     * <p>
     * The implementation method handle in {@code args} is replaced with one
     * linked from the stored code model.
     *
     * @param caller the lookup
     * @param interfaceMethodName the encoded interface-method and code-model
     *                            accessor names, as specified by
     *                            {@link ReflectableLambdaMetafactory#altMetafactory(Lookup,
     *                                   String, MethodType, Object...)}
     * @param factoryType The expected signature of the {@code CallSite}.
     * @param args An array of {@code Object} containing the required
     *              arguments {@code interfaceMethodType}, {@code implementation},
     *              {@code dynamicMethodType}, {@code flags}, and any
     *              optional arguments, as required by
     *              {@link ReflectableLambdaMetafactory#altMetafactory(Lookup,
     *                     String, MethodType, Object...)}, the
     *              {@code implementation} component is ignored and replaced
     *              with one linked from the stored code model
     * @return a CallSite whose target can be used to perform capture, generating
     *         a reflectable lambda instance implementing the functional
     *         interface specified by the return type of {@code factoryType}.
     *         The code model for such instance can be inspected using
     *         {@link Op#ofLambda(Object)}.
     *
     * @throws LambdaConversionException if the implementation cannot be linked
     *         from its stored code model, or if, after the lambda name is
     *         decoded, the parameters of the call are invalid for
     *         {@link ReflectableLambdaMetafactory#altMetafactory(Lookup, String,
     *                MethodType, Object...)}
     * @throws NullPointerException if {@code interfaceMethodName},
     *         {@code factoryType}, or {@code args} is {@code null}, or if a
     *         required component of {@code args} other than
     *         {@code implementation} is {@code null}
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
        MethodHandle generatedImpl = linkLambdaImplementation(caller, interfaceMethodName);
        args[1] = generatedImpl;
        CallSite site = ReflectableLambdaMetafactory.altMetafactory(caller,
                                                                    interfaceMethodName,
                                                                    generatedFactoryType(factoryType, generatedImpl),
                                                                    args);
        return new ConstantCallSite(site.getTarget().asType(factoryType));
    }

    private static MethodType generatedFactoryType(MethodType factoryType, MethodHandle implementation) {
        MethodType implementationType = implementation.type();
        return implementationType.dropParameterTypes(factoryType.parameterCount(), implementationType.parameterCount())
                                 .changeReturnType(factoryType.returnType());
    }

    private static MethodHandle linkLambdaImplementation(MethodHandles.Lookup caller,
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
            bb.context().mapValues(funcOp.parameters(), bb.parameters().subList(0, capturedValues));
            for (int i = 0; i < ops.size() - 2; i++) {
                Op o = ops.get(i);
                bb.add(o);
            }
            bb.transformBody(lambda.body(),
                             bb.parameters().subList(capturedValues, bb.parameters().size()));
        });
    }
}
