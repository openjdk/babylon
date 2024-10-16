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

import hat.ComputeContext;
import hat.buffer.Buffer;
import hat.buffer.KernelContext;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Block;
import java.lang.reflect.code.Value;
import java.lang.reflect.code.op.CoreOp;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.type.MethodRef;
import java.util.Optional;
import java.util.stream.Stream;

// Is this really a root?
public class InvokeOpWrapper extends OpWrapper<CoreOp.InvokeOp> {


    public InvokeOpWrapper(CoreOp.InvokeOp op) {
        super(op);
    }

    public MethodRef methodRef() {
        return op().invokeDescriptor();
    }

    public JavaType javaRefType() {
        return (JavaType) methodRef().refType();
    }

    public boolean isIfaceBufferMethod() {
        return isIface(javaRefType());
    }


    public boolean isRawKernelCall() {
        boolean isRawKernelCall = (operandCount() > 1 && operandNAsValue(0) instanceof Value value
                && value.type() instanceof JavaType javaType
                && (isAssignable(javaType, hat.KernelContext.class) || isAssignable(javaType, hat.buffer.KernelContext.class))
        );
        return isRawKernelCall;
    }

    public boolean isKernelContextMethod() {
        return isAssignable(javaRefType(), KernelContext.class);

    }

    public boolean isComputeContextMethod() {
        return isAssignable(javaRefType(), ComputeContext.class);

    }

    private boolean isReturnTypeAssignableFrom(Class<?> clazz) {
        Optional<Class<?>> optionalClazz = javaReturnClass();
        return optionalClazz.isPresent() && clazz.isAssignableFrom(optionalClazz.get());
    }

    public JavaType javaReturnType() {
        return (JavaType) methodRef().type().returnType();
    }

    public boolean returnsVoid() {
        return javaReturnType().equals(JavaType.VOID);
    }

    public Method methodNoLookup() {
        Class<?> declaringClass = javaRefClass().orElseThrow();
        // TODO this is just matching the name....
        Optional<Method> declaredMethod = Stream.of(declaringClass.getDeclaredMethods())
                .filter(method -> method.getName().equals(methodRef().name()))
                .findFirst();
        if (declaredMethod.isPresent()) {
            return declaredMethod.get();
        }
        Optional<Method> nonDeclaredMethod = Stream.of(declaringClass.getMethods())
                .filter(method -> method.getName().equals(methodRef().name()))
                .findFirst();
        if (nonDeclaredMethod.isPresent()) {
            return nonDeclaredMethod.get();
        } else {
            throw new IllegalStateException("what were we looking for ?"); // getClass causes this
        }
    }
    public Method method(MethodHandles.Lookup lookup) {
        Method invokedMethod = null;
        try {
            invokedMethod = methodRef().resolveToMethod(lookup, op().invokeKind());
            MethodRef methodRef = methodRef();
            return invokedMethod;
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public Value getReceiver() {
        return hasReceiver() ? operandNAsValue(0) : null;
    }

    public boolean hasReceiver() {
        return op().hasReceiver();
    }

    public enum IfaceBufferAccess {None, Access, Mutate}

    public boolean isIfaceAccessor() {
        if (isIfaceBufferMethod() && !returnsVoid()) {
            return !isReturnTypeAssignableFrom(Buffer.class);
        } else {
            return false;
        }
    }


    public boolean isIfaceMutator() {
        return isIfaceBufferMethod() && returnsVoid();
    }

    public IfaceBufferAccess getIfaceBufferAccess() {
        return isIfaceAccessor() ? IfaceBufferAccess.Access : isIfaceMutator() ? IfaceBufferAccess.Mutate : IfaceBufferAccess.None;
    }

    public String name() {
        return op().invokeDescriptor().name();
    }

    public Optional<Class<?>> javaRefClass() {
        try {
            JavaType refType = javaRefType();
            String className = refType.toString();
            Class<?> javaRefClass = Class.forName(className);
            return Optional.of(javaRefClass);
        } catch (ClassNotFoundException e) {
            return Optional.empty();
        }
    }

    public Optional<Class<?>> javaReturnClass() {
        try {
            String className = javaReturnType().toString();
            Class<?> javaRefClass = Class.forName(className);
            return Optional.of(javaRefClass);
        } catch (ClassNotFoundException e) {
            return Optional.empty();
        }
    }

}
