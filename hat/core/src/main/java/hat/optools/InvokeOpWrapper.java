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

import hat.ifacemapper.MappableIface;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.java.ClassType;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.MethodRef;

import java.util.Optional;

public class InvokeOpWrapper extends OpWrapper<JavaOp.InvokeOp> {


    public InvokeOpWrapper( MethodHandles.Lookup lookup,JavaOp.InvokeOp op) {
        super(lookup,op);
    }

    public MethodRef methodRef() {
        return op.invokeDescriptor();
    }

    public JavaType javaRefType() {
        return (JavaType) methodRef().refType();
    }

    public boolean isIfaceBufferMethod() {
        return  isAssignable(lookup,javaRefType(), MappableIface.class) ;
    }

    public boolean isRawKernelCall() {
        return (op.operands().size() > 1 && op.operands().getFirst() instanceof Value value
                && value.type() instanceof JavaType javaType
                && (isAssignable(lookup,javaType, hat.KernelContext.class) || isAssignable(lookup,javaType, KernelContext.class))
        );
    }

    public boolean isComputeContextMethod() {
        return isAssignable(lookup,javaRefType(), ComputeContext.class);
    }
    public JavaType javaReturnType() {
        return (JavaType) methodRef().type().returnType();
    }

    public Method method() {
        try {
            return methodRef().resolveToMethod(lookup, op.invokeKind());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public Value getReceiver() {
        return op.hasReceiver() ? op.operands().getFirst() : null;
    }

    public enum IfaceBufferAccess {None, Access, Mutate}

    public boolean isIfaceAccessor() {
        if (isIfaceBufferMethod() && !javaReturnType().equals(JavaType.VOID)) {
            Optional<Class<?>> optionalClazz = javaReturnClass();
            return optionalClazz.isPresent() && Buffer.class.isAssignableFrom(optionalClazz.get());
        } else {
            return false;
        }
    }


    public boolean isIfaceMutator() {
        return isIfaceBufferMethod() && javaReturnType().equals(JavaType.VOID);
    }

    public IfaceBufferAccess getIfaceBufferAccess() {
        return isIfaceAccessor() ? IfaceBufferAccess.Access : isIfaceMutator() ? IfaceBufferAccess.Mutate : IfaceBufferAccess.None;
    }

    public String name() {
        return op.invokeDescriptor().name();
    }

    public Optional<Class<?>> javaRefClass() {
        if (javaRefType() instanceof ClassType classType) {
            return Optional.of((Class<?>) classTypeToType(lookup,classType));
        }else{
            return Optional.empty();
        }
    }

    public Optional<Class<?>> javaReturnClass() {
        if (javaReturnType() instanceof ClassType classType) {
            return Optional.of((Class<?>) classTypeToType(lookup,classType));
        }else{
            return Optional.empty();
        }
    }

}
