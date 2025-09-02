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

     final public MethodHandles.Lookup lookup;
    public InvokeOpWrapper( MethodHandles.Lookup lookup,JavaOp.InvokeOp op) {
        super(op);
        this.lookup=lookup;
    }
    public static  MethodRef methodRef(JavaOp.InvokeOp op) {
        return op.invokeDescriptor();
    }

    public static JavaType javaRefType(JavaOp.InvokeOp op) {
        return (JavaType) methodRef(op).refType();
    }

    public static  boolean isIfaceBufferMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return  OpTk.isAssignable(lookup,javaRefType(invokeOp), MappableIface.class) ;
    }
  public static boolean isRawKernelCall(MethodHandles.Lookup lookup, JavaOp.InvokeOp op) {
      return (op.operands().size() > 1 && op.operands().getFirst() instanceof Value value
              && value.type() instanceof JavaType javaType
              && (OpTk.isAssignable(lookup,javaType, hat.KernelContext.class) || OpTk.isAssignable(lookup,javaType, KernelContext.class))
      );
  }
    public static boolean isComputeContextMethod(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return OpTk.isAssignable(lookup,javaRefType(invokeOp), ComputeContext.class);
    }

    public static JavaType javaReturnType(JavaOp.InvokeOp op) {
        return (JavaType) methodRef(op).type().returnType();
    }
    public static Method method(MethodHandles.Lookup lookup,JavaOp.InvokeOp op ) {
        try {
            return methodRef(op).resolveToMethod(lookup, op.invokeKind());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        }
    }

    public Method method() {
        return method(lookup,op);
    }


    public enum IfaceBufferAccess {None, Access, Mutate}


    public static boolean isIfaceAccessor(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        if (isIfaceBufferMethod(lookup,invokeOp) && !javaReturnType(invokeOp).equals(JavaType.VOID)) {
            Optional<Class<?>> optionalClazz = javaReturnClass(lookup,invokeOp);
            return optionalClazz.isPresent() && Buffer.class.isAssignableFrom(optionalClazz.get());
        } else {
            return false;
        }
    }

    public static boolean isIfaceMutator(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp) {
        return isIfaceBufferMethod(lookup,invokeOp) && javaReturnType(invokeOp).equals(JavaType.VOID);
    }

    public String name() {
        return op.invokeDescriptor().name();
    }

    public static Optional<Class<?>> javaRefClass(MethodHandles.Lookup lookup,JavaOp.InvokeOp op) {
        if (javaRefType(op) instanceof ClassType classType) {
            return Optional.of((Class<?>) OpTk.classTypeToType(lookup,classType));
        }else{
            return Optional.empty();
        }
    }

    public static Optional<Class<?>> javaReturnClass(MethodHandles.Lookup lookup,JavaOp.InvokeOp op) {
        if (javaReturnType(op) instanceof ClassType classType) {
            return Optional.of((Class<?>) OpTk.classTypeToType(lookup,classType));
        }else{
            return Optional.empty();
        }
    }
}
