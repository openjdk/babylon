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

package jdk.incubator.code.type.impl;

import jdk.incubator.code.op.CoreOp;
import jdk.incubator.code.type.MethodRef;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Method;
import jdk.incubator.code.type.FunctionType;
import jdk.incubator.code.type.JavaType;
import jdk.incubator.code.TypeElement;
import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.joining;

public final class MethodRefImpl implements MethodRef {
    final TypeElement refType;
    final String name;
    final FunctionType type;

    public MethodRefImpl(TypeElement refType, String name, FunctionType type) {
        this.refType = refType;
        this.name = name;
        this.type = type;
    }

    @Override
    public TypeElement refType() {
        return refType;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public FunctionType type() {
        return type;
    }

    public Method resolveToDirectMethod(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return resolveToDirectHandle(l, hr -> hr.mhi().reflectAs(Method.class, l));
    }

    public MethodHandle resolveToDirectHandle(MethodHandles.Lookup l) throws ReflectiveOperationException {
        return resolveToDirectHandle(l, HandleResult::mh);
    }

    <T> T resolveToDirectHandle(MethodHandles.Lookup l, Function<HandleResult, T> f) throws ReflectiveOperationException {
        ReflectiveOperationException roe = null;
        for (CoreOp.InvokeOp.InvokeKind ik :
                List.of(CoreOp.InvokeOp.InvokeKind.STATIC, CoreOp.InvokeOp.InvokeKind.INSTANCE)) {
            try {
                HandleResult hr = resolveToHandleResult(l, ik);
                if (hr.isDirect()) {
                    return f.apply(hr);
                }
            } catch (NoSuchMethodException | IllegalAccessException e) {
                roe = e;
            }
        }
        if (roe == null) {
            roe = new ReflectiveOperationException("Indirect reference to method");
        }
        throw roe;
    }

    @Override
    public Method resolveToMethod(MethodHandles.Lookup l, CoreOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException {
        MethodHandleInfo methodHandleInfo = l.revealDirect(resolveToHandle(l, kind));
        return methodHandleInfo.reflectAs(Method.class, l);
    }

    @Override
    public MethodHandle resolveToHandle(MethodHandles.Lookup l, CoreOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException {
        Class<?> refC = resolve(l, refType);
        MethodType mt = MethodRef.toNominalDescriptor(type).resolveConstantDesc(l);
        return switch (kind) {
            case SUPER -> l.findSpecial(refC, name, mt, l.lookupClass());
            case STATIC -> l.findStatic(refC, name, mt);
            case INSTANCE -> l.findVirtual(refC, name, mt);
        };
    }

    record HandleResult (Class<?> refC, MethodHandle mh, MethodHandleInfo mhi) {
        boolean isDirect() {
            return refC == mhi.getDeclaringClass();
        }
    }

    HandleResult resolveToHandleResult(MethodHandles.Lookup l, CoreOp.InvokeOp.InvokeKind kind) throws ReflectiveOperationException {
        Class<?> refC = resolve(l, refType);
        MethodType mt = MethodRef.toNominalDescriptor(type).resolveConstantDesc(l);
        MethodHandle mh = switch (kind) {
            case SUPER -> l.findSpecial(refC, name, mt, l.lookupClass());
            case STATIC -> l.findStatic(refC, name, mt);
            case INSTANCE -> l.findVirtual(refC, name, mt);
        };
        MethodHandleInfo mhi = l.revealDirect(resolveToHandle(l, kind));
        return new HandleResult(refC, mh, mhi);
    }

    static Class<?> resolve(MethodHandles.Lookup l, TypeElement t) throws ReflectiveOperationException {
        if (t instanceof JavaType jt) {
            return (Class<?>)jt.erasure().resolve(l);
        } else {
            // @@@
            throw new ReflectiveOperationException();
        }
    }

    @Override
    public String toString() {
        return refType.externalize() + "::" + name +
            type.parameterTypes().stream().map(t -> t.externalize().toString())
                    .collect(joining(", ", "(", ")")) + type.returnType().externalize();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        MethodRefImpl that = (MethodRefImpl) o;

        if (!refType.equals(that.refType)) return false;
        if (!name.equals(that.name)) return false;
        return type.equals(that.type);
    }

    @Override
    public int hashCode() {
        int result = refType.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + type.hashCode();
        return result;
    }
}