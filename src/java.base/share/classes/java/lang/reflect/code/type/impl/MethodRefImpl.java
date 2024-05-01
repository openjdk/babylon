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

package java.lang.reflect.code.type.impl;

import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.type.MethodRef;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandleInfo;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Method;
import java.lang.reflect.code.type.FunctionType;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.CodeType;
import java.util.Optional;

import static java.util.stream.Collectors.joining;

public final class MethodRefImpl implements MethodRef {
    final CodeType refType;
    final String name;
    final FunctionType type;

    public MethodRefImpl(CodeType refType, String name, FunctionType type) {
        this.refType = refType;
        this.name = name;
        this.type = type;
    }

    @Override
    public CodeType refType() {
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

    @Override
    public Method resolveToMember(MethodHandles.Lookup l) throws ReflectiveOperationException {
        // @@@ Constructor
        MethodHandleInfo methodHandleInfo = l.revealDirect(resolveToHandle(l));
        return methodHandleInfo.reflectAs(Method.class, l);
    }

    @Override
    public MethodHandle resolveToHandle(MethodHandles.Lookup l) throws ReflectiveOperationException {
        // @@@ kind
        Class<?> refC = resolve(l, refType);

        MethodType mt = MethodRef.toNominalDescriptor(type).resolveConstantDesc(l);

        MethodHandle mh = null;
        ReflectiveOperationException c = null;

        try {
            mh = l.findStatic(refC, name, mt);
        } catch (NoSuchMethodException | IllegalAccessException e) {
            c = e;
        }

        if (c != null) {
            c = null;
            try {
                mh = l.findVirtual(refC, name, mt);
            } catch (NoSuchMethodException | IllegalAccessException e) {
                c = e;
            }
        }

        if (c != null) {
            throw c;
        }

        assert mh != null;
        return mh;
    }

    static Class<?> resolve(MethodHandles.Lookup l, CodeType t) throws ReflectiveOperationException {
        if (t instanceof JavaType jt) {
            return jt.resolve(l);
        } else {
            // @@@
            throw new ReflectiveOperationException();
        }
    }

    // Copied code in jdk.compiler module throws UOE
    @Override
    public Optional<CoreOps.FuncOp> codeModel(MethodHandles.Lookup l) throws ReflectiveOperationException {
/*__throw new UnsupportedOperationException();__*/        return resolveToMember(l).getCodeModel();
    }

    @Override
    public String toString() {
        return refType + "::" + name +
            type.parameterTypes().stream().map(CodeType::toString)
                    .collect(joining(", ", "(", ")")) + type.returnType();
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