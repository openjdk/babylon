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


import hat.Accelerator;
import hat.optools.OpTk;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

public interface HATDialect  extends Function<CoreOp.FuncOp,CoreOp.FuncOp> {
    Accelerator accelerator();

    default boolean isMethodFromHatKernelContext(JavaOp.InvokeOp invokeOp) {
        String kernelContextCanonicalName = hat.KernelContext.class.getName();// URRH Strings
        return invokeOp.invokeDescriptor().refType().toString().equals(kernelContextCanonicalName);
    }

    default boolean isMethod(JavaOp.InvokeOp invokeOp, String methodName) {
        return invokeOp.invokeDescriptor().name().equals(methodName);
    }

    default boolean isIfaceBufferInvokeWithName(JavaOp.InvokeOp invokeOp, String methodName) {
        return OpTk.isIfaceBufferMethod(accelerator().lookup, invokeOp) && isMethod(invokeOp, methodName);
    }

    default boolean isKernelContextInvokeWithName(Op op, String methodName) {
        return op instanceof JavaOp.InvokeOp invokeOp
                && isMethodFromHatKernelContext(invokeOp)
                && isMethod(invokeOp,methodName);
    }

    default void before(OpTk.CallSite callSite, CoreOp.FuncOp funcOp){
        if (accelerator().backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model before " + callSite.clazz().getSimpleName()+": " + funcOp.toText());
        }
    }
    default void after(OpTk.CallSite callSite, CoreOp.FuncOp funcOp){
        if (accelerator().backend.config().showCompilationPhases()) {
            IO.println("[INFO] Code model after " + callSite.clazz().getSimpleName()+": " + funcOp.toText());
        }
    }

    default Set<Class<?>> inspectAllInterfaces(Class<?> klass) {
        Set<Class<?>> interfaceSet = new HashSet<>();
        while (klass != null) {
            Arrays.stream(klass.getInterfaces())
                    .forEach(interfaceClass -> inspectNewLevel(interfaceClass, interfaceSet));
            klass = klass.getSuperclass();
        }
        return interfaceSet;
    }

    default void inspectNewLevel(Class<?> interfaceClass, Set<Class<?>> interfaceSet) {
        if (interfaceClass != null && interfaceSet.add(interfaceClass)) {
            // only if we add a new interface class, we inspect all interfaces that extends the current inspected class
            Arrays.stream(interfaceClass.getInterfaces())
                    .forEach(superInterface -> inspectNewLevel(superInterface, interfaceSet));
        }
    }
}
