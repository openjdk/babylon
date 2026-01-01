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

import hat.types._V;
import jdk.incubator.code.Op;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.Value;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.OpTkl;

import java.lang.invoke.MethodHandles;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Predicate;

import static optkl.Invoke.invokeOpHelper;
import static optkl.OpTkl.isAssignable;

/**
 * This class needs refactoring
 *
 * It seems to be a reimplementation of Optkl isAssignable
 */
public class RefactorMe {
    private static void inspectNewLevel(Class<?> interfaceClass, Set<Class<?>> interfaceSet) {
        if (interfaceClass != null && interfaceSet.add(interfaceClass)) {
            // only if we add a new interface class, we inspect all interfaces that extends the current inspected class
            Arrays.stream(interfaceClass.getInterfaces())
                    .forEach(superInterface -> inspectNewLevel(superInterface, interfaceSet));
        }
    }


    public static Set<Class<?>> inspectAllInterfaces(Class<?> klass) {
        Set<Class<?>> interfaceSet = new HashSet<>();
        while (klass != null) {
            Arrays.stream(klass.getInterfaces())
                    .forEach(interfaceClass -> RefactorMe.inspectNewLevel(interfaceClass, interfaceSet));
            klass = klass.getSuperclass();
        }
        return interfaceSet;
    }


    public  static boolean  isVectorOperation(MethodHandles.Lookup lookup,JavaOp.InvokeOp invokeOp, Value varValue, Predicate<String> namePredicate) {
        if (OpTkl.asResultOrNull(varValue) instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            TypeElement typeElement = varLoadOp.resultType();
            Set<Class<?>> interfaces = Set.of();
            try {
                Class<?> aClass = Class.forName(typeElement.toString());
                interfaces = inspectAllInterfaces(aClass);
            } catch (ClassNotFoundException _) {
            }
            return interfaces.contains(_V.class) && invokeOpHelper(lookup, invokeOp).named( namePredicate);
        }
        return false;
    }
    public static boolean isAMethod(JavaOp.InvokeOp invokeOp, Predicate<String> namePredicate) {
        return namePredicate.test(invokeOp.invokeDescriptor().name());
    }
    public  static boolean  isVectorOperation(JavaOp.InvokeOp invokeOp, Value varValue, Predicate<String> namePredicate) {
        if (OpTkl.asResultOrNull(varValue) instanceof Op.Result r && r.op() instanceof CoreOp.VarAccessOp.VarLoadOp varLoadOp) {
            TypeElement typeElement = varLoadOp.resultType();
            Set<Class<?>> interfaces = Set.of();
            try {
                Class<?> aClass = Class.forName(typeElement.toString());
                interfaces = inspectAllInterfaces(aClass);
            } catch (ClassNotFoundException _) {
            }
            return interfaces.contains(_V.class) && isAMethod(invokeOp, namePredicate);
        }
        return false;
    }
    public static boolean isInvokeDescriptorSubtypeOf(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, Class<?> klass) {

        var wouldReturn = (invokeOp.resultType() instanceof JavaType jt && isAssignable(lookup, jt, klass));

        TypeElement typeElement = invokeOp.invokeDescriptor().refType();
        Set<Class<?>> interfaces = Set.of();
        try {
            Class<?> aClass = Class.forName(typeElement.toString());
            interfaces = inspectAllInterfaces(aClass);
        } catch (ClassNotFoundException _) {
        }
        var butReturns = interfaces.contains(klass);
        if (butReturns != wouldReturn) {
            // System.out.print("isInvokeDescriptorSubtypeOf");
        }
        return butReturns;

    }

    public static boolean isInvokeDescriptorSubtypeOfAnyMatch(MethodHandles.Lookup lookup, JavaOp.InvokeOp invokeOp, Class<?>... klasses) {

        boolean wouldReturn = (invokeOp.resultType() instanceof JavaType jt && isAssignable(lookup, jt, klasses));
        boolean butReturns = false;
        TypeElement typeElement = invokeOp.invokeDescriptor().refType();
        Set<Class<?>> interfaces = Set.of();
        try {
            Class<?> aClass = Class.forName(typeElement.toString());
            interfaces = inspectAllInterfaces(aClass);
        } catch (ClassNotFoundException _) {
        }
        for (Class<?> klass : klasses) {
            if (interfaces.contains(klass)) {
                butReturns = true;
            }
        }
        if (butReturns != wouldReturn) {
            //   System.out.print("isInvokeDescriptorSubtypeOfAnyMatch");
        }
        return butReturns;
    }


}
