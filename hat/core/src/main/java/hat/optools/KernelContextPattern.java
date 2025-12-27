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

import hat.KernelContext;
import jdk.incubator.code.CodeElement;
import jdk.incubator.code.TypeElement;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.Regex;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

import static optkl.OpTkl.AnyFieldAccess;
import static optkl.OpTkl.isAssignable;

public interface KernelContextPattern extends CodeModelPattern {

    static boolean isKernelContext(MethodHandles.Lookup lookup, TypeElement typeElement) {
        return isAssignable(lookup, typeElement, KernelContext.class);
    }


    interface KernelContextFieldAccessPattern extends KernelContextPattern {

        static JavaOp.FieldAccessOp asKernelContextFieldAccessOrNull(MethodHandles.Lookup lookup, CodeElement<?, ?> ce, Predicate<JavaOp.FieldAccessOp> predicate) {
            if (ce instanceof JavaOp.FieldAccessOp fieldAccessOp
                    && KernelContextPattern.isKernelContext(lookup, fieldAccessOp.fieldDescriptor().refType())) {
                return predicate.test(fieldAccessOp) ? fieldAccessOp : null;
            }
            return null;
        }

        static KernelContextFieldAccessPattern matches(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement, Predicate<JavaOp.FieldAccessOp> fieldAccessOpPredicate) {
            if (codeElement instanceof JavaOp.FieldAccessOp fieldAccessOp) {
                if (KernelContextPattern.isKernelContext(lookup, fieldAccessOp.fieldDescriptor().refType()) && fieldAccessOpPredicate.test(fieldAccessOp)) {
                    try {
                        Field field = fieldAccessOp.fieldDescriptor().resolveToField(lookup);
                        record KernelContextFieldAccessPatternImpl(Set<CodeElement<?, ?>> codeElements,
                                                                   JavaOp.FieldAccessOp fieldAccessOp,
                                                                   Field field, String fieldName,
                                                                   TypeElement typeElement) implements KernelContextFieldAccessPattern {
                        }
                        return new KernelContextFieldAccessPatternImpl(Set.of(fieldAccessOp), fieldAccessOp, field, fieldAccessOp.fieldDescriptor().name(), fieldAccessOp.fieldDescriptor().refType());
                    } catch (ReflectiveOperationException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            return null;
        }

    }

    interface KernelContextInvokePattern extends KernelContextPattern {

        static JavaOp.InvokeOp asKernelContextInvokeOpOrNull(MethodHandles.Lookup lookup, CodeElement<?, ?> ce, Predicate<JavaOp.InvokeOp> predicate) {
            return ce instanceof JavaOp.InvokeOp invokeOp
                    && KernelContextPattern.isKernelContext(lookup, invokeOp.invokeDescriptor().refType())
                    && predicate.test(invokeOp)
                    ? invokeOp
                    : null;
        }

        static boolean isKernelContextInvokeOp(MethodHandles.Lookup lookup, CodeElement<?, ?> ce, Predicate<JavaOp.InvokeOp> predicate) {
            return Objects.nonNull(asKernelContextInvokeOpOrNull(lookup, ce, predicate));
        }


        static KernelContextInvokePattern matches(MethodHandles.Lookup lookup, CodeElement<?, ?> codeElement, Predicate<JavaOp.InvokeOp> invokeOpPredicate) {
            record KernelContextInvokePatternImpl(
                    Set<CodeElement<?, ?>> codeElements,
                    JavaOp.InvokeOp invokeOp,
                    Method method,
                    String methodName,
                    TypeElement typeElement) implements KernelContextInvokePattern {
            }

            if (codeElement instanceof JavaOp.InvokeOp invokeOp) {
                if (KernelContextPattern.isKernelContext(lookup, invokeOp.invokeDescriptor().refType()) && invokeOpPredicate.test(invokeOp)) {
                    try {
                        Method method = invokeOp.invokeDescriptor().resolveToMethod(lookup);
                        return new KernelContextInvokePatternImpl(Set.of(invokeOp), invokeOp, method, invokeOp.invokeDescriptor().name(), invokeOp.invokeDescriptor().refType());
                    } catch (ReflectiveOperationException e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            return null;
        }
    }
}
