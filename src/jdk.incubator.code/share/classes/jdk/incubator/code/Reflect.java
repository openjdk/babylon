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

package jdk.incubator.code;

import java.lang.annotation.*;
import java.lang.reflect.Method;

/**
 * An annotation that is used to declare reflectable methods, lambda expressions, and method references. Declaration
 * of such reflectable program elements enables access to their code as a code model.
 * <p>
 * The code model of a reflectable method is accessed by invoking {@link Op#ofMethod(Method)} with an argument
 * that is a {@link Method} instance (retrieved using core reflection) representing the reflectable method. The result
 * is an optional value that contains a root operation modeling the method.
 * <p>
 * The code model of a reflectable lambda expression (or method reference) is accessed by invoking
 * {@link Op#ofLambda(Object)} with an argument that is an instance of a functional interface associated with the
 * reflectable lambda expression. The result is an optional value that contains a {@link Quoted quoted} instance, from
 * which may be retrieved the operation modelling the lambda expression. In addition, it is possible to retrieve a
 * mapping of run time values to items in the code model that model final, or effectively final, variables used but not
 * declared in the lambda expression.
 * <p>
 * There are four syntactic locations where {@code @Reflect} can appear that governs, in increasing scope, what is
 * declared reflectable.
 * <ul>
 * <li>
 * If the annotation appears in a cast expression of a lambda expression (or method reference), annotating the use of
 * the type in the cast operator of the cast expression, then the lambda expression is declared reflectable.
 * </li>
 * <li>
 * If the annotation appears as a modifier for a field declaration or a local variable declaration, annotating the
 * field or local variable, then any lambda expressions (or method references) in the variable initializer expression
 * (if present) are declared reflectable. This is useful when cast expressions become verbose and/or types become hard
 * to reason about. For example, with fluent stream-like expressions where many reflectable lambda expressions are
 * passed as arguments.
 * </li>
 * <li>
 * Finally, if the annotation appears as a modifier for a non-abstract method declaration, annotating the method, then
 * the method and any lambda expressions (or method references) it contains are declared reflectable.
 * </li>
 * </ul>
 * The annotation is ignored if it appears in any other valid syntactic location.
 * <p>
 * Declaring a reflectable lambda expression or method does not implicitly broaden the scope of what is reflectable to
 * methods they invoke. Furthermore, declaring a reflectable lambda expression does broaden the scope to the surrounding
 * code of final, or effectively final, variables used but not declared in the lambda expression.
 * Declaring a reflectable method reference does not implicitly broaden the scope to the referenced method.
 * A reflectable method reference's code model is the same as the code model of an equivalent reflectable lambda
 * expression whose body invokes the referenced method.
 */
@Target({ElementType.LOCAL_VARIABLE, ElementType.FIELD, ElementType.METHOD, ElementType.TYPE_USE})
@Retention(RetentionPolicy.RUNTIME)
public @interface Reflect {
}
