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
 * A program element annotated with this annotation enables code reflection in the annotated program element,
 * or in one or more program elements contained within the annotated program element.
 * <p>
 * The program elements for which code reflection is enabled are said to be a <em>reflectable</em> program elements.
 * There are three kinds of reflectable program elements: methods, lambda expressions and method references.
 * Code models for reflectable methods can be obtained using the {@link Op#ofMethod(Method)} method. Code
 * models for reflectable lambdas and method references can be obtained using the {@link Op#ofQuotable(Object)} method.
 * <p>
 * This annotation only has effect on the program elements listed below:
 * <ul>
 * <li>When a method is annotated with this annotation, the method becomes reflectable, and all the lambda expressions
 * and method references enclosed in it also become reflectable.</li>
 * <li>When a variable declaration (a field, or a local variable) is annotated with this annotation, all lambda expressions
 * and method references enclosed in the variable initializer (if present) also become reflectable.</li>
 * <li>When the type of a cast expression is annotated with this annotation, the lambda expression
 * or method reference the cast refers to (if any) becomes reflectable.</li>
 * </ul>
 */
@Target({ElementType.LOCAL_VARIABLE, ElementType.FIELD,
         ElementType.METHOD, ElementType.TYPE_USE})
@Retention(RetentionPolicy.RUNTIME)
public @interface Reflect {
}
