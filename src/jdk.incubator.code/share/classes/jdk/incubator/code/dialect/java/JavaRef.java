/*
 * Copyright (c) 2025, 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.dialect.java;

import jdk.incubator.code.CodeType;

/**
 * A symbolic reference to a Java class member or a Java type including members,
 * commonly containing symbolic names together with {@link JavaType symbolic descriptions}
 * of Java types.
 * <p>
 * A symbolic Java reference can be resolved to a corresponding instance of its
 * reflected representation, much like the symbolic description of a Java type
 * can be resolved to an instance of {@link java.lang.reflect.Type Type}.
 */
public sealed interface JavaRef extends CodeType
        permits MethodRef, FieldRef, RecordTypeRef {
    // @@@ Make RecordTypeRef.ComponentRef implement JavaRef?
    //     - resolve to RecordComponent
    //     - (RecordTypeRef resolves to Type.)
    // @@@ AnnotatedElement is the common top type for resolved Java refs and types
}
