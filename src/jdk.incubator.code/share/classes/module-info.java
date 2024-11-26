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

/**
 * A module which provides classes and interfaces for obtaining reflective information about
 * classes and objects.
 * {@incubating}
 *
 * @moduleGraph
 */

import jdk.incubator.code.internal.ReflectMethods;
import jdk.internal.javac.ParticipatesInPreview;

@ParticipatesInPreview
module jdk.incubator.code {
    requires transitive jdk.compiler;

    exports jdk.incubator.code;
    exports jdk.incubator.code.parser;
    exports jdk.incubator.code.op;
    exports jdk.incubator.code.type;
    exports jdk.incubator.code.analysis;
    exports jdk.incubator.code.bytecode;
    exports jdk.incubator.code.interpreter;
    exports jdk.incubator.code.writer;
    exports jdk.incubator.code.tools.dot;
    exports jdk.incubator.code.tools.renderer;

    opens jdk.incubator.code.internal to java.base;

    provides com.sun.tools.javac.comp.CodeReflectionTransformer with
            ReflectMethods.Provider;
}
