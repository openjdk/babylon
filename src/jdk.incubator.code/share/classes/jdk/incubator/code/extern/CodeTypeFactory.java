/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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
package jdk.incubator.code.extern;

import jdk.incubator.code.CodeType;

/**
 * A code type factory for construction a {@link CodeType} from its
 * {@link ExternalizedCodeType external content}.
 */
@FunctionalInterface
public interface CodeTypeFactory {

    /**
     * Constructs a {@link CodeType} from its
     * {@link ExternalizedCodeType external content}.
     * <p>
     * If there is no mapping from the external content to a code type
     * then this method returns {@code null}.
     *
     * @param tree the externalized code type.
     * @return the code type.
     */
    CodeType constructType(ExternalizedCodeType tree);

    /**
     * Compose this code type factory with another code type factory.
     * <p>
     * If there is no mapping in this code type factory then the result
     * of the other code type factory is returned.
     *
     * @param after the other code type factory.
     * @return the composed code type factory.
     */
    default CodeTypeFactory andThen(CodeTypeFactory after) {
        return t -> {
            CodeType te = constructType(t);
            return te != null ? te : after.constructType(t);
        };
    }
}
