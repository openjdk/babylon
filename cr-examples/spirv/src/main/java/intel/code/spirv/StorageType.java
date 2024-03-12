/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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

package intel.code.spirv;

import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.TypeDefinition;
import java.util.List;
import java.util.Objects;

public sealed abstract class StorageType extends SpirvType
    permits StorageType.Input, StorageType.Workgroup, StorageType.CrossWorkgroup, StorageType.Private, StorageType.Function {

    public static final Input INPUT = new Input();
    public static final Workgroup WORKGROUP = new Workgroup();
    public static final CrossWorkgroup CROSSWORKGROUP= new CrossWorkgroup();
    public static final Private PRIVATE = new Private();
    public static final Function FUNCTION = new Function();

    protected final String NAME;

    protected StorageType(String name) {
        this.NAME = name;
    }

    static final class Input extends StorageType {
        protected Input(){
            super("Input");
        }
    }

    static final class Workgroup extends StorageType {
        protected Workgroup() {
            super("Workgroup");
        }
    }

    static final class CrossWorkgroup extends StorageType
    {
        protected CrossWorkgroup() {
            super("CrossWorkgroup");
        }
    }

    static final class Private extends StorageType
    {
        protected Private() {
            super("Private");
        }
    }

    static final class Function extends StorageType
    {
        protected Function() {
            super("Function");
        }
    }

    @Override
    public boolean equals(Object obj) {
        return obj != null && obj.getClass() != this.getClass();
    }

    @Override
    public int hashCode() {
        return Objects.hash(NAME);
    }

    @Override
    public TypeDefinition toTypeDefinition() {
        return new TypeDefinition(NAME, List.of());
    }

    @Override
    public String toString() {
        return toTypeDefinition().toString();
    }
}