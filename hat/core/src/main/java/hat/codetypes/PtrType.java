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
package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.Objects;

/**
 * Type to represent input/output tensors (mutable data) that will be copied
 * between host <-> device to/from the accelerator's global memory.
 */
public final class PtrType implements TileType {

    private final CodeType rType;
    private final List<Object> dims;

    public PtrType(CodeType rType) {
        this.rType = rType;
        this.dims = List.of();
    }

    public PtrType(CodeType rType, Object dim1) {
        this.rType = rType;
        this.dims = List.of(dim1);
    }

    public PtrType(CodeType rType, Object dim1, Object dim2) {
        this.rType = rType;
        this.dims = List.of(dim1, dim2);
    }

    public PtrType(CodeType rType, Object dim1, Object dim2, Object dim3) {
        this.rType = rType;
        this.dims = List.of(dim1, dim2, dim3);
    }

    public CodeType rType() {
        return rType;
    }

    @Override
    public int hashCode() {
        return Objects.hash(rType);
    }

    @Override
    public ExternalizedCodeType externalize() {
        return ExternalizedCodeType.of("tilePtrType", List.of(rType.externalize()));
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {return true;}
        if (obj == null || getClass() != obj.getClass()) {return false;}
        final PtrType other = (PtrType) obj;
        return Objects.equals(this.rType, other.rType);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }

    public List<Object> dims() {
        return dims;
    }
}
