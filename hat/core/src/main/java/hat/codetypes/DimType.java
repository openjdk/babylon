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

import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Internal Type to the TypeInterpreter for providing/passing through the
 * tile shapes. This facilites shapes checking at runtime, before specializing
 * the code model.
 */
public final class DimType implements TileType {

    private final List<Integer> dims;

    public DimType(ConstantType ...dims) {
        this.dims = list(Arrays.stream(dims).toList());
    }

    public int dimensions() {
        return dims.size();
    }

    public List<Integer> list(List<ConstantType> shapes) {
        return shapes.stream()
                .map(shape -> (Integer) shape.value)
                .collect(Collectors.toList());
    }

    public List<Integer> list() {
        return dims;
    }

    @Override
    public ExternalizedCodeType externalize() {
        List<ExternalizedCodeType> externalizedTypes = dims.stream().map(s -> new ExternalizedCodeType("s" + s, List.of())).collect(Collectors.toList());
        return ExternalizedCodeType.of("dim", externalizedTypes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.dims);
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == this) { return true; }
        if (obj == null || obj.getClass() != DimType.class) { return false; }
        DimType other = (DimType) obj;
        return Objects.equals(this.dims, other.dims);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
