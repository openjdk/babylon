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
package hat.codetypes;

import jdk.incubator.code.CodeType;
import jdk.incubator.code.extern.ExternalizedCodeType;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Type to be used within the TypeInterpreter to pass build/pass
 * tensors with a specific shape.
 */
public final class TensorType implements TileType {

    private final CodeType dType;   // dType
    private final List<Integer> shape;  // Shape

    public TensorType(CodeType codeType, List<Integer> shape) {
        this.dType = codeType;
        this.shape = shape;
    }

    public CodeType elementType() {
        return dType;
    }

    public List<Integer> shape() {
        return shape;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) { return true; }
        if (o == null || getClass() != o.getClass()) { return false; }
        TensorType tensorType = (TensorType) o;
        return Objects.equals(dType, tensorType.dType) && Objects.equals(shape, tensorType.shape);
    }

    @Override
    public ExternalizedCodeType externalize() {
        List<ExternalizedCodeType> externalizedTypes = shape.stream().map(s -> new ExternalizedCodeType("x" + s, List.of())).collect(Collectors.toList());
        externalizedTypes.add(dType.externalize());
        return ExternalizedCodeType.of("tensor", externalizedTypes);
    }

    @Override
    public int hashCode() {
        return Objects.hash(dType, shape);
    }

    @Override
    public String toString() {
        return externalize().toString();
    }
}
