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
package hat.types;

import jdk.incubator.code.CodeType;
import optkl.IfaceValue;

/**
 * Data type to represent views over tensors (tiles) within the compute kernels.
 */
public record Tile() implements IfaceValue {

    /**
     * Transpose an input tile (swap first and second dimensions of the tile shape.
     * @return
     *    Transposed tile
     */
    public Tile transpose() {
        return new Tile();
    }

    /**
     * Transform an input tile to type dType.
     *
     * @param dType
     *    Type to be transformed
     * @return
     *    A new Tile of type dType.
     */
    public Tile asType(CodeType dType) {
        return new Tile();
    }
}
