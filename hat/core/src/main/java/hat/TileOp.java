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
package hat;

import hat.types.Tile;

/**
 * Provides arithmetic, reduction, shape-transformation and construction operations
 * for the {@link Tile} Programming Model implementation.
 *
 * <p>Binary operations involving two tiles are performed element-wise.
 * Operations that involve a tile and a scalar values apply the scalar
 * to every element of the tile. Compatible tile shapes may be broadcasted
 * by the compiler (or HAT Transformer).</p>
 *
 * <p>All methods in this class are compiler intrinsics. Calls made from a HAT
 * Tile kernel are replaced by the HAT Tile Transformer and lowered to the
 * backend-specific operations.</p>
 *
 * @apiNote These methods have no ordinary Java implementation and throws an
 * {@link UnsupportedOperationException} exception. Support for a Java CPU
 * execution is planned for future versions of the Tile programming model
 * implementation within HAT.
 */
public class TileOp {

}