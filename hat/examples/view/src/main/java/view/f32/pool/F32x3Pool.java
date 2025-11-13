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
package view.f32.pool;

import view.f32.F32x3;

public class F32x3Pool extends F32Pool<F32x3,F32x3Pool> implements F32x3.Factory {
    private final static int X = 0;
    private final static int Y = 1;
    private final static int Z = 2;

    public record PoolEntry(F32x3Pool pool, int idx) implements Pool.PoolEntry<F32x3, F32x3Pool>, F32x3 {
        private int xIdx() {
            return pool.floatStride * idx + X;
        }

        @Override
        public float x() {
            return pool.floatEntries[xIdx()];
        }

        private int yIdx() {
            return pool.floatStride * idx + Y;
        }

        @Override
        public float y() {
            return pool.floatEntries[yIdx()];
        }

        private int zIdx() {
            return pool.floatStride * idx + Z;
        }

        @Override
        public float z() {
            return pool.floatEntries[zIdx()];
        }
    }

    public F32x3Pool(int max) {
        super(3, max);
    }

    @Override
    public F32x3 entry(int idx) {
        return new PoolEntry(this, idx);
    }

    @Override
    public F32x3 of(Float x, Float y, Float z) {
        PoolEntry i = (PoolEntry) entry(count++);
        floatEntries[i.xIdx()] = x;
        floatEntries[i.yIdx()] = y;
        floatEntries[i.zIdx()] = z;
        return i;
    }
}
