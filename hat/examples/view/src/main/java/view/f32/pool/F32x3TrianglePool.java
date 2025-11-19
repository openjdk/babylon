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
import view.f32.F32x3Triangle;

public class F32x3TrianglePool extends Pool<F32x3Triangle,F32x3TrianglePool> implements F32x3Triangle.Factory{
    private static final int V0 = 0;
    private static final int V1 = 1;
    private static final int V2 = 2;

    public record PoolEntry(F32x3TrianglePool pool, int idx)
            implements Pool.PoolEntry<F32x3Triangle,F32x3TrianglePool>, F32x3Triangle {
        int v0Idx() {
            return F32x3TrianglePool.f32x3Stride * idx + V0;
        }

        public F32x3 v0() {
            return pool.f32x3Entries[v0Idx()];
        }

        public int v1Idx() {
            return F32x3TrianglePool.f32x3Stride * idx + V1;
        }

        public F32x3 v1() {
            return pool.f32x3Entries[v1Idx()];
        }

        public int v2Idx() {
            return F32x3TrianglePool.f32x3Stride * idx + V2;
        }

        public F32x3 v2() {
            return pool.f32x3Entries[v2Idx()];
        }

        int rgbIdx() {
            return F32x3TrianglePool.rgbStride * idx;
        }

        public int rgb() {
            return pool.rgbEntries[rgbIdx()];
        }
    }

    public final F32x3 f32x3Entries[];
    private final static int f32x3Stride = 3;
    public final int rgbEntries[];
    private final static int rgbStride = 1;

    public F32x3TrianglePool(int max) {
        super(max);
        this.f32x3Entries = new F32x3[max * f32x3Stride];
        this.rgbEntries = new int[max];
    }

    @Override
    public F32x3Triangle entry(int idx) {
        return new PoolEntry(this, idx);
    }


    @Override
    public F32x3Triangle of(F32x3 v0, F32x3 v1, F32x3 v2, Integer rgb) {
        var i = (PoolEntry)entry(count++);
        f32x3Entries[i.v0Idx()] = v0;
        f32x3Entries[i.v1Idx()] = v1;
        f32x3Entries[i.v2Idx()] = v2;
        rgbEntries[i.rgbIdx()] = rgb;
        return i;
    }
}
