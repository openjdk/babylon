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

import view.f32.F32;
import view.f32.F32x2;
import view.f32.F32x2Triangle;

public class F32x2TrianglePool extends Pool<F32x2Triangle, F32x2TrianglePool> implements F32x2Triangle.Factory {

    public record PoolEntry(F32x2TrianglePool pool,
                            int idx) implements Pool.PoolEntry<F32x2Triangle, F32x2TrianglePool>, F32x2Triangle {

        private static final int V0 = 0;
        private static final int V1 = 1;
        private static final int V2 = 2;

        int v0Idx() {
            return f32x2Stride * idx + V0;
        }

        public F32x2 v0() {
            return pool.f32X2Entries[v0Idx()];
        }

        private int v1Idx() {
            return f32x2Stride * idx + V1;
        }

        public F32x2 v1() {
            return pool.f32X2Entries[v1Idx()];
        }

        private int v2Idx() {
            return f32x2Stride * idx + V2;
        }

        public F32x2 v2() {
            return pool.f32X2Entries[v2Idx()];
        }

        private int zplaneIdx() {
            return idx * zplaneStride;
        }

        public float zplane() {
            return pool.zplaneEntries[zplaneIdx()];
        }

        private int normalIdx() {
            return idx * normalStride;
        }

        public float normal() {
            return pool.normalEntries[rgbIdx()];
        }

        private int rgbIdx() {
            return idx * rgbStride;
        }

        public int rgb() {
            return pool.rgbEntries[rgbIdx()];
        }


    }

    public final F32x2 f32X2Entries[];
    public final float zplaneEntries[];
    public final float normalEntries[];
    private final int rgbEntries[];
    private static final int f32x2Stride = 3;
    private static final int rgbStride = 1;
    private static final int zplaneStride = 1;
    private static final int normalStride = 1;

    public F32x2TrianglePool(int max) {
        super(max);
        this.f32X2Entries = new F32x2[max * f32x2Stride];
        this.zplaneEntries = new float[max * zplaneStride];
        this.normalEntries = new float[max * normalStride];
        this.rgbEntries = new int[max * rgbStride];
    }

    @Override
    public F32x2Triangle entry(int idx) {
        return new PoolEntry(this, idx);
    }


    @Override
    public F32x2Triangle of(F32x2 v0, F32x2 v1, F32x2 v2, Float zplane, Float normal, Integer rgb) {
        var i = (PoolEntry) entry(count++);
        var side = F32.side(v0.x(), v0.y(), v1, v2) > 0; // We need the triangle to be clock wound
        f32X2Entries[i.v0Idx()] = v0;
        f32X2Entries[i.v1Idx()] = side ? v1 : v2;
        f32X2Entries[i.v2Idx()] = side ? v2 : v1;

        zplaneEntries[i.zplaneIdx()] = zplane;
        normalEntries[i.normalIdx()] = normal;
        rgbEntries[i.rgbIdx()] = rgb;
        return i;
    }
}
