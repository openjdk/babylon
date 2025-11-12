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

import view.f32.F32x4x4;

public class F32x4x4Pool extends F32Pool<F32x4x4,F32x4x4Pool> implements F32x4x4.Factory {
    static int X0Y0 = 0;
    static int X1Y0 = 1;
    static int X2Y0 = 2;
    static int X3Y0 = 3;
    static int X0Y1 = 4;
    static int X1Y1 = 5;
    static int X2Y1 = 6;
    static int X3Y1 = 7;
    static int X0Y2 = 8;
    static int X1Y2 = 9;
    static int X2Y2 = 10;
    static int X3Y2 = 11;
    static int X0Y3 = 12;
    static int X1Y3 = 13;
    static int X2Y3 = 14;
    static int X3Y3 = 15;




    public record PoolEntry(F32x4x4Pool pool, int idx) implements Pool.PoolEntry<F32x4x4, F32x4x4Pool>, F32x4x4 {
        private int x0y0Idx() {
            return idx * pool.stride + X0Y0;
        }

        private int x1y0Idx() {
            return idx * pool.stride + X1Y0;
        }

        private int x2y0Idx() {
            return idx * pool.stride + X2Y0;
        }

        private int x3y0Idx() {
            return idx * pool.stride + X3Y0;
        }

        private int x0y1Idx() {
            return idx * pool.stride + X0Y1;
        }

        private int x1y1Idx() {
            return idx * pool.stride + X1Y1;
        }

        private int x2y1Idx() {
            return idx * pool.stride + X2Y1;
        }

        private int x3y1Idx() {
            return idx * pool.stride + X3Y1;
        }

        int x0y2Idx() {
            return idx * pool.stride + X0Y2;
        }

        int x1y2Idx() {
            return idx * pool.stride + X1Y2;
        }

        int x2y2Idx() {
            return idx * pool.stride + X2Y2;
        }

        int x3y2Idx() {
            return idx * pool.stride + X3Y2;
        }

        int x0y3Idx() {
            return idx * pool.stride + X0Y3;
        }

        int x1y3Idx() {
            return idx * pool.stride + X1Y3;
        }

        int x2y3Idx() {
            return idx * pool.stride + X2Y3;
        }

        int x3y3Idx() {
            return idx * pool.stride + X3Y3;
        }

        @Override
        public float x0y0() {
            return pool.floatEntries[x0y0Idx()];
        }

        @Override
        public float x1y0() {
            return pool.floatEntries[x1y0Idx()];
        }

        @Override
        public float x2y0() {
            return pool.floatEntries[x2y0Idx()];
        }

        @Override
        public float x3y0() {
            return pool.floatEntries[x3y0Idx()];
        }

        @Override
        public float x0y1() {
            return pool.floatEntries[x0y1Idx()];
        }

        @Override
        public float x1y1() {
            return pool.floatEntries[x1y1Idx()];
        }

        @Override
        public float x2y1() {
            return pool.floatEntries[x2y1Idx()];
        }

        @Override
        public float x3y1() {
            return pool.floatEntries[x3y1Idx()];
        }

        @Override
        public float x0y2() {
            return pool.floatEntries[x0y2Idx()];
        }

        @Override
        public float x1y2() {
            return pool.floatEntries[x1y2Idx()];
        }

        @Override
        public float x2y2() {
            return pool.floatEntries[x2y2Idx()];
        }

        @Override
        public float x3y2() {
            return pool.floatEntries[x3y2Idx()];
        }

        @Override
        public float x0y3() {
            return pool.floatEntries[x0y3Idx()];
        }

        @Override
        public float x1y3() {
            return pool.floatEntries[x1y3Idx()];
        }

        @Override
        public float x2y3() {
            return pool.floatEntries[x2y3Idx()];
        }

        @Override
        public float x3y3() {
            return pool.floatEntries[x3y3Idx()];
        }
    }

    public F32x4x4Pool(int max) {
        super(16, max);
    }

    @Override
    public F32x4x4 entry(int idx) {
        return new PoolEntry(this, idx);
    }
@Override
    public F32x4x4 of(Float x0y0, Float x1y0, Float x2y0, Float x3y0,
                      Float x0y1, Float x1y1, Float x2y1, Float x3y1,
                      Float x0y2, Float x1y2, Float x2y2, Float x3y2,
                      Float x0y3, Float x1y3, Float x2y3, Float x3y3) {
        var i = (PoolEntry)entry(count++);
        floatEntries[i.x0y0Idx()] = x0y0;
        floatEntries[i.x1y0Idx()] = x1y0;
        floatEntries[i.x2y0Idx()] = x2y0;
        floatEntries[i.x3y0Idx()] = x3y0;
        floatEntries[i.x0y1Idx()] = x0y1;
        floatEntries[i.x1y1Idx()] = x1y1;
        floatEntries[i.x2y1Idx()] = x2y1;
        floatEntries[i.x3y1Idx()] = x3y1;
        floatEntries[i.x0y2Idx()] = x0y2;
        floatEntries[i.x1y2Idx()] = x1y2;
        floatEntries[i.x2y2Idx()] = x2y2;
        floatEntries[i.x3y2Idx()] = x3y2;
        floatEntries[i.x0y3Idx()] = x0y3;
        floatEntries[i.x1y3Idx()] = x1y3;
        floatEntries[i.x2y3Idx()] = x2y3;
        floatEntries[i.x3y3Idx()] = x3y3;
        return i;
    }

}
