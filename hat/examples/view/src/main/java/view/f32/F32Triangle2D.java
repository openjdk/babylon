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
package view.f32;

public interface F32Triangle2D {
    F32Vec2 v0();

    F32Vec2 v1();

    F32Vec2 v2();

    int rgb();

    static void reset(int marked) {
        F32Triangle2D.f32Triangle2DPool.count = marked;
    }

    class F32Triangle2DPool extends Pool<F32Triangle2D.F32Triangle2DPool> {
        static int V0 = 0;
        static int V1 = 1;
        static int V2 = 2;


        public record Idx(F32Triangle2DPool pool, int idx) implements Pool.Idx<F32Triangle2D.F32Triangle2DPool>, F32Triangle2D {
            int v0Idx() {
                return pool.stride * idx + V0;
            }

            public F32Vec2 v0() {
                return pool.entries[v0Idx()];
            }

            int v1Idx() {
                return pool.stride * idx + V1;
            }

            public F32Vec2 v1() {
                return pool.entries[v1Idx()];
            }

            int v2Idx() {
                return pool.stride * idx + V2;
            }

            public F32Vec2 v2() {
                return pool.entries[v2Idx()];
            }

            int rgbIdx() {
                return idx;
            }

            public int rgb() {return pool.rgbs[rgbIdx()];}
        }

        public final F32Vec2 entries[];
        public final int rgbs[];

        F32Triangle2DPool(int max) {
            super(3, max);
            this.entries = new F32Vec2[max * stride];
            this.rgbs = new int[max];
        }

        @Override
        public F32Triangle2D.F32Triangle2DPool.Idx idx(int idx) {
            return new F32Triangle2D.F32Triangle2DPool.Idx(this, idx);
        }

        public F32Triangle2D of(F32Vec2 v0, F32Vec2 v1, F32Vec2 v2, int rgb) {
            var i = idx(count++);
            var side = side(v0.x(),v0.y(), v1, v2) > 0; // We need the triangle to be clock wound
            entries[i.v0Idx()] = v0;
            entries[i.v1Idx()] = side?v1:v2;
            entries[i.v2Idx()] = side?v2:v1;
            rgbs[i.rgbIdx()] = rgb;
            return i;
        }
    }

    F32Triangle2DPool f32Triangle2DPool = new F32Triangle2DPool(12800);

    static float side(float x,float y, F32Vec2 v0, F32Vec2 v1) {
        return (v1.y() - v0.y() * x - v0.x() + (-v1.x() + v0.x()) * (y -v0.y()));
    }

    static boolean intriangle(float x, float y,  F32Vec2 v0, F32Vec2 v1, F32Vec2 v2) {
        return side(x,y, v0, v1) >= 0 && side(x,y, v1, v2) >= 0 && side(x,y, v2, v0) >= 0;
    }

    static boolean online(float x, float y,  F32Vec2 v0, F32Vec2 v1, float deltaSquare) {
        float dxl = v1.x() - v0.x();
        float dyl = v1.y() - v0.y();;
        float cross = (x - v0.x()) * dyl - (y - v0.y()) * dxl;
        if ((cross * cross) < deltaSquare) {
            if (dxl * dxl >= dyl * dyl)
                return dxl > 0 ? v0.x() <= x && x <= v1.x() : v1.x() <= x && x <= v0.x();
            else
                return dyl > 0 ? v0.y() <= y && y <= v1.y() : v1.y() <= y && y <= v0.y();
        } else {
            return false;
        }
    }

    float deltaSquare = 10000f;

    static boolean onedge(float x, float y, F32Vec2 v0,F32Vec2 v1,F32Vec2 v2) {
        return online(x, y, v0,v1, deltaSquare) || online(x, y,v1,v2, deltaSquare) || online(x, y, v2,v0, deltaSquare);
    }

}
