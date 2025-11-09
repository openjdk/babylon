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
            var side = false;//side(v0.x(),v0.y(), v1, v2) > 0; // We need the triangle to be clock wound
            entries[i.v0Idx()] = v0;
            entries[i.v1Idx()] = side?v1:v2;
            entries[i.v2Idx()] = side?v2:v1;
            rgbs[i.rgbIdx()] = rgb;
            return i;
        }
    }

    /*
           return (
               v1.Y() - v0.Y() * (v.X() - v0.X()) + (-v1.X() + v0.X()) * (v.Y() - v0.Y())
               );

     */
    /*
       return (v1.Y() - v0.Y() * (v.X() - v0.X()) + (-v1.X() + v0.X()) * (v.Y() - v0.Y()));

     */

    F32Triangle2DPool f32Triangle2DPool = new F32Triangle2DPool(12800);

    static float side(float x,float y, F32Vec2 v0, F32Vec2 v1) {
         return    (v1.y() - v0.y() * (x - v0.x()) + (-v1.x() + v0.x()) * (y - v0.y()));
    }

    /*
              V0                V0
              |  \              |  \
              |    \            |    \        P2
              |  P1  \          |      \
              V1------V0        V1------V0


Barycentric coordinate allows to express new p coordinates as a linear combination of p1, p2, p3.
 More precisely, it defines 3 scalars a, b, c such that :

x = a * x1 + b * x2  + c * x3
y = a * y1 + b * y2 + c * y3
a + b + c = 1


a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
c = 1 - a - b

p lies in T if and only if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
*/

    static boolean intriangle (float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
        var denominator = ((y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2));
        var a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denominator;
        var b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denominator;
        var c = 1 - a - b;
        return 0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1;
    }
    static boolean intriangle(float x, float y,  F32Vec2 v0, F32Vec2 v1, F32Vec2 v2) {
        return intriangle(x,y, v0.x(), v0.y(), v1.x(), v1.y(),v2.x(), v2.y());
    }

    static boolean intriangle(float x, float y,  F32Triangle2D tri) {
       return intriangle(x,y, tri.v0(), tri.v1(),tri.v2());
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

    float deltaSquare = 2000f;

    static boolean onedge(float x, float y, F32Triangle2D tri) {
        return online(x, y, tri.v0(), tri.v1(), deltaSquare)
                || online(x, y,tri.v1(),tri.v2(), deltaSquare)
                || online(x, y, tri.v2(),tri.v0(), deltaSquare);
    }

    static  boolean useRgb(boolean filled, float x, float y,  F32Triangle2D tri){
        return filled? intriangle(x,y,tri):onedge(x,y,tri);
    }

    static  int rgb(boolean filled, float x, float y,  F32Triangle2D tri, int rgb){
        return useRgb(filled,x,y,tri)? tri.rgb() : rgb;
    }


}
