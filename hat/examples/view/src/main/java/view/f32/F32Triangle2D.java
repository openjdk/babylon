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

import java.util.ArrayList;
import java.util.List;

public interface F32Triangle2D {
    List<F32Triangle2D> arr = new ArrayList<>();
    static void reset(int marked) {
        F32Triangle2D.pool.count = marked;
        while (arr.size()>marked){
            arr.removeLast();
        }
    }


    F32Vec2 v0(); void v0(F32Vec2 v0);
    F32Vec2 v1(); void v1(F32Vec2 v1);
    F32Vec2 v2(); void v2(F32Vec2 v2);
    int rgb(); void rgb(int rgb);
    class Impl implements F32Triangle2D{
        public int id;
        F32Vec2 v0,v1,v2;
        int rgb;
        public int  id() {return id;}
        public void id(int id) {this.id= id;}
        @Override public F32Vec2 v0() {return v0;}
        @Override public void v0(F32Vec2 v0) {this.v0 = v0;}
        @Override public F32Vec2 v1() {return v1;}
        @Override public void v1(F32Vec2 v1) {this.v1 = v1;}
        @Override public F32Vec2 v2() {return v2;}
        @Override public void v2(F32Vec2 v2) {this.v2 = v2;}
        @Override public int  rgb() {return rgb;}
        @Override public void rgb(int rgb) {this.rgb = rgb;}
        Impl(int id, F32Vec2 v0, F32Vec2 v1,F32Vec2 v2,int rgb){id(id); v0(v0);v1(v1);v2(v2);rgb(rgb);}
    }


    int V0 = 0;
     int V1 = 1;
     int V2 = 2;
    int RGB=3;


    class Pool extends IndexPool{
        Pool( int max) {
            super(4, max);
        }
    }
    Pool pool = new Pool(9000);

      static float side(float x, float y, float x0, float y0, float x1, float y1) {
        return (y1 - y0) * (x - x0) + (-x1 + x0) * (y - y0);
    }

     static float side(int v, int v0, int v1) {
        v*= F32Vec2.pool.stride;
        v0*= F32Vec2.pool.stride;
        v1*= F32Vec2.pool.stride;
        return (F32Vec2.pool.entries[v1+ F32Vec2.Y] - F32Vec2.pool.entries[v0+ F32Vec2.Y] * (F32Vec2.pool.entries[v+ F32Vec2.X] - F32Vec2.pool.entries[v0+ F32Vec2.X]) + (-F32Vec2.pool.entries[v1+ F32Vec2.X] + F32Vec2.pool.entries[v0+ F32Vec2.X]) * (F32Vec2.pool.entries[v+ F32Vec2.Y] - F32Vec2.pool.entries[v0+ F32Vec2.Y]));
    }

     static boolean intriangle(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
        return side(x, y, x0, y0, x1, y1) >= 0 && side(x, y, x1, y1, x2, y2) >= 0 && side(x, y, x2, y2, x0, y0) >= 0;
    }
     static boolean intriangle(int v, int v0, int v1, int v2){
        return side(v, v0, v1) >= 0 && side(v, v1, v2) >= 0 && side(v, v2, v0) >= 0;
    }

     static boolean online(float x, float y, float x0, float y0, float x1, float y1, float deltaSquare) {
        float dxl = x1 - x0;
        float dyl = y1 - y0;
        float cross = (x - x0) * dyl - (y - y0) * dxl;
        if ((cross * cross) < deltaSquare) {
            if (dxl * dxl >= dyl * dyl)
                return dxl > 0 ? x0 <= x && x <= x1 : x1 <= x && x <= x0;
            else
                return dyl > 0 ? y0 <= y && y <= y1 : y1 <= y && y <= y0;
        } else {
            return false;
        }
    }
    float deltaSquare = 10000f;

     static boolean onedge(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
        return online(x, y, x0, y0, x1, y1, deltaSquare) || F32Triangle2D.online(x, y, x1, y1, x2, y2, deltaSquare) || F32Triangle2D.online(x, y, x2, y2, x0, y0, deltaSquare);
    }


 static Impl createTriangle(int x0, int y0, int x1, int y1, int x2, int y2, int col) {
         var impl =side(x0, y0, x1, y1, x2, y2)>0 // We need the triangle to be clock wound
                ?new Impl(arr.size(),F32Vec2.createVec2(x0,y0),F32Vec2.createVec2(x1,y1),F32Vec2.createVec2(x2,y2),col )
                :new Impl(arr.size(),F32Vec2.createVec2(x0,y0),F32Vec2.createVec2(x2,y2),F32Vec2.createVec2(x1,y1),col );
        pool.entries[pool.count * pool.stride + V0] = ((F32Vec2.Impl)impl.v0).id();//F32Vec2.createVec2(x0,y0).id;
        pool.entries[pool.count * pool.stride + V1] = ((F32Vec2.Impl)impl.v1).id();
        pool.entries[pool.count * pool.stride + V2] = ((F32Vec2.Impl)impl.v2).id();
         pool.entries[pool.count++ * pool.stride + RGB]= impl.rgb();
        return impl;
    }

}
