
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

public interface F32Vec2 {
    List<F32Vec2> arr = new ArrayList<>();
    static void reset(int markedVec2) {
        F32Vec2.pool.count = markedVec2;
        while (arr.size()>markedVec2){
            arr.removeLast();
        }
    }

    float x(); void x(float x);
    float y(); void y(float y);
    int X = 0;
    int Y = 1;
    class Pool {
        public final float entries[];
        public final int stride=2;
        public final int max=12800;
        public int count =0 ;
        Pool() {
            this.entries = new float[max * stride];
        }
    }
    Pool pool = new Pool();

    class Impl implements F32Vec2{
        public int id;
        float x,y;
        public int  id() {return id;}
        public void id(int id) {this.id= id;}
        @Override public float x() {return x;}
        @Override public void x(float x) {this.x = x;}
        @Override public float y() {return y;}
        @Override public void y(float y) {this.y = y;}
        Impl(int id, float x, float y){id(id); x(x);y(y);}
    }
     static Impl createVec2(float x, float y) {
        pool.entries[pool.count * pool.stride + X] = x;
        pool.entries[pool.count++ * pool.stride + Y] = y;
        var impl = new Impl(arr.size(), x,y);
        arr.add(impl);
        return impl;
    }
    /*

    static int mulScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] * s, pool.entries[i + Y] * s);
    }

    static int addScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] + s, pool.entries[i + Y] + s);
    }

    static int divScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] / s, pool.entries[i + Y] / s);
    }

    static int addVec2(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec2(pool.entries[lhs + X] + pool.entries[rhs + X], pool.entries[lhs + Y] + pool.entries[rhs + Y]);
    }

    static int subVec2(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec2(pool.entries[lhs + X] - pool.entries[rhs + X], pool.entries[lhs + Y] - pool.entries[rhs + Y]);
    }


    static float dotProd(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return pool.entries[lhs + X] * pool.entries[rhs + X] + pool.entries[lhs + Y] * pool.entries[rhs + Y];
    }

    static String asString(int i) {
        i *= pool.stride;
        return pool.entries[i + X] + "," + pool.entries[i + Y];
    } */
}
