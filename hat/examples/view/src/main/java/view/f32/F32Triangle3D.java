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

public interface F32Triangle3D{
    class F32Triangle3DPool extends Pool<F32Triangle3DPool> {
       static  int V0 = 0;
       static int V1 = 1;
       static  int V2 = 2;
       static  int RGB = 3;
        public record Idx(F32Triangle3DPool pool, int idx) implements Pool.Idx<F32Triangle3DPool> {
            int v0(){return pool.stride * idx+V0;}
            int v0Entry(){return pool.entries[v0()];}
            int v1(){return pool.stride * idx+V1;}
            int v1Entry(){return pool.entries[v1()];}
            int v2(){return pool.stride * idx+V2;}
            int v2Entry(){return pool.entries[v2()];}
            int rgb(){return pool.stride * idx+RGB;}
            int rgbEntry(){return pool.entries[rgb()];}
        }
        public final int entries[];
        F32Triangle3DPool(int max) {
            super(4, max);
            this.entries = new int[max * stride];
        }

        @Override
        Idx idx(int idx) {
            return new Idx(this, idx);
        }
    }

    F32Triangle3DPool f32Triangle3DPool = new F32Triangle3DPool(12800);

     /*
       v0----v1         v0----v2
        \    |           \    |
         \   |            \   |
          \  |    --->     \  |
           \ |              \ |
            \|               \|
             v2               v1
   */

     static F32Triangle3DPool.Idx rewind(F32Triangle3DPool.Idx i) {
        int temp = i.v1Entry();
        f32Triangle3DPool.entries[i.v1()] =  i.v2Entry();
        f32Triangle3DPool.entries[i.v2()] = temp;
        return i;
    }

     static F32Triangle3DPool.Idx of(int v0, int v1, int v2, int rgb) {
         var i = f32Triangle3DPool.idx(f32Triangle3DPool.count++);//pool.count * pool.stride
         f32Triangle3DPool.entries[i.v0()] = v0;
         f32Triangle3DPool.entries[i.v1()] = v1;
         f32Triangle3DPool.entries[i.v2()] = v2;
         f32Triangle3DPool.entries[i.rgb()] = rgb;
        return i;
    }

    static String asString(F32Triangle3DPool.Idx i) {
        return F32Vec3.asString(i.v0Entry()) + " -> " + F32Vec3.asString(i.v1Entry()) + " -> " + F32Vec3.asString(i.v2Entry()) + " =" + String.format("0x%8x", i.rgbEntry());
    }

     static F32Triangle3DPool.Idx mulMat4(F32Triangle3DPool.Idx i, F32Matrix4x4.F32Matrix4x4Pool.Idx  m4) {
        return of(F32Vec3.mulMat4(i.v0Entry(), m4), F32Vec3.mulMat4(i.v1Entry(), m4), F32Vec3.mulMat4(i.v2Entry(), m4), i.rgbEntry());
    }

     static F32Triangle3DPool.Idx addVec3(F32Triangle3DPool.Idx i, int v3) {
        return of(F32Vec3.addVec3(i.v0Entry(), v3), F32Vec3.addVec3(i.v1Entry(), v3), F32Vec3.addVec3(i.v2Entry(), v3), i.rgbEntry());
    }

     static F32Triangle3DPool.Idx mulScaler(F32Triangle3DPool.Idx i, float s) {
        return of(F32Vec3.mulScaler(i.v0Entry(), s), F32Vec3.mulScaler(i.v1Entry(), s), F32Vec3.mulScaler(i.v2Entry(), s), i.rgbEntry());
    }

     static F32Triangle3DPool.Idx addScaler(F32Triangle3DPool.Idx i, float s) {
        return of(F32Vec3.addScaler(i.v0Entry(), s), F32Vec3.addScaler(i.v1Entry(), s), F32Vec3.addScaler(i.v2Entry(), s), i.rgbEntry());
    }

     static int getCentre(F32Triangle3DPool.Idx i){// the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

     static int getVectorSum(F32Triangle3DPool.Idx i){// the sum of all the vertices
        return F32Vec3.addVec3(F32Vec3.addVec3(i.v0Entry(), i.v1Entry()),i.v2Entry());
    }


     static int normal(F32Triangle3DPool.Idx i) {
        int line1Vec3 = F32Vec3.subVec3(i.v1Entry(), i.v0Entry());
        int line2Vec3 = F32Vec3.subVec3(i.v2Entry(),  i.v0Entry());
        return F32Vec3.crossProd(line1Vec3, line2Vec3);
    }

     static int normalSumOfSquares(F32Triangle3DPool.Idx i) {
        int normalVec3 = normal(i);
        return F32Vec3.divScaler(normalVec3,  F32Vec3.sumOfSquares(normalVec3));
    }

    record F32Triangle3DImpl(F32Triangle3DPool.Idx id) implements F32Triangle3D {
        public static List<F32Triangle3DImpl> all() {
                List<F32Triangle3DImpl> all = new ArrayList<>();
                for (int t = 0; t < f32Triangle3DPool.count; t++) {
                    all.add(new F32Triangle3DImpl(f32Triangle3DPool.idx(t))/*Pool.Idx.of(t))*/);
                }
                return all;
            }

            public F32Triangle3DImpl mul(F32Matrix4x4.Impl m) {
                return new F32Triangle3DImpl(mulMat4(id, m.id()));
            }

            public F32Triangle3DImpl add(F32Vec3.F32Vec3Impl v) {
                return new F32Triangle3DImpl(addVec3(id, v.id().idx()));

            }

            public F32Vec3.F32Vec3Impl normalSumOfSquares() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normalSumOfSquares(id)));
            }

            public F32Vec3.F32Vec3Impl normal() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normal(id)));
            }

            public F32Vec3.F32Vec3Impl v0() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v0Entry()));
            }

            public F32Vec3.F32Vec3Impl v1() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v1Entry()));
            }

            public F32Vec3.F32Vec3Impl v2() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v2Entry()));
            }

            public F32Triangle3DImpl mul(float s) {
                return new F32Triangle3DImpl(mulScaler(id, s));
            }

            public F32Triangle3DImpl add(float s) {
                return new F32Triangle3DImpl(addScaler(id, s));
            }

            public int rgb() {
                return id.rgbEntry();
            }

            public F32Vec3.F32Vec3Impl center() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(getCentre(id)));
            }
        }
}
