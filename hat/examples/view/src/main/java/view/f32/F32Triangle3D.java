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

public interface F32Triangle3D {
    F32Vec3 v0();

    F32Vec3 v1();

    F32Vec3 v2();

    int rgb();

    int idx();

    default String asString() {
        return v0().asString() + " -> " + v1().asString() + " -> " + v2().asString() + " =" + String.format("0x%8x", rgb());
    }

    class F32Triangle3DPool extends Pool<F32Triangle3DPool> {
       static  int V0 = 0;
       static int V1 = 1;
       static  int V2 = 2;
       static  int RGB = 3;
        public record Idx(F32Triangle3DPool pool, int idx) implements Pool.Idx<F32Triangle3DPool>,F32Triangle3D {
            int v0Idx(){return pool.stride * idx+V0;}
           public F32Vec3 v0(){return pool.entries[v0Idx()];}
            int v1Idx(){return pool.stride * idx+V1;}
             public F32Vec3 v1(){return pool.entries[v1Idx()];}
            int v2Idx(){return pool.stride * idx+V2;}
             public  F32Vec3 v2(){return pool.entries[v2Idx()];}
            int rgbIdx(){return idx;}
             public int rgb(){return pool.rgbs[rgbIdx()];}
        }
        public final F32Vec3 entries[];
        public final int rgbs[];
        F32Triangle3DPool(int max) {
            super(3, max);
            this.entries = new F32Vec3[max * stride];
            this.rgbs = new int[max];
        }

        @Override
        public Idx idx(int idx) {
            return new Idx(this, idx);
        }

        public F32Triangle3D of(F32Vec3 v0, F32Vec3 v1, F32Vec3 v2, int rgb) {
            var i = idx(count++);//pool.count * pool.stride
            entries[i.v0Idx()] = v0;
            entries[i.v1Idx()] = v1;
            entries[i.v2Idx()] = v2;
            if (rgb == 0){
                throw new IllegalStateException("rgb = 0");
            }
            rgbs[i.rgbIdx()] = rgb;
            return i;
        }
        public F32Triangle3D of(int v0Idx, int v1Idx, int v2Idx, int rgb) {
            if (rgb == 0){
                throw new IllegalStateException("rgb is 0");
            }
          return of(F32Vec3.f32Vec3Pool.idx(v0Idx),F32Vec3.f32Vec3Pool.idx(v1Idx),F32Vec3.f32Vec3Pool.idx(v2Idx),rgb);
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

    static F32Triangle3D rewind(F32Triangle3D i) {
        var temp = i.v1();
        ((F32Triangle3D.F32Triangle3DPool.Idx) i).pool.entries[((F32Triangle3D.F32Triangle3DPool.Idx) i).v1Idx()] = i.v2();
        ((F32Triangle3D.F32Triangle3DPool.Idx) i).pool.entries[((F32Triangle3D.F32Triangle3DPool.Idx) i).v2Idx()] = temp;
        return i;
    }

   //  static F32Triangle3D of(F32Vec3 v0, F32Vec3 v1, F32Vec3 v2, int rgb) {
     //    var i = f32Triangle3DPool.idx(f32Triangle3DPool.count++);//pool.count * pool.stride
       //  f32Triangle3DPool.entries[i.v0Idx()] = v0;
       //  f32Triangle3DPool.entries[i.v1Idx()] = v1;
       //  f32Triangle3DPool.entries[i.v2Idx()] = v2;
       //  f32Triangle3DPool.rgbs[i.rgbIdx()] = rgb;
       // return i;
   // }
   // static F32Triangle3D of(int v0, int v1, int v2, int rgb) {
     // return f32Triangle3DPool.of(F32Vec3.F32Vec3Impl.of(v0),F32Vec3.F32Vec3Impl.of(v1),F32Vec3.F32Vec3Impl.of(v2),rgb);

   // }


     static F32Triangle3D mulMat4(F32Triangle3D i, F32Matrix4x4.F32Matrix4x4Pool.Idx  m4) {
        return f32Triangle3DPool.of(F32Vec3.mulMat4(i.v0().idx(), m4), F32Vec3.mulMat4(i.v1().idx(), m4), F32Vec3.mulMat4(i.v2().idx(), m4), i.rgb());
    }
    static F32Triangle3D mulMat4(F32Triangle3D i, F32Matrix4x4  m4) {
        if (i.rgb()==0){
            throw new IllegalStateException("i.rgb == 0");
        }
        return f32Triangle3DPool.of(F32Vec3.mulMat4(i.v0().idx(), m4), F32Vec3.mulMat4(i.v1().idx(), m4), F32Vec3.mulMat4(i.v2().idx(), m4), i.rgb());
    }


    static F32Triangle3D addVec3(F32Triangle3D i, int v3) {
        if (i.rgb() == 0){
            throw new IllegalStateException("i.rgb() == 0");
        }
        return f32Triangle3DPool.of(F32Vec3.addVec3(i.v0().idx(), v3), F32Vec3.addVec3(i.v1().idx(), v3), F32Vec3.addVec3(i.v2().idx(), v3), i.rgb());
    }

     static F32Triangle3D mulScaler(F32Triangle3D i, float s) {
        return f32Triangle3DPool.of(F32Vec3.mulScaler(i.v0().idx(), s), F32Vec3.mulScaler(i.v1().idx(), s), F32Vec3.mulScaler(i.v2().idx(), s), i.rgb());
    }

     static F32Triangle3D addScaler(F32Triangle3D i, float s) {
        return f32Triangle3DPool.of(F32Vec3.addScaler(i.v0().idx(), s), F32Vec3.addScaler(i.v1().idx(), s), F32Vec3.addScaler(i.v2().idx(), s), i.rgb());
    }

     static int getCentre(F32Triangle3D i){// the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

     static int getVectorSum(F32Triangle3D i){// the sum of all the vertices
        return F32Vec3.addVec3(F32Vec3.addVec3(i.v0().idx(), i.v1().idx()),i.v2().idx());
    }


     static int normal(F32Triangle3D i) {
        int line1Vec3 = F32Vec3.subVec3(i.v1().idx(), i.v0().idx());
        int line2Vec3 = F32Vec3.subVec3(i.v2().idx(),  i.v0().idx());
        return F32Vec3.crossProd(line1Vec3, line2Vec3);
    }

     static int normalSumOfSquares(F32Triangle3D i) {
        int normalVec3 = normal(i);
        return F32Vec3.divScaler(normalVec3,  F32Vec3.sumOfSquares(normalVec3));
    }

    record F32Triangle3DImpl(F32Triangle3D id) implements F32Triangle3D {
        public static List<F32Triangle3DImpl> all() {
                List<F32Triangle3DImpl> all = new ArrayList<>();
                for (int t = 0; t < f32Triangle3DPool.count; t++) {
                    all.add(new F32Triangle3DImpl(f32Triangle3DPool.idx(t))/*Pool.Idx.of(t))*/);
                }
                return all;
            }

            public F32Triangle3DImpl mul(F32Matrix4x4 m) {
                if (id.rgb() == 0){
                    throw new RuntimeException("rgb() is 0");
                }
            return new F32Triangle3DImpl(mulMat4(id, m));
            }

            public F32Triangle3DImpl add(F32Vec3.F32Vec3Impl v) {
                if (id.rgb() == 0){
                    throw new RuntimeException("rgb() is 0");
                }
                return new F32Triangle3DImpl(addVec3(id, v.id().idx()));

            }

            public F32Vec3.F32Vec3Impl normalSumOfSquares() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normalSumOfSquares(id)));
            }

            public F32Vec3.F32Vec3Impl normal() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normal(id)));
            }

            public F32Vec3.F32Vec3Impl v0() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v0().idx()));
            }

            public F32Vec3.F32Vec3Impl v1() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v1().idx()));
            }

            public F32Vec3.F32Vec3Impl v2() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(id.v2().idx()));
            }

            public F32Triangle3DImpl mul(float s) {
                return new F32Triangle3DImpl(mulScaler(id, s));
            }

            public F32Triangle3DImpl add(float s) {
                return new F32Triangle3DImpl(addScaler(id, s));
            }

            public int rgb() {
                return id.rgb();
            }

        @Override
        public int idx() {
           return id.idx();
        }

        public  F32Vec3.F32Vec3Impl center() {
                return new F32Vec3.F32Vec3Impl(F32Vec3.f32Vec3Pool.idx(getCentre(id)));
            }
        }
}
