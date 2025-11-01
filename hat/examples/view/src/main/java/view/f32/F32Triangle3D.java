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
    List<F32Triangle3D> arr = new ArrayList<>();
    static void reset(int marked) {
        F32Triangle3D.f32Triangle3DPool.count = marked;
        while (arr.size()>marked){
            arr.removeLast();
        }
    }

    int V0 = 0;
    int V1 = 1;
    int V2 = 2;
    int RGB = 3;

    static  int v0(F32Triangle3DPool.Idx<F32Triangle3DPool> idx){
        return idx.idx(V0);
    }
    static int v1(F32Triangle3DPool.Idx<F32Triangle3DPool> idx){
        return idx.idx(V1);
    }
    static int v2(F32Triangle3DPool.Idx<F32Triangle3DPool> idx){
        return idx.idx(V2);
    }
    static int rgb(F32Triangle3DPool.Idx<F32Triangle3DPool> idx){
        return idx.idx(RGB);
    }
    class F32Triangle3DPool extends IndexPool<F32Triangle3DPool> {
        F32Triangle3DPool(int max) {
            super(4, max);
        }

        @Override
        Idx<F32Triangle3DPool> idx(int idx) {
            return new Idx<>(this, idx);
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

     static F32Triangle3DPool.Idx<F32Triangle3DPool> rewind(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        int temp = f32Triangle3DPool.entries[i.idx(V1)];
        f32Triangle3DPool.entries[v1(i)] =  f32Triangle3DPool.entries[v2(i)];
        f32Triangle3DPool.entries[v2(i)] = temp;
        return i;
    }

     static F32Triangle3DPool.Idx<F32Triangle3DPool> of(int v0, int v1, int v2, int rgb) {
         var i = f32Triangle3DPool.idx(f32Triangle3DPool.count * f32Triangle3DPool.stride);//pool.count * pool.stride
         f32Triangle3DPool.entries[v0(i)] = v0;
         f32Triangle3DPool.entries[v1(i)] = v1;
         f32Triangle3DPool.entries[v2(i)] = v2;
         f32Triangle3DPool.entries[rgb(i)] = rgb;
        return f32Triangle3DPool.idx(f32Triangle3DPool.count++);//pool.count++
    }

    static String asString(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return F32Vec3.asString(f32Triangle3DPool.entries[v0(i)]) + " -> " + F32Vec3.asString(f32Triangle3DPool.entries[v1(i)]) + " -> " + F32Vec3.asString(f32Triangle3DPool.entries[v2(i)]) + " =" + String.format("0x%8x", f32Triangle3DPool.entries[rgb(i)]);
    }

     static F32Triangle3DPool.Idx<F32Triangle3DPool> mulMat4(F32Triangle3DPool.Idx<F32Triangle3DPool> i, F32Matrix4x4.F32Matrix4x4Pool.Idx  m4) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return of(F32Vec3.mulMat4(f32Triangle3DPool.entries[v0(i)], m4), F32Vec3.mulMat4(f32Triangle3DPool.entries[v1(i)], m4), F32Vec3.mulMat4(f32Triangle3DPool.entries[v2(i)], m4), f32Triangle3DPool.entries[rgb(i)]);
    }

     static F32Triangle3DPool.Idx<F32Triangle3DPool> addVec3(F32Triangle3DPool.Idx<F32Triangle3DPool> i, int v3) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return of(F32Vec3.addVec3(f32Triangle3DPool.entries[v0(i)], v3), F32Vec3.addVec3(f32Triangle3DPool.entries[v1(i)], v3), F32Vec3.addVec3(f32Triangle3DPool.entries[v2(i)], v3), f32Triangle3DPool.entries[rgb(i)]);
    }

     static F32Triangle3DPool.Idx<F32Triangle3DPool> mulScaler(F32Triangle3DPool.Idx<F32Triangle3DPool> i, float s) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return of(F32Vec3.mulScaler(f32Triangle3DPool.entries[v0(i)], s), F32Vec3.mulScaler(f32Triangle3DPool.entries[v1(i)], s), F32Vec3.mulScaler(f32Triangle3DPool.entries[v2(i)], s), f32Triangle3DPool.entries[rgb(i)]);
    }

     static F32Triangle3DPool.Idx<F32Triangle3DPool> addScaler(F32Triangle3DPool.Idx<F32Triangle3DPool> i, float s) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return of(F32Vec3.addScaler(f32Triangle3DPool.entries[v0(i)], s), F32Vec3.addScaler(f32Triangle3DPool.entries[v1(i)], s), F32Vec3.addScaler(f32Triangle3DPool.entries[v2(i)], s), f32Triangle3DPool.entries[rgb(i)]);
    }

     static int getCentre(F32Triangle3DPool.Idx<F32Triangle3DPool> i){
        // the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

     static int getVectorSum(F32Triangle3DPool.Idx<F32Triangle3DPool> i){
        // the sum of all the vertices
        return F32Vec3.addVec3(F32Vec3.addVec3(getV0(i), getV1(i)), getV2(i));
    }


     static int getV0(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return F32Triangle3D.f32Triangle3DPool.entries[i.idx() + F32Triangle3D.V0];
    }

     static int getV1(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return F32Triangle3D.f32Triangle3DPool.entries[i.idx(F32Triangle3D.V1)];
    }

     static int getV2(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return F32Triangle3D.f32Triangle3DPool.entries[i.idx(F32Triangle3D.V2)];
    }

     static int getRGB(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        i = f32Triangle3DPool.idx(i.idx() * f32Triangle3DPool.stride);//i.idx() * pool.stride
        return F32Triangle3D.f32Triangle3DPool.entries[i.idx(F32Triangle3D.RGB)];
    }


     static int normal(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {

        int v0 = F32Triangle3D.getV0(i);
        int v1 = F32Triangle3D.getV1(i);
        int v2 = F32Triangle3D.getV2(i);

        int line1Vec3 = F32Vec3.subVec3(v1, v0);
        int line2Vec3 = F32Vec3.subVec3(v2, v0);

        return F32Vec3.crossProd(line1Vec3, line2Vec3);
    }

     static int normalSumOfSquares(F32Triangle3DPool.Idx<F32Triangle3DPool> i) {
        int normalVec3 = normal(i);
        return F32Vec3.divScaler(normalVec3,  F32Vec3.sumOfSquares(normalVec3));
    }

    interface Impl extends F32Triangle3D {
        F32Triangle3DPool.Idx<F32Triangle3DPool> id();
    }

    record tri(F32Triangle3DPool.Idx<F32Triangle3DPool> id) implements Impl {

        public static List<tri> all() {
                List<tri> all = new ArrayList<>();
                for (int t = 0; t < f32Triangle3DPool.count; t++) {
                    all.add(new tri(f32Triangle3DPool.idx(t))/*Pool.Idx.of(t))*/);
                }
                return all;
            }

            public tri mul(F32Matrix4x4.Impl m) {
                return new tri(mulMat4(id, m.id()));
            }

            public tri add(F32Vec3.vec3 v) {
                return new tri(addVec3(id, v.id().idx()));

            }

            public F32Vec3.vec3 normalSumOfSquares() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normalSumOfSquares(id)));
            }

            public F32Vec3.vec3 normal() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(F32Triangle3D.normal(id)));
            }

            public F32Vec3.vec3 v0() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(getV0(id)));
            }

            public F32Vec3.vec3 v1() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(getV1(id)));
            }

            public F32Vec3.vec3 v2() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(getV2(id)));
            }

            public tri mul(float s) {
                return new tri(mulScaler(id, s));
            }

            public tri add(float s) {
                return new tri(addScaler(id, s));
            }

            public int rgb() {
                return getRGB(id);
            }

            public F32Vec3.vec3 center() {
                return new F32Vec3.vec3(F32Vec3.f32Vec3Pool.idx(getCentre(id)));
            }
        }
}
