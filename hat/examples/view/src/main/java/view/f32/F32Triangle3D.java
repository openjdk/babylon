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
    int V0 = 0;
    int V1 = 1;
    int V2 = 2;
    int RGB = 3;

    static  int v0(view.f32.Pool.Idx idx){
        return idx.idx(V0);
    }
    static int v1(view.f32.Pool.Idx idx){
        return idx.idx(V1);
    }
    static int v2(view.f32.Pool.Idx idx){
        return idx.idx(V2);
    }
    static int rgb(view.f32.Pool.Idx idx){
        return idx.idx(RGB);
    }
    class Pool extends IndexPool {
        Pool(int max) {
            super(4, max);
        }
    }
    Pool pool = new Pool(12800);

     /*
       v0----v1         v0----v2
        \    |           \    |
         \   |            \   |
          \  |    --->     \  |
           \ |              \ |
            \|               \|
             v2               v1
   */

     static Pool.Idx rewind(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        int temp =pool.entries[i.idx(V1)];
        pool.entries[v1(i)] =  pool.entries[v2(i)];
        pool.entries[v2(i)] = temp;
        return i;
    }

    static Pool.Idx fillTriangle3D(Pool.Idx i, int v0, int v1, int v2, int rgb) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        pool.entries[v0(i)] = v0;
        pool.entries[v1(i)] = v1;
        pool.entries[v2(i)] = v2;
        pool.entries[rgb(i)] = rgb;
        return i;
    }

     static Pool.Idx of(int v0, int v1, int v2, int rgb) {
        fillTriangle3D(Pool.Idx.of(pool.count), v0, v1, v2, rgb);
        return Pool.Idx.of(pool.count++);
    }

    static String asString(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Vec3.asString(pool.entries[v0(i)]) + " -> " + F32Vec3.asString(pool.entries[v1(i)]) + " -> " + F32Vec3.asString(pool.entries[v2(i)]) + " =" + String.format("0x%8x", pool.entries[rgb(i)]);
    }

     static Pool.Idx mulMat4(Pool.Idx i, F32Matrix4x4.Pool.Idx  m4) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return of(F32Vec3.mulMat4(pool.entries[v0(i)], m4), F32Vec3.mulMat4(pool.entries[v1(i)], m4), F32Vec3.mulMat4(pool.entries[v2(i)], m4), pool.entries[rgb(i)]);
    }

     static Pool.Idx addVec3(Pool.Idx i, int v3) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return of(F32Vec3.addVec3(pool.entries[v0(i)], v3), F32Vec3.addVec3(pool.entries[v1(i)], v3), F32Vec3.addVec3(pool.entries[v2(i)], v3), pool.entries[rgb(i)]);
    }

     static Pool.Idx mulScaler(Pool.Idx i, float s) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return of(F32Vec3.mulScaler(pool.entries[v0(i)], s), F32Vec3.mulScaler(pool.entries[v1(i)], s), F32Vec3.mulScaler(pool.entries[v2(i)], s), pool.entries[rgb(i)]);
    }

     static Pool.Idx addScaler(Pool.Idx i, float s) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return of(F32Vec3.addScaler(pool.entries[v0(i)], s), F32Vec3.addScaler(pool.entries[v1(i)], s), F32Vec3.addScaler(pool.entries[v2(i)], s), pool.entries[rgb(i)]);
    }

     static int getCentre(Pool.Idx i){
        // the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

     static int getVectorSum(Pool.Idx i){
        // the sum of all the vertices
        return F32Vec3.addVec3(F32Vec3.addVec3(getV0(i), getV1(i)), getV2(i));
    }


     static int getV0(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx() + F32Triangle3D.V0];
    }

     static int getV1(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.V1)];
    }

     static int getV2(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.V2)];
    }

     static int getRGB(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.RGB)];
    }


     static int normal(Pool.Idx i) {

        int v0 = F32Triangle3D.getV0(i);
        int v1 = F32Triangle3D.getV1(i);
        int v2 = F32Triangle3D.getV2(i);

        int line1Vec3 = F32Vec3.subVec3(v1, v0);
        int line2Vec3 = F32Vec3.subVec3(v2, v0);

        return F32Vec3.crossProd(line1Vec3, line2Vec3);
    }

     static int normalSumOfSquares(Pool.Idx i) {
        int normalVec3 = normal(i);
        return F32Vec3.divScaler(normalVec3,  F32Vec3.sumOfSquares(normalVec3));
    }
    interface Impl extends F32Triangle3D {
        Pool.Idx id();
    }

      class tri implements Impl {
        private Pool.Idx id;
        public view.f32.Pool.Idx id(){
            return id;
        }
        public tri( Pool.Idx id) {
            this.id = id;
        }

        public static List<tri> all() {
            List<tri> all = new ArrayList<>();
            for (int t = 0; t < pool.count; t++) {
                all.add(new tri(Pool.Idx.of(t)));
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
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(F32Triangle3D.normalSumOfSquares(id)));
        }

        public F32Vec3.vec3 normal() {
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(F32Triangle3D.normal(id)));
        }

        public F32Vec3.vec3 v0() {
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(getV0(id)));
        }

        public F32Vec3.vec3 v1() {
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(getV1(id)));
        }

        public F32Vec3.vec3 v2() {
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(getV2(id)));
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
            return new F32Vec3.vec3(F32Vec3.Pool.Idx.of(getCentre(id)));
        }
    }
}
