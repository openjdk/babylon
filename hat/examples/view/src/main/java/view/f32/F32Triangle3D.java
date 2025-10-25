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

public class F32Triangle3D {
    static final int V0 = 0;
    static final int V1 = 1;
    static final int V2 = 2;
    static final int RGB = 3;

     /*
       v0----v1         v0----v2
        \    |           \    |
         \   |            \   |
          \  |    --->     \  |
           \ |              \ |
            \|               \|
             v2               v1
   */

    public static Pool.Idx rewind(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        int temp =pool.entries[i.idx(V1)];
        pool.entries[i.idx(V1)] =  pool.entries[i.idx(V2)];
        pool.entries[i.idx(V2)] = temp;
        return i;
    }

    public static class Pool extends IndexPool {
        Pool(int max) {
            super(4, max);
        }
        int v0(Idx idx){
            return idx.idx(V0);
        }
        int v1(Idx idx){
            return idx.idx(V1);
        }
        int v2(Idx idx){
            return idx.idx(V2);
        }
        int rgb(Idx idx){
            return idx.idx(RGB);
        }
    }

    public static Pool pool = new Pool(12800);

    static Pool.Idx fillTriangle3D(Pool.Idx i, int v0, int v1, int v2, int rgb) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        pool.entries[pool.v0(i)] = v0;
        pool.entries[pool.v1(i)] = v1;
        pool.entries[pool.v2(i)] = v2;
        pool.entries[pool.rgb(i)] = rgb;
        return i;
    }

    public static Pool.Idx createTriangle3D(int v0, int v1, int v2, int rgb) {
        fillTriangle3D(Pool.Idx.of(pool.count), v0, v1, v2, rgb);
        return Pool.Idx.of(pool.count++);
    }

    static String asString(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Vec3.asString(pool.entries[i.idx(V0)]) + " -> " + F32Vec3.asString(pool.entries[i.idx(V1)]) + " -> " + F32Vec3.asString(pool.entries[i.idx(V2)]) + " =" + String.format("0x%8x", pool.entries[i.idx(RGB)]);
    }

    public static Pool.Idx mulMat4(Pool.Idx i, F32Mat4.Pool.Idx  m4) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return createTriangle3D(F32Vec3.mulMat4(pool.entries[i.idx(V0)], m4.idx()), F32Vec3.mulMat4(pool.entries[i.idx(V1)], m4.idx()), F32Vec3.mulMat4(pool.entries[i.idx(V2)], m4.idx()), pool.entries[i.idx(RGB)]);
    }

    public static Pool.Idx addVec3(Pool.Idx i, int v3) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return createTriangle3D(F32Vec3.addVec3(pool.entries[i.idx(V0)], v3), F32Vec3.addVec3(pool.entries[i.idx(V1)], v3), F32Vec3.addVec3(pool.entries[i.idx(V2)], v3), pool.entries[i.idx(RGB)]);
    }

    public static Pool.Idx mulScaler(Pool.Idx i, float s) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return createTriangle3D(F32Vec3.mulScaler(pool.entries[i.idx(V0)], s), F32Vec3.mulScaler(pool.entries[i.idx(V1)], s), F32Vec3.mulScaler(pool.entries[i.idx(V2)], s), pool.entries[i.idx(RGB)]);
    }

    public static Pool.Idx addScaler(Pool.Idx i, float s) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return createTriangle3D(F32Vec3.addScaler(pool.entries[i.idx(V0)], s), F32Vec3.addScaler(pool.entries[i.idx(V1)], s), F32Vec3.addScaler(pool.entries[i.idx(V2)], s), pool.entries[i.idx(RGB)]);
    }

    public static int getCentre(Pool.Idx i){
        // the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

    public static int getVectorSum(Pool.Idx i){
        // the sum of all the vertices
        return F32Vec3.addVec3(F32Vec3.addVec3(getV0(i), getV1(i)), getV2(i));
    }


    public static int getV0(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx() + F32Triangle3D.V0];
    }

    public static int getV1(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.V1)];
    }

    public static int getV2(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.V2)];
    }

    public static int getRGB(Pool.Idx i) {
        i = Pool.Idx.of(i.idx() * pool.stride);
        return F32Triangle3D.pool.entries[i.idx(F32Triangle3D.RGB)];
    }


    public static int normal(Pool.Idx i) {

        int v0 = F32Triangle3D.getV0(i);
        int v1 = F32Triangle3D.getV1(i);
        int v2 = F32Triangle3D.getV2(i);

        int line1Vec3 = F32Vec3.subVec3(v1, v0);
        int line2Vec3 = F32Vec3.subVec3(v2, v0);

        return F32Vec3.crossProd(line1Vec3, line2Vec3);
    }

    public static int normalSumOfSquares(Pool.Idx i) {
        int normalVec3 = normal(i);
        return F32Vec3.divScaler(normalVec3,  F32Vec3.sumOfSquares(normalVec3));
    }


}
