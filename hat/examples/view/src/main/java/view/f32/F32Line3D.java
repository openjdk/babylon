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

public interface  F32Line3D {

    int V0 = 0;
    int V1 = 1;
    int RGB = 2;

    class Pool extends IndexPool{
        Pool( int max) {
            super(1,max);
        }
    }
   Pool pool = new Pool(2000);


    static int fillLine3D(int i, int v0, int v1, int rgb) {
        i *= pool.stride;
        pool.entries[i + V0] = v0;
        pool.entries[i + V1] = v1;
        pool.entries[i + RGB] = rgb;
        return i;
    }

     static int createLine3D(int v0, int v1, int rgb) {
        fillLine3D(pool.count, v0, v1,rgb);
        return pool.count++;
    }

    static String asString(int i) {
        i *= pool.stride;
        return F32Vec3.asString(pool.entries[i + V0]) + " -> " + F32Vec3.asString(pool.entries[i + V1]) + " =" + String.format("0x%8x", pool.entries[i + RGB]);
    }



     static int addVec3(int i, int v3) {
        i *= pool.stride;
        return createLine3D(F32Vec3.addVec3(pool.entries[i + V0], v3), F32Vec3.addVec3(pool.entries[i + V1], v3), pool.entries[i + RGB]);
    }

     static int mulScaler(int i, float s) {
        i *= pool.stride;
        return createLine3D(F32Vec3.mulScaler(pool.entries[i + V0], s), F32Vec3.mulScaler(pool.entries[i + V1], s), pool.entries[i + RGB]);
    }

     static int addScaler(int i, float s) {
        i *= pool.stride;
        return createLine3D(F32Vec3.addScaler(pool.entries[i + V0], s), F32Vec3.addScaler(pool.entries[i + V1], s), pool.entries[i + RGB]);
    }

     static int getCentre(int i){
        // the average of all the vertices
        return F32Vec3.divScaler(getVectorSum(i), 3);
    }

     static int getVectorSum(int i){
        // the sum of all the vertices
        return F32Vec3.addVec3(getV0(i), getV1(i));
    }


     static int getV0(int i) {
        i *= pool.stride;
        return F32Line3D.pool.entries[i + F32Line3D.V0];
    }

     static int getV1(int i) {
        i *= pool.stride;
        return F32Line3D.pool.entries[i + F32Line3D.V1];
    }


     static int getRGB(int i) {
        i *= pool.stride;
        return F32Line3D.pool.entries[i + F32Line3D.RGB];
    }

}
