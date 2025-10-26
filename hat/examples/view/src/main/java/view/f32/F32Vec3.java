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

public interface F32Vec3 {
     int X = 0;
     int Y = 1;
     int Z = 2;

     class Pool extends FloatPool {
        Pool(int stride, int max) {
           super(stride,max);
        }
    }
     Pool pool = new Pool(3, 90000);
    interface Impl extends F32Vec3 {
        Pool.Idx id();
    }

    static int createVec3(float x, float y, float z) {
        pool.entries[pool.count * pool.stride + X] = x;
        pool.entries[pool.count * pool.stride + Y] = y;
        pool.entries[pool.count * pool.stride + Z] = z;
        return pool.count++;
    }


    // return another vec3 after multiplying by m4
    // we pad this vec3 to vec 4 with '1' as w
    // we normalize the result

    static int mulMat4(int i, F32Matrix4x4.Pool.Idx m4) {
        i *= pool.stride;
        m4  = F32Matrix4x4.Pool.Idx.of(m4.idx()* F32Matrix4x4.pool.stride);
        int o = createVec3(
                pool.entries[i + X] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X0Y0] + pool.entries[i + Y] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X0Y1] + pool.entries[i + Z] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X0Y2] + 1f * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X0Y3],
                pool.entries[i + X] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X1Y0] + pool.entries[i + Y] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X1Y1] + pool.entries[i + Z] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X1Y2] + 1f * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X1Y3],
                pool.entries[i + X] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X2Y0] + pool.entries[i + Y] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X2Y1] + pool.entries[i + Z] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X2Y2] + 1f * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X2Y3]
        );

        float w = pool.entries[i + X] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X3Y0] + pool.entries[i + Y] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X3Y1] + pool.entries[i + Z] * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X3Y2] + 1 * F32Matrix4x4.pool.entries[m4.idx() + F32Matrix4x4.X3Y3];
        if (w != 0.0) {
            o = F32Vec3.divScaler(o, w);
        }
        return o;
    }

    static int mulScaler(int i, float s) {
        i *= pool.stride;
        return createVec3(pool.entries[i + X] * s, pool.entries[i + Y] * s, pool.entries[i + Z] * s);
    }

    static int addScaler(int i, float s) {
        i *= pool.stride;
        return createVec3(pool.entries[i + X] + s, pool.entries[i + Y] + s, pool.entries[i + Z] + s);
    }

    static int divScaler(int i, float s) {
        i *= pool.stride;
        return createVec3(pool.entries[i + X] / s, pool.entries[i + Y] / s, pool.entries[i + Z] / s);
    }

     static int addVec3(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec3(pool.entries[lhs + X] + pool.entries[rhs + X], pool.entries[lhs + Y] + pool.entries[rhs + Y], pool.entries[lhs + Z] + pool.entries[rhs + Z]);
    }

    static int subVec3(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec3(pool.entries[lhs + X] - pool.entries[rhs + X], pool.entries[lhs + Y] - pool.entries[rhs + Y], pool.entries[lhs + Z] - pool.entries[rhs + Z]);
    }
     static int mulVec3(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec3(pool.entries[lhs + X] * pool.entries[rhs + X], pool.entries[lhs + Y] * pool.entries[rhs + Y], pool.entries[lhs + Z] * pool.entries[rhs + Z]);
    }
    static int divVec3(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec3(pool.entries[lhs + X] / pool.entries[rhs + X], pool.entries[lhs + Y] / pool.entries[rhs + Y], pool.entries[lhs + Z] / pool.entries[rhs + Z]);
    }


    static float sumOfSquares(int i) {
        i *= pool.stride;
        return pool.entries[i + X] * pool.entries[i + X] + pool.entries[i + Y] * pool.entries[i + Y] + pool.entries[i + Z] * pool.entries[i + Z];
    }
     static float sumOf(int i) {
        i *= pool.stride;
        return pool.entries[i + X]  + pool.entries[i + Y] + pool.entries[i + Z] ;
    }

    static float hypot(int i) {
        return (float) Math.sqrt(sumOfSquares(i));
    }

    /*
        lhs= | 1|   rhs= | 2|
             | 3|        | 7|
             | 4|        |-5|

        lhs xprod rhs = | x  y  z| =  | 3  4|x - | 1  4|y  | 1  3|
                        | 1  3  4|    | 7 -5|    | 2 -5|   | 2  7|
                        | 2  7 -5|

                      = (-15-28)x - (-5 -8)y + (7 - 6)z

                      = -43x - (-13)y +1z
                      = -43x + 14y +z

     */

    static int crossProd(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec3(
                pool.entries[lhs + Y] * pool.entries[rhs + Z] - pool.entries[lhs + Z] * pool.entries[rhs + X],
                pool.entries[lhs + Z] * pool.entries[rhs + X] - pool.entries[lhs + X] * pool.entries[rhs + Z],
                pool.entries[lhs + X] * pool.entries[rhs + Y] - pool.entries[lhs + Y] * pool.entries[rhs + X]);

    }

    /*
        lhs= | 1|   rhs= | 2|
             | 3|        | 7|
             | 4|        |-5|

        lhs0*rhs0 + lhs1*rhs1 + lhs2*rhs2
         1  * 2   +  3  * 7   +  4  *-5

            3     +    21     +   -20

                       4

     */

    static float dotProd(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;

       return pool.entries[lhs + X] * pool.entries[rhs + X] + pool.entries[lhs + Y] * pool.entries[rhs + Y] +
               pool.entries[lhs + Z] * pool.entries[rhs + Z];

    }

    static String asString(int i) {
        i *= pool.stride;
        return pool.entries[i + X] + "," + pool.entries[i + Y] + "," + pool.entries[i + Z];
    }

     static float getX(int i) {
        i *= pool.stride;
        return pool.entries[i + X];
    }

     static float getY(int i) {
        i *= pool.stride;
        return pool.entries[i + Y];
    }

     static float getZ(int i) {
        i *= pool.stride;
        return pool.entries[i + Z];
    }

    class vec3 implements Impl{
        view.f32.Pool.Idx id;
        vec3(int id){
            this.id = view.f32.Pool.Idx.of(id);
        }
        @Override
        public  view.f32.Pool.Idx id(){
            return id;
        }


        private vec3(float x, float y, float z) {
            this(createVec3(x,y,z));
        }
        public static vec3 of(float x, float y, float z){
            return new vec3(x,y,z);
        }

        public vec3 sub(vec3 v) {
            return new vec3(subVec3(id.idx(), v.id.idx()));
        }
        public vec3 add(vec3 v) {
            return new vec3(addVec3(id.idx(), v.id.idx()));
        }
        public vec3 mul(vec3 v) {
            return new vec3(mulVec3(id.idx(), v.id.idx()));
        }

        public float dotProd(vec3 v){
            return F32Vec3.dotProd(id.idx(), v.id.idx());
        }
        public float sumOf(){
            return F32Vec3.sumOf(id.idx());
        }

        public float x() {
            return getX(id.idx());
        }
        public float y() {
            return getY(id.idx());
        }
        public float z() {
            return getZ(id.idx());
        }
    }
}
