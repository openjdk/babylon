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

public class F32Vec3 {
    static final int SIZE = 3;

    static final int X = 0;
    static final int Y = 1;
    static final int Z = 2;
    public static class Pool {
        public final int max;
        public int count = 0;
        public final float entries[];
        Pool(int max) {
            this.max = max;
            this.entries = new float[max * SIZE];
        }
    }
    public static Pool pool = new Pool(6000);

    public static int createVec3(float x, float y, float z) {
        pool.entries[pool.count * SIZE + X] = x;
        pool.entries[pool.count * SIZE + Y] = y;
        pool.entries[pool.count * SIZE + Z] = z;
        return pool.count++;
    }


    // return another vec3 after multiplying by m4
    // we pad this vec3 to vec 4 with '1' as w
    // we normalize the result
    static int mulMat4(int i, int m4) {
        i *= SIZE;
        m4 *= F32Mat4.SIZE;
        int o = createVec3(
                pool.entries[i + X] * F32Mat4.pool.entries[m4 + F32Mat4.X0Y0] + pool.entries[i + Y] * F32Mat4.pool.entries[m4 + F32Mat4.X0Y1] + pool.entries[i + Z] * F32Mat4.pool.entries[m4 + F32Mat4.X0Y2] + 1f * F32Mat4.pool.entries[m4 + F32Mat4.X0Y3],
                pool.entries[i + X] * F32Mat4.pool.entries[m4 + F32Mat4.X1Y0] + pool.entries[i + Y] * F32Mat4.pool.entries[m4 + F32Mat4.X1Y1] + pool.entries[i + Z] * F32Mat4.pool.entries[m4 + F32Mat4.X1Y2] + 1f * F32Mat4.pool.entries[m4 + F32Mat4.X1Y3],
                pool.entries[i + X] * F32Mat4.pool.entries[m4 + F32Mat4.X2Y0] + pool.entries[i + Y] * F32Mat4.pool.entries[m4 + F32Mat4.X2Y1] + pool.entries[i + Z] * F32Mat4.pool.entries[m4 + F32Mat4.X2Y2] + 1f * F32Mat4.pool.entries[m4 + F32Mat4.X2Y3]
        );

        float w = pool.entries[i + X] * F32Mat4.pool.entries[m4 + F32Mat4.X3Y0] + pool.entries[i + Y] * F32Mat4.pool.entries[m4 + F32Mat4.X3Y1] + pool.entries[i + Z] * F32Mat4.pool.entries[m4 + F32Mat4.X3Y2] + 1 * F32Mat4.pool.entries[m4 + F32Mat4.X3Y3];
        if (w != 0.0) {
            o = F32Vec3.divScaler(o, w);
        }
        return o;
    }

    static int mulScaler(int i, float s) {
        i *= SIZE;
        return createVec3(pool.entries[i + X] * s, pool.entries[i + Y] * s, pool.entries[i + Z] * s);
    }

    static int addScaler(int i, float s) {
        i *= SIZE;
        return createVec3(pool.entries[i + X] + s, pool.entries[i + Y] + s, pool.entries[i + Z] + s);
    }

    static int divScaler(int i, float s) {
        i *= SIZE;
        return createVec3(pool.entries[i + X] / s, pool.entries[i + Y] / s, pool.entries[i + Z] / s);
    }

    public static int addVec3(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec3(pool.entries[lhs + X] + pool.entries[rhs + X], pool.entries[lhs + Y] + pool.entries[rhs + Y], pool.entries[lhs + Z] + pool.entries[rhs + Z]);
    }

    public static int subVec3(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec3(pool.entries[lhs + X] - pool.entries[rhs + X], pool.entries[lhs + Y] - pool.entries[rhs + Y], pool.entries[lhs + Z] - pool.entries[rhs + Z]);
    }
    public static int mulVec3(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec3(pool.entries[lhs + X] * pool.entries[rhs + X], pool.entries[lhs + Y] * pool.entries[rhs + Y], pool.entries[lhs + Z] * pool.entries[rhs + Z]);
    }
    static int divVec3(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec3(pool.entries[lhs + X] / pool.entries[rhs + X], pool.entries[lhs + Y] / pool.entries[rhs + Y], pool.entries[lhs + Z] / pool.entries[rhs + Z]);
    }


    static float sumOfSquares(int i) {
        i *= SIZE;
        return pool.entries[i + X] * pool.entries[i + X] + pool.entries[i + Y] * pool.entries[i + Y] + pool.entries[i + Z] * pool.entries[i + Z];
    }
    public static float sumOf(int i) {
        i *= SIZE;
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
        lhs *= SIZE;
        rhs *= SIZE;
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
        lhs *= SIZE;
        rhs *= SIZE;

       return pool.entries[lhs + X] * pool.entries[rhs + X] + pool.entries[lhs + Y] * pool.entries[rhs + Y] +
               pool.entries[lhs + Z] * pool.entries[rhs + Z];

    }

    static String asString(int i) {
        i *= SIZE;
        return pool.entries[i + X] + "," + pool.entries[i + Y] + "," + pool.entries[i + Z];
    }

    public static float getX(int i) {
        i *= SIZE;
        return pool.entries[i + X];
    }

    public static float getY(int i) {
        i *= SIZE;
        return pool.entries[i + Y];
    }

    public static float getZ(int i) {
        i *= SIZE;
        return pool.entries[i + Z];
    }
}
