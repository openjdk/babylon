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

public class F32Mat4 {
    static final int SIZE = 16;
    static final int X0Y0 = 0;
    static final int X1Y0 = 1;
    static final int X2Y0 = 2;
    static final int X3Y0 = 3;
    static final int X0Y1 = 4;
    static final int X1Y1 = 5;
    static final int X2Y1 = 6;
    static final int X3Y1 = 7;
    static final int X0Y2 = 8;
    static final int X1Y2 = 9;
    static final int X2Y2 = 10;
    static final int X3Y2 = 11;
    static final int X0Y3 = 12;
    static final int X1Y3 = 13;
    static final int X2Y3 = 14;
    static final int X3Y3 = 15;

    public static class Pool {
        public final int max;
        public int count = 0;
        public final float entries[];
        Pool(int max) {
            this.max = max;
            this.entries = new float[max * SIZE];
        }
    }
    
    public static Pool pool = new Pool(100);

    public static int createMat4(float x0y0, float x1y0, float x2y0, float x3y0,
                          float x0y1, float x1y1, float x2y1, float x3y1,
                          float x0y2, float x1y2, float x2y2, float x3y2,
                          float x0y3, float x1y3, float x2y3, float x3y3) {
        pool.entries[pool.count * SIZE + X0Y0] = x0y0;
        pool.entries[pool.count * SIZE + X1Y0] = x1y0;
        pool.entries[pool.count * SIZE + X2Y0] = x2y0;
        pool.entries[pool.count * SIZE + X3Y0] = x3y0;
        pool.entries[pool.count * SIZE + X0Y1] = x0y1;
        pool.entries[pool.count * SIZE + X1Y1] = x1y1;
        pool.entries[pool.count * SIZE + X2Y1] = x2y1;
        pool.entries[pool.count * SIZE + X3Y1] = x3y1;
        pool.entries[pool.count * SIZE + X0Y2] = x0y2;
        pool.entries[pool.count * SIZE + X1Y2] = x1y2;
        pool.entries[pool.count * SIZE + X2Y2] = x2y2;
        pool.entries[pool.count * SIZE + X3Y2] = x3y2;
        pool.entries[pool.count * SIZE + X0Y3] = x0y3;
        pool.entries[pool.count * SIZE + X1Y3] = x1y3;
        pool.entries[pool.count * SIZE + X2Y3] = x2y3;
        pool.entries[pool.count * SIZE + X3Y3] = x3y3;
        return pool.count++;
    }
  //  https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380
    public static int mulMat4(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createMat4(
                pool.entries[lhs + X0Y0] * pool.entries[rhs + X0Y0] + pool.entries[lhs + X1Y0] * pool.entries[rhs + X0Y1] + pool.entries[lhs + X2Y0] * pool.entries[rhs + X0Y2] + pool.entries[lhs + X3Y0] * pool.entries[rhs + X0Y3],
                pool.entries[lhs + X0Y0] * pool.entries[rhs + X1Y0] + pool.entries[lhs + X1Y0] * pool.entries[rhs + X1Y1] + pool.entries[lhs + X2Y0] * pool.entries[rhs + X1Y2] + pool.entries[lhs + X3Y0] * pool.entries[rhs + X1Y3],
                pool.entries[lhs + X0Y0] * pool.entries[rhs + X2Y0] + pool.entries[lhs + X1Y0] * pool.entries[rhs + X2Y1] + pool.entries[lhs + X2Y0] * pool.entries[rhs + X2Y2] + pool.entries[lhs + X3Y0] * pool.entries[rhs + X2Y3],
                pool.entries[lhs + X0Y0] * pool.entries[rhs + X3Y0] + pool.entries[lhs + X1Y0] * pool.entries[rhs + X3Y1] + pool.entries[lhs + X2Y0] * pool.entries[rhs + X3Y2] + pool.entries[lhs + X3Y0] * pool.entries[rhs + X3Y3],

                pool.entries[lhs + X0Y1] * pool.entries[rhs + X0Y0] + pool.entries[lhs + X1Y1] * pool.entries[rhs + X0Y1] + pool.entries[lhs + X2Y1] * pool.entries[rhs + X0Y2] + pool.entries[lhs + X3Y1] * pool.entries[rhs + X0Y3],
                pool.entries[lhs + X0Y1] * pool.entries[rhs + X1Y0] + pool.entries[lhs + X1Y1] * pool.entries[rhs + X1Y1] + pool.entries[lhs + X2Y1] * pool.entries[rhs + X1Y2] + pool.entries[lhs + X3Y1] * pool.entries[rhs + X1Y3],
                pool.entries[lhs + X0Y1] * pool.entries[rhs + X2Y0] + pool.entries[lhs + X1Y1] * pool.entries[rhs + X2Y1] + pool.entries[lhs + X2Y1] * pool.entries[rhs + X2Y2] + pool.entries[lhs + X3Y1] * pool.entries[rhs + X2Y3],
                pool.entries[lhs + X0Y1] * pool.entries[rhs + X3Y0] + pool.entries[lhs + X1Y1] * pool.entries[rhs + X3Y1] + pool.entries[lhs + X2Y1] * pool.entries[rhs + X3Y2] + pool.entries[lhs + X3Y1] * pool.entries[rhs + X3Y3],

                pool.entries[lhs + X0Y2] * pool.entries[rhs + X0Y0] + pool.entries[lhs + X1Y2] * pool.entries[rhs + X0Y1] + pool.entries[lhs + X2Y2] * pool.entries[rhs + X0Y2] + pool.entries[lhs + X3Y2] * pool.entries[rhs + X0Y3],
                pool.entries[lhs + X0Y2] * pool.entries[rhs + X1Y0] + pool.entries[lhs + X1Y2] * pool.entries[rhs + X1Y1] + pool.entries[lhs + X2Y2] * pool.entries[rhs + X1Y2] + pool.entries[lhs + X3Y2] * pool.entries[rhs + X1Y3],
                pool.entries[lhs + X0Y2] * pool.entries[rhs + X2Y0] + pool.entries[lhs + X1Y2] * pool.entries[rhs + X2Y1] + pool.entries[lhs + X2Y2] * pool.entries[rhs + X2Y2] + pool.entries[lhs + X3Y2] * pool.entries[rhs + X2Y3],
                pool.entries[lhs + X0Y2] * pool.entries[rhs + X3Y0] + pool.entries[lhs + X1Y2] * pool.entries[rhs + X3Y1] + pool.entries[lhs + X2Y2] * pool.entries[rhs + X3Y2] + pool.entries[lhs + X3Y2] * pool.entries[rhs + X3Y3],

                pool.entries[lhs + X0Y3] * pool.entries[rhs + X0Y0] + pool.entries[lhs + X1Y3] * pool.entries[rhs + X0Y1] + pool.entries[lhs + X2Y3] * pool.entries[rhs + X0Y2] + pool.entries[lhs + X3Y3] * pool.entries[rhs + X0Y3],
                pool.entries[lhs + X0Y3] * pool.entries[rhs + X1Y0] + pool.entries[lhs + X1Y3] * pool.entries[rhs + X1Y1] + pool.entries[lhs + X2Y3] * pool.entries[rhs + X1Y2] + pool.entries[lhs + X3Y3] * pool.entries[rhs + X1Y3],
                pool.entries[lhs + X0Y3] * pool.entries[rhs + X2Y0] + pool.entries[lhs + X1Y3] * pool.entries[rhs + X2Y1] + pool.entries[lhs + X2Y3] * pool.entries[rhs + X2Y2] + pool.entries[lhs + X3Y3] * pool.entries[rhs + X2Y3],
                pool.entries[lhs + X0Y3] * pool.entries[rhs + X3Y0] + pool.entries[lhs + X1Y3] * pool.entries[rhs + X3Y1] + pool.entries[lhs + X2Y3] * pool.entries[rhs + X3Y2] + pool.entries[lhs + X3Y3] * pool.entries[rhs + X3Y3]

        );
    }

    static String asString(int i) {
        i *= SIZE;
        return String.format("|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                        "|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                        "|%5.2f, %5.2f, %5.2f, %5.2f|\n" +
                        "|%5.2f, %5.2f, %5.2f, %5.2f|\n",
                pool.entries[i + X0Y0], pool.entries[i + X1Y0], pool.entries[i + X2Y0], pool.entries[i + X3Y0],
                pool.entries[i + X0Y1], pool.entries[i + X1Y1], pool.entries[i + X2Y1], pool.entries[i + X3Y1],
                pool.entries[i + X0Y2], pool.entries[i + X1Y2], pool.entries[i + X2Y2], pool.entries[i + X3Y2],
                pool.entries[i + X0Y3], pool.entries[i + X1Y3], pool.entries[i + X2Y3], pool.entries[i + X3Y3]);
    }

    public static int createProjectionMatrix(float width, float height, float near, float far, float fieldOfViewDeg) {

        // Projection Matrix

        float aspectRatio = height / width;
        float fieldOfViewRadians = (float) (1.0f / Math.tan((fieldOfViewDeg * 0.5f) / 180.0 * Math.PI));

            /*
              https://youtu.be/ih20l3pJoeU?t=973
              https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380^
             --------------------            far
              \                /              ^    ^
               \              /               |    |   far-near
                \            /                |    |
                 \__________/         near    |    v
                                       ^      |
                                       v      v
                     \^/
                   [x,y,z]

            */

        return createMat4(
                aspectRatio * fieldOfViewRadians, 0f, 0f, 0f,
                0f, fieldOfViewRadians, 0f, 0f,
                0f, 0f, far / (far - near), (-far * near) / (far - near),
                0f, 0f, (-far * near) / (far - near), 0f);

    }

    public static int createRotXMat4(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return createMat4(
                1, 0, 0, 0,
                0, cosTheta, -sinTheta, 0,
                0, sinTheta, cosTheta, 0,
                0, 0, 0, 1

        );
    }

    public static int createRotZMat4(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return createMat4(
                cosTheta, sinTheta, 0, 0,
                -sinTheta, cosTheta, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );
    }

    public static int createRotYMat4(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return createMat4(
                cosTheta, 0, sinTheta, 0,
                0, 1, 0, 0,
                -sinTheta, 0, cosTheta, 0,
                0, 0, 0, 1
        );
    }


}
