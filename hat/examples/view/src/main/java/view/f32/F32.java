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

import view.f32.pool.F32x3TrianglePool;

public interface F32 {
    F32x4x4.Factory f32x4x4Factory();
    F32x3.Factory f32x3Factory();
    F32x2.Factory f32x2Factory();
    F32x3Triangle.Factory f32x3TriangleFactory();
    F32x2Triangle.Factory f32x2TriangleFactory();
    //  https://stackoverflow.com/questions/28075743/how-do-i-compose-a-rotation-matrix-with-human-readable-angles-from-scratch/28084380#28084380
     default F32x4x4 mul(F32x4x4 lhs, F32x4x4 rhs) {
        return f32x4x4Factory().of(
                lhs.x0y0() * rhs.x0y0() + lhs.x1y0() * rhs.x0y1() + lhs.x2y0() * rhs.x0y2() + lhs.x3y0() * rhs.x0y3(),
                lhs.x0y0() * rhs.x1y0() + lhs.x1y0() * rhs.x1y1() + lhs.x2y0() * rhs.x1y2() + lhs.x3y0() * rhs.x1y3(),
                lhs.x0y0() * rhs.x2y0() + lhs.x1y0() * rhs.x2y1() + lhs.x2y0() * rhs.x2y2() + lhs.x3y0() * rhs.x2y3(),
                lhs.x0y0() * rhs.x3y0() + lhs.x1y0() * rhs.x3y1() + lhs.x2y0() * rhs.x3y2() + lhs.x3y0() * rhs.x3y3(),

                lhs.x0y1() * rhs.x0y0() + lhs.x1y1() * rhs.x0y1() + lhs.x2y1() * rhs.x0y2() + lhs.x3y1() * rhs.x0y3(),
                lhs.x0y1() * rhs.x1y0() + lhs.x1y1() * rhs.x1y1() + lhs.x2y1() * rhs.x1y2() + lhs.x3y1() * rhs.x1y3(),
                lhs.x0y1() * rhs.x2y0() + lhs.x1y1() * rhs.x2y1() + lhs.x2y1() * rhs.x2y2() + lhs.x3y1() * rhs.x2y3(),
                lhs.x0y1() * rhs.x3y0() + lhs.x1y1() * rhs.x3y1() + lhs.x2y1() * rhs.x3y2() + lhs.x3y1() * rhs.x3y3(),

                lhs.x0y2() * rhs.x0y0() + lhs.x1y2() * rhs.x0y1() + lhs.x2y2() * rhs.x0y2() + lhs.x3y2() * rhs.x0y3(),
                lhs.x0y2() * rhs.x1y0() + lhs.x1y2() * rhs.x1y1() + lhs.x2y2() * rhs.x1y2() + lhs.x3y2() * rhs.x1y3(),
                lhs.x0y2() * rhs.x2y0() + lhs.x1y2() * rhs.x2y1() + lhs.x2y2() * rhs.x2y2() + lhs.x3y2() * rhs.x2y3(),
                lhs.x0y2() * rhs.x3y0() + lhs.x1y2() * rhs.x3y1() + lhs.x2y2() * rhs.x3y2() + lhs.x3y2() * rhs.x3y3(),

                lhs.x0y3() * rhs.x0y0() + lhs.x1y3() * rhs.x0y1() + lhs.x2y3() * rhs.x0y2() + lhs.x3y3() * rhs.x0y3(),
                lhs.x0y3() * rhs.x1y0() + lhs.x1y3() * rhs.x1y1() + lhs.x2y3() * rhs.x1y2() + lhs.x3y3() * rhs.x1y3(),
                lhs.x0y3() * rhs.x2y0() + lhs.x1y3() * rhs.x2y1() + lhs.x2y3() * rhs.x2y2() + lhs.x3y3() * rhs.x2y3(),
                lhs.x0y3() * rhs.x3y0() + lhs.x1y3() * rhs.x3y1() + lhs.x2y3() * rhs.x3y2() + lhs.x3y3() * rhs.x3y3()

        );
    }

    //https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8
    // record Transformation(F32Matrix4x4Pool id) implements Impl {
     default F32x4x4 transformation(float x, float y, float z) {
        return f32x4x4Factory().of(
                1f, 0f, 0f, 0f,
                0f, 1f, 0f, 0f,
                0f, 0f, 1f, 0f,
                x, y, z, 1f
        );
    }

     default F32x4x4 transformation(float v) {
        return transformation(v, v, v);
    }

     default F32x4x4 scale(float x, float y, float z) {
        return f32x4x4Factory().of(
                x, 0f, 0f, 0f,
                0f, y, 0f, 0f,
                0f, 0f, z, 0f,
                0f, 0f, 0f, 1f

        );
    }

     default F32x4x4 scale(float v) {
        return scale(v, v, v);
    }

     default F32x4x4 rotX(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32x4x4Factory().of(
                1f, 0f, 0f, 0f,
                0f, cosTheta, -sinTheta, 0f,
                0f, sinTheta, cosTheta, 0f,
                0f, 0f, 0f, 1f

        );
    }

     default F32x4x4 rotZ(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32x4x4Factory().of(
                cosTheta, sinTheta, 0f, 0f,
                -sinTheta, cosTheta, 0f, 0f,
                0f, 0f, 1f, 0f,
                0f, 0f, 0f, 1f
        );
    }

     default F32x4x4 rotY(float thetaRadians) {
        float sinTheta = (float) Math.sin(thetaRadians);
        float cosTheta = (float) Math.cos(thetaRadians);
        return f32x4x4Factory().of(
                cosTheta, 0f, sinTheta, 0f,
                0f, 1f, 0f, 0f,
                -sinTheta, 0f, cosTheta, 0f,
                0f, 0f, 0f, 1f
        );
    }

     default F32x4x4 rot(float thetaX, float thetaY, float thetaZ) {
        return mul(mul(rotX(thetaX), rotY(thetaY)), rotZ(thetaZ));
    }

    // https://medium.com/swlh/understanding-3d-matrix-transforms-with-pixijs-c76da3f8bd8



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

     default F32x4x4 projection(float width, float height, float near, float far, float fieldOfViewDeg) {
        float aspectRatio = height / width;
        float fieldOfViewRadians = (float) (1.0f / Math.tan((fieldOfViewDeg * 0.5f) / 180.0 * Math.PI));
        return f32x4x4Factory().of(
                aspectRatio * fieldOfViewRadians, 0f, 0f, 0f,
                0f, fieldOfViewRadians, 0f, 0f,
                0f, 0f, far / (far - near), (-far * near) / (far - near),
                    0f, 0f, (-far * near) / (far - near), 0f);

    }

   static float side(float x, float y, F32x2 v0, F32x2 v1) {
         return    (v1.y() - v0.y() * (x - v0.x()) + (-v1.x() + v0.x()) * (y - v0.y()));
    }

     /*
              V0                V0
              |  \              |  \
              |    \            |    \        P2
              |  P1  \          |      \
              V1------V0        V1------V0


Barycentric coordinate allows to express new p coordinates as a linear combination of p1, p2, p3.
 More precisely, it defines 3 scalars a, b, c such that :

x = a * x1 + b * x2  + c * x3
y = a * y1 + b * y2 + c * y3
a + b + c = 1


a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
c = 1 - a - b

p lies in T if and only if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1
*/

   static boolean inside(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2) {
        var denominator = ((y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2));
        var a = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denominator;
        var b = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denominator;
        var c = 1 - a - b;
        return 0 <= a && a <= 1 && 0 <= b && b <= 1 && 0 <= c && c <= 1;
    }

   static boolean inside(float x, float y, F32x2 v0, F32x2 v1, F32x2 v2) {
        return inside(x,y, v0.x(), v0.y(), v1.x(), v1.y(),v2.x(), v2.y());
    }

    static boolean inside(float x, float y, F32x2Triangle tri) {
       return inside(x,y, tri.v0(), tri.v1(),tri.v2());
    }

    static boolean onLine(float x, float y, F32x2 v0, F32x2 v1, float deltaSquare) {
        float dxl = v1.x() - v0.x();
        float dyl = v1.y() - v0.y();;
        float cross = (x - v0.x()) * dyl - (y - v0.y()) * dxl;
        if ((cross * cross) < deltaSquare) {
            if (dxl * dxl >= dyl * dyl)
                return dxl > 0 ? v0.x() <= x && x <= v1.x() : v1.x() <= x && x <= v0.x();
            else
                return dyl > 0 ? v0.y() <= y && y <= v1.y() : v1.y() <= y && y <= v0.y();
        } else {
            return false;
        }
    }

    static boolean onEdge(float x, float y, F32x2Triangle tri) {
        return onLine(x, y, tri.v0(), tri.v1(), F32x2Triangle.deltaSquare)
                || onLine(x, y,tri.v1(),tri.v2(), F32x2Triangle.deltaSquare)
                || onLine(x, y, tri.v2(),tri.v0(), F32x2Triangle.deltaSquare);
    }

    static   boolean useRgb(boolean filled, float x, float y, F32x2Triangle tri){
        return filled? inside(x,y,tri): onEdge(x,y,tri);
    }

    static   int rgb(boolean filled, float x, float y, F32x2Triangle tri, int rgb){
        return useRgb(filled,x,y,tri)? tri.rgb() : rgb;
    }

    /*
      v0----v1         v0----v2
       \    |           \    |
        \   |            \   |
         \  |    --->     \  |
          \ |              \ |
           \|               \|
            v2               v1
  */
    static F32x3Triangle rewind(F32x3Triangle i) {
        var temp = i.v1();
        ((F32x3TrianglePool.PoolEntry) i).pool().f32x3Entries[((F32x3TrianglePool.PoolEntry) i).v1Idx()] = i.v2();
        ((F32x3TrianglePool.PoolEntry) i).pool().f32x3Entries[((F32x3TrianglePool.PoolEntry) i).v2Idx()] = temp;
        return i;
    }

   default F32x3Triangle mul(F32x3Triangle i, F32x4x4 m4) {
        return f32x3TriangleFactory().of(mul(i.v0(), m4), mul(i.v1(), m4), mul(i.v2(), m4), i.rgb());
    }

    default F32x3Triangle add(F32x3Triangle i, F32x3 v3) {
        return f32x3TriangleFactory().of(add(i.v0(), v3), add(i.v1(), v3), add(i.v2(), v3), i.rgb());
    }

    default F32x3Triangle mul(F32x3Triangle i, float s) {
        return f32x3TriangleFactory().of(mul(i.v0(), s), mul(i.v1(), s), mul(i.v2(), s), i.rgb());
    }

   default F32x3Triangle add(F32x3Triangle i, float s) {
        return f32x3TriangleFactory().of(add(i.v0(), s), add(i.v1(), s), add(i.v2(), s), i.rgb());
    }

    default F32x3 centre(F32x3Triangle i) {// the average of all the vertices
        return div(getVectorSum(i), 3);
    }

    default F32x3 getVectorSum(F32x3Triangle i) {// the sum of all the vertices
        return add(add(i.v0(), i.v1()), i.v2());
    }

    default F32x3 normal(F32x3Triangle i) {
        return crossProd(sub(i.v1(), i.v0()), sub(i.v2(), i.v0()));
    }

    default F32x3 normalSumOfSquares(F32x3Triangle i) {
        return div(normal(i), sumOfSq(normal(i)));
    }

    /* return another vec3 after multiplying by m4
     we pad this vec3 to vec 4 with '1' as w
     we normalize the result
     */

    default F32x3 mul(F32x3 f32x3, F32x4x4 m) {
        F32x3 o = f32x3Factory().of(
                f32x3.x() * m.x0y0() + f32x3.y() * m.x0y1() + f32x3.z() * m.x0y2() + 1f * m.x0y3(),
                f32x3.x() * m.x1y0() + f32x3.y() * m.x1y1() + f32x3.z() * m.x1y2() + 1f * m.x1y3(),
                f32x3.x() * m.x2y0() + f32x3.y() * m.x2y1() + f32x3.z() * m.x2y2() + 1f * m.x2y3()
        );
        float w = f32x3.x() * m.x3y0() + f32x3.y() * m.x3y1() + f32x3.z() * m.x3y2() + 1f * m.x3y3();
      //  if (w!=0.0) {
            o = div(o, w);
       // }
        return o;
    }

    default F32x3 mul(F32x3 i, float s) {
        return f32x3Factory().of(i.x() * s, i.y() * s, i.z() * s);
    }

    default F32x3 add(F32x3 i, float s) {

        return f32x3Factory().of(i.x() + s, i.y() + s, i.z() + s);
    }

    default F32x3 div(F32x3 i, float s) {
        if (s==0){
            return i;
        }
        return f32x3Factory().of(i.x() / s, i.y() / s, i.z() / s);
    }

    default F32x3 add(F32x3 lhs, F32x3 rhs) {
        return f32x3Factory().of(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());
    }

    default F32x3 sub(F32x3 lhs, F32x3 rhs) {
        return f32x3Factory().of(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
    }

    default F32x3 mul(F32x3 lhs, F32x3 rhs) {
        return f32x3Factory().of(lhs.x() * rhs.x(), lhs.y() * rhs.y(), lhs.z() * rhs.z());
    }

    default F32x3 div(F32x3 lhs, F32x3 rhs) {
        return f32x3Factory().of(lhs.x() / rhs.x(), lhs.y() / rhs.y(), lhs.z() / rhs.z());
    }

    default float sumOfSq(F32x3 i) {
        return i.x() * i.x() + i.y() * i.y() + i.z() * i.z();
    }

    default float sumOf(F32x3 i) {
        return i.x() + i.y() + i.z();
    }

    default float hypot(F32x3 i) {
        return (float) Math.sqrt(sumOfSq(i));
    }

    default F32x3 crossProd(F32x3 lhs, F32x3 rhs) {
        return f32x3Factory().of(
                lhs.y() * rhs.z() - lhs.z() * rhs.x(),
                lhs.z() * rhs.x() - lhs.x() * rhs.z(),
                lhs.x() * rhs.y() - lhs.y() * rhs.x());

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
    default float dotProd(F32x3 lhs, F32x3 rhs) {
        return lhs.x() * rhs.x() + lhs.y() * rhs.y() + lhs.z() * rhs.z();
    }
}
