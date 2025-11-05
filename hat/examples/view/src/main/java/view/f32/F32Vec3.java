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
    float x();
    float y();
    float z();
    int idx();
    default String asString() {
        return x() + "," + y() + "," + z();
    }


     class F32Vec3Pool extends FloatPool<F32Vec3Pool> {
              public static int X = 0;
              public static  int Y = 1;
              public static  int Z = 2;
         public record Idx(F32Vec3Pool pool, int idx) implements Pool.Idx<F32Vec3Pool>,F32Vec3{
             private int xIdx(){return pool.stride * idx+X;}
             @Override public float x(){return pool.entries[xIdx()];}
             private int yIdx(){return pool.stride * idx+Y;}
             @Override public float y(){return pool.entries[yIdx()];}
             private int  zIdx(){return pool.stride * idx+Z;}
             @Override public float z(){return pool.entries[zIdx()];}
         }
        F32Vec3Pool( int max) {
           super(3,max);
        }
         @Override
         Idx idx(int idx) {
             return new Idx(this, idx);
         }
          F32Vec3Pool.Idx of(float x, float y, float z) {
             F32Vec3Pool.Idx i =idx(count++);
             entries[i.xIdx()] = x;
             entries[i.yIdx()] = y;
             entries[i.zIdx()] = z;
             return i;
         }
     }
    F32Vec3Pool f32Vec3Pool = new F32Vec3Pool( 90000);

    // return another vec3 after multiplying by m4
    // we pad this vec3 to vec 4 with '1' as w
    // we normalize the result


    static F32Vec3 mulMat4(F32Vec3 f32Vec3, F32Matrix4x4 m) {
        F32Vec3 o = f32Vec3Pool.of(
                f32Vec3.x() * m.x0y0() + f32Vec3.y() * m.x0y1() + f32Vec3.z() * m.x0y2() + 1f * m.x0y3(),
                f32Vec3.x() * m.x1y0() + f32Vec3.y() * m.x1y1() + f32Vec3.z()  * m.x1y2() + 1f * m.x1y3(),
                f32Vec3.x() * m.x2y0() + f32Vec3.y() * m.x2y1() + f32Vec3.z()  * m.x2y2() + 1f * m.x2y3()
        );
        float w = f32Vec3.x() * m.x3y0() + f32Vec3.y() * m.x3y1() + f32Vec3.z()  * m.x3y2() + 1 * m.x3y3();
        if (w != 0.0) {
            o = F32Vec3.divScaler(o, w);
        }
        return o;
    }

    static F32Vec3 mulScaler(F32Vec3 i, float s) {
        return f32Vec3Pool.of(i.x() * s, i.y() * s, i.z() * s);
    }

    static F32Vec3 addScaler(F32Vec3 i, float s) {

        return f32Vec3Pool.of(i.x() + s, i.y() + s, i.z() + s);
    }

    static F32Vec3 divScaler(F32Vec3 i, float s) {

        return f32Vec3Pool.of(i.x() / s, i.y() / s, i.z() / s);
    }

    static F32Vec3 addVec3(F32Vec3 lhs, F32Vec3 rhs) {
        return f32Vec3Pool.of(lhs.x() + rhs.x(), lhs.y() + rhs.y(), lhs.z() + rhs.z());  }

    static F32Vec3 subVec3(F32Vec3 lhs, F32Vec3 rhs) {
        return f32Vec3Pool.of(lhs.x() - rhs.x(), lhs.y() - rhs.y(), lhs.z() - rhs.z());
    }
    static F32Vec3 mulVec3(F32Vec3 lhs, F32Vec3 rhs) {
        return f32Vec3Pool.of(lhs.x() * rhs.x(), lhs.y() * rhs.y(), lhs.z() * rhs.z());
    }

    static F32Vec3 divVec3(F32Vec3 lhs, F32Vec3 rhs) {
        return f32Vec3Pool.of(lhs.x() / rhs.x(), lhs.y() / rhs.y(), lhs.z() / rhs.z());
    }

    static float sumOfSquares(F32Vec3 i) {
        return i.x() * i.x() + i.y() * i.y() + i.z() * i.z();
    }
     static float sumOf(F32Vec3 i) {
        return i.x()  + i.y() + i.z() ;
    }

    static float hypot(F32Vec3 i) {
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

    static F32Vec3 crossProd(F32Vec3 lhs, F32Vec3 rhs) {
        return f32Vec3Pool.of(
                lhs.y()  * rhs.z() - lhs.z() * rhs.x(),
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

    static float dotProd(F32Vec3 lhs, F32Vec3 rhs) {
       return lhs.x() * rhs.x() + lhs.y() * rhs.y() +lhs.z() * rhs.z();
    }

     static float getX(int i) {
        i *= f32Vec3Pool.stride;
        return f32Vec3Pool.entries[i +F32Vec3Pool.X];
    }

     static float getY(int i) {
        i *= f32Vec3Pool.stride;
        return f32Vec3Pool.entries[i +F32Vec3Pool.Y];
    }

     static float getZ(int i) {
        i *= f32Vec3Pool.stride;
        return f32Vec3Pool.entries[i +F32Vec3Pool.Z];
    }
    record F32Vec3Impl(F32Vec3 id) implements F32Vec3 {
        public static F32Vec3Impl of(F32Vec3 id){
            return new F32Vec3Impl(id);
        }
        public static F32Vec3 of(float x, float y, float z){
            return of(f32Vec3Pool.idx(F32Vec3.f32Vec3Pool.of(x,y,z).idx()));
        }

        public F32Vec3Impl sub(F32Vec3 v) {
            return F32Vec3Impl.of(f32Vec3Pool.idx(subVec3(id, v).idx()));//of(Pool.Idx.of(subVec3(id.idx(), v.id.idx())));
        }

        public F32Vec3Impl add(F32Vec3Impl v) {
            return F32Vec3Impl.of(f32Vec3Pool.idx(addVec3(id,v.id).idx()));//Pool.Idx.of(addVec3(id.idx(), v.id.idx())));
        }


        public float dotProd(F32Vec3Impl v){
            return F32Vec3.dotProd(id, v.id);
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
        public int idx(){return id.idx();}
    }
}
