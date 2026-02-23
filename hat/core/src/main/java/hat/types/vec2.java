/* Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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
        package hat.types;

// Auto generated DO NOT EDIT

import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import optkl.IfaceValue;
import hat.types.F32;
import static hat.types.F32.*;

public interface vec2 extends IfaceValue.vec{
    Shape shape=Shape.of(JavaType.FLOAT, 2);

    float x();
    float y();

    AtomicInteger count=new AtomicInteger(0);
    AtomicBoolean collect=new AtomicBoolean(false);
    /*
    This allows us to add this type to interface mapped segments
    */
    interface Field extends vec2{
        void x(float x);
        void y(float y);
        default vec2 of(float x, float y){
            x(x);
            y(y);
            return this;
        }

        default vec2 of(vec2 vec2){
            of(vec2.x(), vec2.y());
            return this;
        }


    }

    static vec2 vec2(float x, float y){
        record Impl(float x, float y) implements vec2{

        }
        // Uncomment to collect stats
        //    if (collect.get())count.getAndIncrement();
        return new Impl(x, y);
    }

    static vec2 vec2(float scalar){
        return vec2(scalar, scalar);
    }

    static vec2 mul(float xl, float xr, float yl, float yr){
        return vec2(xl*xr, yl*yr);
    }

    static vec2 mul(vec2 l, vec2 r){
        return mul(l.x(), r.x(), l.y(), r.y());
    }

    static vec2 mul(float l, vec2 r){
        return mul(l, r.x(), l, r.y());
    }

    static vec2 mul(vec2 l, float r){
        return mul(l.x(), r, l.y(), r);
    }

    static vec2 sub(float xl, float xr, float yl, float yr){
        return vec2(xl-xr, yl-yr);
    }

    static vec2 sub(vec2 l, vec2 r){
        return sub(l.x(), r.x(), l.y(), r.y());
    }

    static vec2 sub(float l, vec2 r){
        return sub(l, r.x(), l, r.y());
    }

    static vec2 sub(vec2 l, float r){
        return sub(l.x(), r, l.y(), r);
    }

    static vec2 add(float xl, float xr, float yl, float yr){
        return vec2(xl+xr, yl+yr);
    }

    static vec2 add(vec2 l, vec2 r){
        return add(l.x(), r.x(), l.y(), r.y());
    }

    static vec2 add(float l, vec2 r){
        return add(l, r.x(), l, r.y());
    }

    static vec2 add(vec2 l, float r){
        return add(l.x(), r, l.y(), r);
    }

    static vec2 div(float xl, float xr, float yl, float yr){
        return vec2(xl/xr, yl/yr);
    }

    static vec2 div(vec2 l, vec2 r){
        return div(l.x(), r.x(), l.y(), r.y());
    }

    static vec2 div(float l, vec2 r){
        return div(l, r.x(), l, r.y());
    }

    static vec2 div(vec2 l, float r){
        return div(l.x(), r, l.y(), r);
    }

    static vec2 pow(vec2 l, vec2 r){
        return vec2(F32.pow(l.x(), r.x()), F32.pow(l.y(), r.y()));
    }

    static vec2 min(vec2 l, vec2 r){
        return vec2(F32.min(l.x(), r.x()), F32.min(l.y(), r.y()));
    }

    static vec2 max(vec2 l, vec2 r){
        return vec2(F32.max(l.x(), r.x()), F32.max(l.y(), r.y()));
    }

    static vec2 floor(vec2 v){
        return vec2(F32.floor(v.x()), F32.floor(v.y()));
    }

    static vec2 round(vec2 v){
        return vec2(F32.round(v.x()), F32.round(v.y()));
    }

    static vec2 fract(vec2 v){
        return vec2(F32.fract(v.x()), F32.fract(v.y()));
    }

    static vec2 abs(vec2 v){
        return vec2(F32.abs(v.x()), F32.abs(v.y()));
    }

    static vec2 log(vec2 v){
        return vec2(F32.log(v.x()), F32.log(v.y()));
    }

    static vec2 sin(vec2 v){
        return vec2(F32.sin(v.x()), F32.sin(v.y()));
    }

    static vec2 cos(vec2 v){
        return vec2(F32.cos(v.x()), F32.cos(v.y()));
    }

    static vec2 tan(vec2 v){
        return vec2(F32.tan(v.x()), F32.tan(v.y()));
    }

    static vec2 sqrt(vec2 v){
        return vec2(F32.sqrt(v.x()), F32.sqrt(v.y()));
    }

    static vec2 inversesqrt(vec2 v){
        return vec2(F32.inversesqrt(v.x()), F32.inversesqrt(v.y()));
    }

    static vec2 neg(vec2 v){
        return vec2(0f-v.x(), 0f-v.y());
    }

    static float dot(vec2 l, vec2 r){
        return l.x()*r.x()+l.y()*r.y();
    }

    static float sumOfSquares(vec2 v){
        return dot(v, v);
    }

    static float length(vec2 v){
        return F32.sqrt(sumOfSquares(v));
    }


    /* safe to copy to here */

    static vec2 xy(vec3 vec3) {return vec2(vec3.x(), vec3.y());}
    static vec2 xz(vec3 vec3) {return vec2(vec3.x(), vec3.z());}
    static vec2 yz(vec3 vec3) {return vec2(vec3.y(), vec3.z());}


    static vec2 mul(vec2 l, mat2 rhs) {return vec2(l.x()*rhs._00()+l.x()+rhs._01(),l.y()*rhs._10()+l.y()+rhs._11());}

    static vec2 mod(vec2 v, float r){return vec2(F32.mod(v.x(),r),F32.mod(v.y(),r));}

    static vec2 max(float x,vec2 rhs){return vec2(F32.max(x,rhs.x()), F32.max(x,rhs.y()));}
    static vec2 max(vec2 lhs, float y){return vec2(F32.max(lhs.x(),y), F32.max(lhs.y(),y));}

    static vec2 mix(vec2 lhs,vec2 rhs, vec2 a){return vec2(F32.mix(lhs.x(),rhs.x(),a.x()), F32.mix(lhs.y(),rhs.y(),a.y()));}

    static vec2 normalize(vec2 vec2){
        float lenSq = sumOfSquares(vec2);
        return (lenSq > 0.0f)?mul(vec2, F32.inversesqrt(lenSq)):vec2(0.0f); // Handle zero-length case
    }



   /*
   We should be able to use vec16 for mat4


            float16 mat4_mul(float16 A, float16 B) {
                float16 C;

                // We compute C row by row
                // Each row of C is the sum of the rows of B scaled by the components of A

                // Row 0
                C.s0123 = A.s0 * B.s0123 + A.s1 * B.s4567 + A.s2 * B.s89ab + A.s3 * B.scdef;
                // Row 1
                C.s4567 = A.s4 * B.s0123 + A.s5 * B.s4567 + A.s6 * B.s89ab + A.s7 * B.scdef;
                // Row 2
                C.s89ab = A.s8 * B.s0123 + A.s9 * B.s4567 + A.sa * B.s89ab + A.sb * B.scdef;
                // Row 3
                C.scdef = A.sc * B.s0123 + A.sd * B.s4567 + A.se * B.s89ab + A.sf * B.scdef;

                return C;
            }


            #define TS 16 // Tile Size

            __kernel void mat4_mul_tiled(__global const float16* A,\s
                                         __global const float16* B,\s
                                         __global float16* C,
                                         const int Width) { // Width in terms of float16 units

                // Local memory for tiles of float16 matrices
                __local float16 tileA[TS][TS];
                __local float16 tileB[TS][TS];

                int row = get_local_id(1);
                int col = get_local_id(0);
                int globalRow = get_global_id(1);
                int globalCol = get_global_id(0);

                float16 accumulated = (float16)(0.0f);

                // Loop over tiles
                for (int t = 0; t < (Width / TS); t++) {

                    // Cooperative Load: Each thread loads one float16 into local memory
                    tileA[row][col] = A[globalRow * Width + (t * TS + col)];
                    tileB[row][col] = B[(t * TS + row) * Width + globalCol];

                    // Synchronize to ensure the tile is fully loaded
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // Compute partial product for this tile
                    for (int k = 0; k < TS; k++) {
                        accumulated = mat4_mul_core(accumulated, tileA[row][k], tileB[k][col]);
                    }

                    // Synchronize before loading the next tile
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                // Write result to global memory
                C[globalRow * Width + globalCol] = accumulated;
            }
    */

}
