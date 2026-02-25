/*
* Copyright (c) 2025-2026, Oracle and/or its affiliates. All rights reserved.
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

import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;

public interface vec4 extends IfaceValue.vec{
    Shape shape = Shape.of(JavaType.FLOAT, 4);
    float x();
    float y();
    float z();
    float w();

    /*
    This allows us to add this type to interface mapped segments
    */
    interface Field extends vec4{
        void x(float x);
        void y(float y);
        void z(float z);
        void w(float w);
        default vec4 of(float x, float y, float z, float w){
            x(x);
            y(y);
            z(z);
            w(w);
            return this;
        }

        default vec4 of(vec4 vec4){
            of(vec4.x(), vec4.y(), vec4.z(), vec4.w());
            return this;
        }


    }

    static vec4 vec4(float x, float y, float z, float w){
        record Impl(float x, float y, float z, float w) implements vec4{

        }

        return new Impl(x, y, z, w);
    }

    static vec4 vec4(float scalar){
        return vec4(scalar, scalar, scalar, scalar);
    }

    static vec4 mul(float xl, float xr, float yl, float yr, float zl, float zr, float wl, float wr){
        return vec4(xl*xr, yl*yr, zl*zr, wl*wr);
    }

    static vec4 mul(vec4 l, vec4 r){
        return mul(l.x(), r.x(), l.y(), r.y(), l.z(), r.z(), l.w(), r.w());
    }

    static vec4 mul(float l, vec4 r){
        return mul(l, r.x(), l, r.y(), l, r.z(), l, r.w());
    }

    static vec4 mul(vec4 l, float r){
        return mul(l.x(), r, l.y(), r, l.z(), r, l.w(), r);
    }

    static vec4 div(float xl, float xr, float yl, float yr, float zl, float zr, float wl, float wr){
        return vec4(xl/xr, yl/yr, zl/zr, wl/wr);
    }

    static vec4 div(vec4 l, vec4 r){
        return div(l.x(), r.x(), l.y(), r.y(), l.z(), r.z(), l.w(), r.w());
    }

    static vec4 div(float l, vec4 r){
        return div(l, r.x(), l, r.y(), l, r.z(), l, r.w());
    }

    static vec4 div(vec4 l, float r){
        return div(l.x(), r, l.y(), r, l.z(), r, l.w(), r);
    }

    static vec4 add(float xl, float xr, float yl, float yr, float zl, float zr, float wl, float wr){
        return vec4(xl+xr, yl+yr, zl+zr, wl+wr);
    }

    static vec4 add(vec4 l, vec4 r){
        return add(l.x(), r.x(), l.y(), r.y(), l.z(), r.z(), l.w(), r.w());
    }

    static vec4 add(float l, vec4 r){
        return add(l, r.x(), l, r.y(), l, r.z(), l, r.w());
    }

    static vec4 add(vec4 l, float r){
        return add(l.x(), r, l.y(), r, l.z(), r, l.w(), r);
    }

    static vec4 sub(float xl, float xr, float yl, float yr, float zl, float zr, float wl, float wr){
        return vec4(xl-xr, yl-yr, zl-zr, wl-wr);
    }

    static vec4 sub(vec4 l, vec4 r){
        return sub(l.x(), r.x(), l.y(), r.y(), l.z(), r.z(), l.w(), r.w());
    }

    static vec4 sub(float l, vec4 r){
        return sub(l, r.x(), l, r.y(), l, r.z(), l, r.w());
    }

    static vec4 sub(vec4 l, float r){
        return sub(l.x(), r, l.y(), r, l.z(), r, l.w(), r);
    }

    static vec4 pow(vec4 l, vec4 r){
        return vec4(F32.pow(l.x(), r.x()), F32.pow(l.y(), r.y()), F32.pow(l.z(), r.z()), F32.pow(l.w(), r.w()));
    }

    static vec4 pow(float l, vec4 r){
        return vec4(F32.pow(l, r.x()), F32.pow(l, r.y()), F32.pow(l, r.z()), F32.pow(l, r.w()));
    }

    static vec4 pow(vec4 l, float r){
        return vec4(F32.pow(l.x(), r), F32.pow(l.y(), r), F32.pow(l.z(), r), F32.pow(l.w(), r));
    }

    static vec4 min(vec4 l, vec4 r){
        return vec4(F32.min(l.x(), r.x()), F32.min(l.y(), r.y()), F32.min(l.z(), r.z()), F32.min(l.w(), r.w()));
    }

    static vec4 min(float l, vec4 r){
        return vec4(F32.min(l, r.x()), F32.min(l, r.y()), F32.min(l, r.z()), F32.min(l, r.w()));
    }

    static vec4 min(vec4 l, float r){
        return vec4(F32.min(l.x(), r), F32.min(l.y(), r), F32.min(l.z(), r), F32.min(l.w(), r));
    }

    static vec4 max(vec4 l, vec4 r){
        return vec4(F32.max(l.x(), r.x()), F32.max(l.y(), r.y()), F32.max(l.z(), r.z()), F32.max(l.w(), r.w()));
    }

    static vec4 max(float l, vec4 r){
        return vec4(F32.max(l, r.x()), F32.max(l, r.y()), F32.max(l, r.z()), F32.max(l, r.w()));
    }

    static vec4 max(vec4 l, float r){
        return vec4(F32.max(l.x(), r), F32.max(l.y(), r), F32.max(l.z(), r), F32.max(l.w(), r));
    }

    static vec4 floor(vec4 v){
        return vec4(F32.floor(v.x()), F32.floor(v.y()), F32.floor(v.z()), F32.floor(v.w()));
    }

    static vec4 round(vec4 v){
        return vec4(F32.round(v.x()), F32.round(v.y()), F32.round(v.z()), F32.round(v.w()));
    }

    static vec4 fract(vec4 v){
        return vec4(F32.fract(v.x()), F32.fract(v.y()), F32.fract(v.z()), F32.fract(v.w()));
    }

    static vec4 abs(vec4 v){
        return vec4(F32.abs(v.x()), F32.abs(v.y()), F32.abs(v.z()), F32.abs(v.w()));
    }

    static vec4 log(vec4 v){
        return vec4(F32.log(v.x()), F32.log(v.y()), F32.log(v.z()), F32.log(v.w()));
    }

    static vec4 sin(vec4 v){
        return vec4(F32.sin(v.x()), F32.sin(v.y()), F32.sin(v.z()), F32.sin(v.w()));
    }

    static vec4 cos(vec4 v){
        return vec4(F32.cos(v.x()), F32.cos(v.y()), F32.cos(v.z()), F32.cos(v.w()));
    }

    static vec4 tan(vec4 v){
        return vec4(F32.tan(v.x()), F32.tan(v.y()), F32.tan(v.z()), F32.tan(v.w()));
    }

    static vec4 sqrt(vec4 v){
        return vec4(F32.sqrt(v.x()), F32.sqrt(v.y()), F32.sqrt(v.z()), F32.sqrt(v.w()));
    }

    static vec4 inversesqrt(vec4 v){
        return vec4(F32.inversesqrt(v.x()), F32.inversesqrt(v.y()), F32.inversesqrt(v.z()), F32.inversesqrt(v.w()));
    }

    static vec4 neg(vec4 v){
        return vec4(0f-v.x(), 0f-v.y(), 0f-v.z(), 0f-v.w());
    }

    static vec4 vec4(vec2 vec2, float z, float w){
        return vec4(vec2.x(), vec2.y(), z, w);
    }

    static vec4 vec4(vec3 vec3, float w){
        return vec4(vec3.x(), vec3.y(), vec3.z(), w);
    }

    static float dot(vec4 l, vec4 r){
        return l.x()*r.x()+l.y()*r.y()+l.z()*r.z()+l.w()*r.w();
    }

    static float sumOfSquares(vec4 v){
        return dot(v, v);
    }

    static float length(vec4 v){
        return F32.sqrt(sumOfSquares(v));
    }

    static vec4 clamp(vec4 v, float min, float max){
        return vec4(F32.clamp(v.x(), min, max), F32.clamp(v.y(), min, max), F32.clamp(v.z(), min, max), F32.clamp(v.w(), min, max));
    }

    static vec4 normalize(vec4 v){
        float lenSq = sumOfSquares(v);

        return (lenSq > 0f)?(mul(v, F32.inversesqrt(lenSq))):(vec4(0f));
    }

    static vec4 reflect(vec4 l, vec4 r){
        // lhs - 2f * dot(rhs, lhs) * rhs
        return vec4.sub(l, mul(mul(r, l), 2f));
    }

    static float distance(vec4 l, vec4 r){
        float dx = r.x()-l.x();
        float dy = r.y()-l.y();
        float dz = r.z()-l.z();
        float dw = r.w()-l.w();
        return F32.sqrt(dx*dx+dy*dy+dz*dz+dw*dw);
    }

    static vec4 smoothstep(vec4 e0, vec4 e1, vec4 v){
        return vec4(
            F32.smoothstep(e0.x(), e1.x(), v.x()),
            F32.smoothstep(e0.y(), e1.y(), v.y()),
            F32.smoothstep(e0.z(), e1.z(), v.z()),
            F32.smoothstep(e0.w(), e1.w(), v.w())
        );
    }

    static vec4 mix(vec4 l, vec4 r, float v){
        return vec4(
            F32.mix(l.x(), r.x(), v),
            F32.mix(l.y(), r.y(), v),
            F32.mix(l.z(), r.z(), v),
            F32.mix(l.w(), r.w(), v)
        );
    }


}