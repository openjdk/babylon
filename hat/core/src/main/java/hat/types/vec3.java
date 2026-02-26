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

public interface vec3 extends IfaceValue.vec{
    Shape shape = Shape.of(JavaType.FLOAT, 3);
    float x();
    float y();
    float z();

    /*
    This allows us to add this type to interface mapped segments
    */
    interface Field extends vec3{
        void x(float x);
        void y(float y);
        void z(float z);
        default vec3 of(float x, float y, float z){
            x(x);
            y(y);
            z(z);
            return this;
        }

        default vec3 of(vec3 vec3){
            of(vec3.x(), vec3.y(), vec3.z());
            return this;
        }


    }

    static vec3 vec3(float x, float y, float z){
        record Impl(float x, float y, float z) implements vec3{

        }

        return new Impl(x, y, z);
    }

    static vec3 vec3(float scalar){
        return vec3(scalar, scalar, scalar);
    }

    static vec3 div(float xl, float xr, float yl, float yr, float zl, float zr){
        return vec3(xl/xr, yl/yr, zl/zr);
    }

    static vec3 div(vec3 l, vec3 r){
        return div(l.x(), r.x(), l.y(), r.y(), l.z(), r.z());
    }

    static vec3 div(float l, vec3 r){
        return div(l, r.x(), l, r.y(), l, r.z());
    }

    static vec3 div(vec3 l, float r){
        return div(l.x(), r, l.y(), r, l.z(), r);
    }

    static vec3 add(float xl, float xr, float yl, float yr, float zl, float zr){
        return vec3(xl+xr, yl+yr, zl+zr);
    }

    static vec3 add(vec3 l, vec3 r){
        return add(l.x(), r.x(), l.y(), r.y(), l.z(), r.z());
    }

    static vec3 add(float l, vec3 r){
        return add(l, r.x(), l, r.y(), l, r.z());
    }

    static vec3 add(vec3 l, float r){
        return add(l.x(), r, l.y(), r, l.z(), r);
    }

    static vec3 sub(float xl, float xr, float yl, float yr, float zl, float zr){
        return vec3(xl-xr, yl-yr, zl-zr);
    }

    static vec3 sub(vec3 l, vec3 r){
        return sub(l.x(), r.x(), l.y(), r.y(), l.z(), r.z());
    }

    static vec3 sub(float l, vec3 r){
        return sub(l, r.x(), l, r.y(), l, r.z());
    }

    static vec3 sub(vec3 l, float r){
        return sub(l.x(), r, l.y(), r, l.z(), r);
    }

    static vec3 mul(float xl, float xr, float yl, float yr, float zl, float zr){
        return vec3(xl*xr, yl*yr, zl*zr);
    }

    static vec3 mul(vec3 l, vec3 r){
        return mul(l.x(), r.x(), l.y(), r.y(), l.z(), r.z());
    }

    static vec3 mul(float l, vec3 r){
        return mul(l, r.x(), l, r.y(), l, r.z());
    }

    static vec3 mul(vec3 l, float r){
        return mul(l.x(), r, l.y(), r, l.z(), r);
    }

    static vec3 pow(vec3 l, vec3 r){
        return vec3(F32.pow(l.x(), r.x()), F32.pow(l.y(), r.y()), F32.pow(l.z(), r.z()));
    }

    static vec3 pow(float l, vec3 r){
        return vec3(F32.pow(l, r.x()), F32.pow(l, r.y()), F32.pow(l, r.z()));
    }

    static vec3 pow(vec3 l, float r){
        return vec3(F32.pow(l.x(), r), F32.pow(l.y(), r), F32.pow(l.z(), r));
    }

    static vec3 min(vec3 l, vec3 r){
        return vec3(F32.min(l.x(), r.x()), F32.min(l.y(), r.y()), F32.min(l.z(), r.z()));
    }

    static vec3 min(float l, vec3 r){
        return vec3(F32.min(l, r.x()), F32.min(l, r.y()), F32.min(l, r.z()));
    }

    static vec3 min(vec3 l, float r){
        return vec3(F32.min(l.x(), r), F32.min(l.y(), r), F32.min(l.z(), r));
    }

    static vec3 max(vec3 l, vec3 r){
        return vec3(F32.max(l.x(), r.x()), F32.max(l.y(), r.y()), F32.max(l.z(), r.z()));
    }

    static vec3 max(float l, vec3 r){
        return vec3(F32.max(l, r.x()), F32.max(l, r.y()), F32.max(l, r.z()));
    }

    static vec3 max(vec3 l, float r){
        return vec3(F32.max(l.x(), r), F32.max(l.y(), r), F32.max(l.z(), r));
    }

    static vec3 floor(vec3 v){
        return vec3(F32.floor(v.x()), F32.floor(v.y()), F32.floor(v.z()));
    }

    static vec3 round(vec3 v){
        return vec3(F32.round(v.x()), F32.round(v.y()), F32.round(v.z()));
    }

    static vec3 fract(vec3 v){
        return vec3(F32.fract(v.x()), F32.fract(v.y()), F32.fract(v.z()));
    }

    static vec3 abs(vec3 v){
        return vec3(F32.abs(v.x()), F32.abs(v.y()), F32.abs(v.z()));
    }

    static vec3 log(vec3 v){
        return vec3(F32.log(v.x()), F32.log(v.y()), F32.log(v.z()));
    }

    static vec3 sin(vec3 v){
        return vec3(F32.sin(v.x()), F32.sin(v.y()), F32.sin(v.z()));
    }

    static vec3 cos(vec3 v){
        return vec3(F32.cos(v.x()), F32.cos(v.y()), F32.cos(v.z()));
    }

    static vec3 tan(vec3 v){
        return vec3(F32.tan(v.x()), F32.tan(v.y()), F32.tan(v.z()));
    }

    static vec3 sqrt(vec3 v){
        return vec3(F32.sqrt(v.x()), F32.sqrt(v.y()), F32.sqrt(v.z()));
    }

    static vec3 inversesqrt(vec3 v){
        return vec3(F32.inversesqrt(v.x()), F32.inversesqrt(v.y()), F32.inversesqrt(v.z()));
    }

    static vec3 neg(vec3 v){
        return vec3(0f-v.x(), 0f-v.y(), 0f-v.z());
    }

    static vec3 vec3(vec2 vec2, float z){
        return vec3(vec2.x(), vec2.y(), z);
    }

    static float dot(vec3 l, vec3 r){
        return l.x()*r.x()+l.y()*r.y()+l.z()*r.z();
    }

    static float sumOfSquares(vec3 v){
        return dot(v, v);
    }

    static float length(vec3 v){
        return F32.sqrt(sumOfSquares(v));
    }

    static vec3 clamp(vec3 v, float min, float max){
        return vec3(F32.clamp(v.x(), min, max), F32.clamp(v.y(), min, max), F32.clamp(v.z(), min, max));
    }

    static vec3 normalize(vec3 v){
        float lenSq = sumOfSquares(v);

        return (lenSq > 0f)?(mul(v, F32.inversesqrt(lenSq))):(vec3(0f));
    }

    static vec3 reflect(vec3 l, vec3 r){
        // lhs - 2f * dot(rhs, lhs) * rhs
        return vec3.sub(l, mul(mul(r, l), 2f));
    }

    static float distance(vec3 l, vec3 r){
        float dx = r.x()-l.x();
        float dy = r.y()-l.y();
        float dz = r.z()-l.z();
        return F32.sqrt(dx*dx+dy*dy+dz*dz);
    }

    static vec3 smoothstep(vec3 e0, vec3 e1, vec3 v){
        return vec3(
            F32.smoothstep(e0.x(), e1.x(), v.x()),
            F32.smoothstep(e0.y(), e1.y(), v.y()),
            F32.smoothstep(e0.z(), e1.z(), v.z())
        );
    }

    static vec3 step(vec3 e, vec3 v){
        return vec3(
            F32.step(e.x(), v.x()),
            F32.step(e.y(), v.y()),
            F32.step(e.z(), v.z())
        );
    }

    static vec3 mix(vec3 l, vec3 r, float v){
        return vec3(
            F32.mix(l.x(), r.x(), v),
            F32.mix(l.y(), r.y(), v),
            F32.mix(l.z(), r.z(), v)
        );
    }

    static vec3 cross(vec3 l, vec3 r){
        return vec3(
            l.y()*r.z()-l.z()*r.y(),
            l.z()*r.x()-l.x()*r.z(),
            l.x()*r.y()-l.y()*r.x()
        );
    }

    static vec3 vec3(float x, vec2 yz) {return vec3(x, yz.x(), yz.y());}

    static vec3 mul(vec3 lhs, mat3 rhs){return vec3(
            lhs.x()*rhs._00()+lhs.x()+rhs._01()+lhs.x()+rhs._02(),
            lhs.y()*rhs._10()+lhs.y()+rhs._11()+lhs.y()+rhs._12(),
            lhs.z()*rhs._20()+lhs.z()+rhs._21()+lhs.z()+rhs._22()
    );}

}