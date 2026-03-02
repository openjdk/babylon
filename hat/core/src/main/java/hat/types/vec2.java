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
import hat.types.vec4;
import hat.types.vec3;
import hat.types.vec2;
import hat.types.vec4;
import static hat.types.vec4.*;
import hat.types.vec3;
import static hat.types.vec3.*;
import hat.types.vec2;
import static hat.types.vec2.*;

public interface vec2 extends IfaceValue.vec{
    Shape shape = Shape.of(JavaType.FLOAT, 2);
    float x();
    float y();

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

    static vec2 pow(vec2 l, vec2 r){
        return vec2(F32.pow(l.x(), r.x()), F32.pow(l.y(), r.y()));
    }

    static vec2 pow(float l, vec2 r){
        return vec2(F32.pow(l, r.x()), F32.pow(l, r.y()));
    }

    static vec2 pow(vec2 l, float r){
        return vec2(F32.pow(l.x(), r), F32.pow(l.y(), r));
    }

    static vec2 min(vec2 l, vec2 r){
        return vec2(F32.min(l.x(), r.x()), F32.min(l.y(), r.y()));
    }

    static vec2 min(float l, vec2 r){
        return vec2(F32.min(l, r.x()), F32.min(l, r.y()));
    }

    static vec2 min(vec2 l, float r){
        return vec2(F32.min(l.x(), r), F32.min(l.y(), r));
    }

    static vec2 max(vec2 l, vec2 r){
        return vec2(F32.max(l.x(), r.x()), F32.max(l.y(), r.y()));
    }

    static vec2 max(float l, vec2 r){
        return vec2(F32.max(l, r.x()), F32.max(l, r.y()));
    }

    static vec2 max(vec2 l, float r){
        return vec2(F32.max(l.x(), r), F32.max(l.y(), r));
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

    static vec2 clamp(vec2 v, float min, float max){
        return vec2(F32.clamp(v.x(), min, max), F32.clamp(v.y(), min, max));
    }

    static vec2 normalize(vec2 v){
        float lenSq = sumOfSquares(v);

        return (lenSq >0f)?(mul(v, F32.inversesqrt(lenSq))):(vec2(0f));
    }

    static vec2 reflect(vec2 l, vec2 r){
        // lhs - 2f * dot(rhs, lhs) * rhs
        return vec2.sub(l, mul(mul(r, l), 2f));
    }

    static float distance(vec2 l, vec2 r){
        float dx = r.x()-l.x();
        float dy = r.y()-l.y();
        return F32.sqrt(dx*dx+dy*dy);
    }

    static vec2 smoothstep(vec2 e1, vec2 e2, vec2 r){
        return vec2(
            F32.smoothstep(e1.x(), e2.x(), r.x()),
            F32.smoothstep(e1.y(), e2.y(), r.y())
        );
    }

    static vec2 step(vec2 e, vec2 r){
        return vec2(
            F32.step(e.x(), r.x()),
            F32.step(e.y(), r.y())
        );
    }

    static vec2 mix(vec2 l, vec2 r, float v){
        return vec2(
            F32.mix(l.x(), r.x(), v),
            F32.mix(l.y(), r.y(), v)
        );
    }

    static vec2 mix(vec2 l, vec2 r, vec2 v){
        return vec2(
            F32.mix(l.x(), r.x(), v.x()),
            F32.mix(l.y(), r.y(), v.y())
        );
    }

    static vec2 mod(vec2 l, vec2 r){
        return vec2(
            F32.mod(l.x(), r.x()),
            F32.mod(l.y(), r.y())
        );
    }

    static vec2 mod(vec2 l, float r){
        return vec2(
            F32.mod(l.x(), r),
            F32.mod(l.y(), r)
        );
    }

    static vec2 mul(vec2 l, mat2 r){
        return vec2(
            l.x()*r._00()+l.x()*r._01(),
            l.y()*r._10()+l.y()*r._11()
        );
    }

    static vec2 xx(vec3 v){
        return vec2(v.x(), v.x());
    }

    static vec2 xy(vec3 v){
        return vec2(v.x(), v.y());
    }

    static vec2 yy(vec3 v){
        return vec2(v.y(), v.y());
    }

    static vec2 xz(vec3 v){
        return vec2(v.x(), v.z());
    }

    static vec2 yz(vec3 v){
        return vec2(v.y(), v.z());
    }


}