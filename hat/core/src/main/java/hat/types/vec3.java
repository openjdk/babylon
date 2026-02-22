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

import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaType;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import optkl.IfaceValue;
import hat.types.F32;
import static hat.types.F32.*;

public interface vec3 extends IfaceValue.vec{
    Shape shape=Shape.of(JavaType.FLOAT, 3);

    float x();
    float y();
    float z();

    AtomicInteger count=new AtomicInteger(0);
    AtomicBoolean collect=new AtomicBoolean(false);
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
        // Uncomment to collect stats
        //    if (collect.get())count.getAndIncrement();
        return new Impl(x, y, z);
    }

    static vec3 vec3(float scalar){
        return vec3(scalar, scalar, scalar);
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

    /* safe to copy to here */



    static vec3 vec3(vec2 xy, float z) {
        return vec3(xy.x(), xy.y(), z);
    }
    static vec3 vec3(float x, vec2 yz) {
        return vec3(x, yz.x(), yz.y());
    }

    static vec2 xy(vec3 vec3){ return vec2.vec2(vec3.x(),vec3.y());}
    static vec2 yz(vec3 vec3){ return vec2.vec2(vec3.y(),vec3.z());}
    static vec2 xz(vec3 vec3){ return vec2.vec2(vec3.x(),vec3.z());}
    static vec3 mul(vec3 lhs, mat3 rhs){return vec3(
            lhs.x()*rhs._00()+lhs.x()+rhs._01()+lhs.x()+rhs._02(),
            lhs.y()*rhs._10()+lhs.y()+rhs._11()+lhs.y()+rhs._12(),
            lhs.z()*rhs._20()+lhs.z()+rhs._21()+lhs.z()+rhs._22()
    );}

    static vec3 mix(vec3 l, vec3 r, float a) {
        return vec3(
                F32.mix(l.x(),r.x(),a),
                F32.mix(l.y(),r.y(),a),
                F32.mix(l.z(),r.z(),a)
        );
    }



    static vec3 max(vec3 lhs, vec3 rhs){
        return vec3(F32.max(lhs.x(),rhs.x()),F32.max(lhs.y(),rhs.y()),F32.max(lhs.z(),rhs.z()));
    }
    static vec3 min(vec3 lhs, vec3 rhs){
        return vec3(F32.min(lhs.x(),rhs.x()),F32.min(lhs.y(),rhs.y()),F32.min(lhs.z(),rhs.z()));
    }

    static float dot(vec3 lhs, vec3 rhs) { return lhs.x()*rhs.x()+lhs.y()*rhs.y()+lhs.z()*rhs.z();}
    static float sumOfSquares(vec3 v) { return dot(v,v);}
    static vec3 reflect(vec3 I, vec3 N) {
        // I - 2.0 * dot(N, I) * N
        return vec3.sub(I, vec3.mul(vec3.mul(N, dot(N,I)),2f));
    }
    static float distance(vec3 lhs, vec3 rhs){
        var dx = rhs.x()-lhs.x();
        var dy = rhs.y()-lhs.y();
        var dz = rhs.z()-lhs.z();
        return F32.sqrt(dx*dx+dy*dy+dz*dz);
    }
    static float length(vec3 vec3){
        return F32.sqrt(sumOfSquares(vec3));
    }
    static vec3 normalize(vec3 vec3){
        float lenSq = sumOfSquares(vec3);
        return (lenSq > 0.0f)?mul(vec3,F32.inversesqrt(lenSq)):vec3(0.0f);
    }


    static vec3 cross(vec3 a, vec3 b) {
        return vec3(
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x()
        );
    }

    static vec3 clamp(vec3 a, float f1,float f2) {
        return vec3(F32.clamp(a.x(),f1,f2),F32.clamp(a.y(),f1,f2),F32.clamp(a.z(),f1,f2));

    }
    static vec3 pow(vec3 lhs, vec3 rhs){
        return vec3(F32.pow(lhs.x(),rhs.x()),F32.pow(lhs.y(),rhs.y()),F32.pow(lhs.z(),rhs.z()));
    }

    static vec3 pow(vec3 lhs, float scalar){
        return vec3(F32.pow(lhs.x(),scalar),F32.pow(lhs.y(),scalar),F32.pow(lhs.z(),scalar));
    }


}
