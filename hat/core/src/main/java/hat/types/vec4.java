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

public interface vec4 extends IfaceValue.vec{
    Shape shape=Shape.of(JavaType.FLOAT, 4);

    float x();
    float y();
    float z();
    float w();

    AtomicInteger count=new AtomicInteger(0);
    AtomicBoolean collect=new AtomicBoolean(false);
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
        // Uncomment to collect stats
        //    if (collect.get())count.getAndIncrement();
        return new Impl(x, y, z, w);
    }

    static vec4 vec4(float scalar){
        return vec4(scalar, scalar, scalar, scalar);
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

/* save to copy to here */
    static vec3 xyz(vec4 vec4){return vec3.vec3(vec4.x(), vec4.y(), vec4.z());}
     static vec4 vec4(vec3 vec3, float w) {return vec4(vec3.x(), vec3.y(), vec3.z(), w);}
    static vec4 vec4(vec2 vec2, float z,float w) {return vec4(vec2.x(), vec2.y(), z, w);}



    static vec4 clamp(vec4 rhs,float min, float max){
        return vec4(Math.clamp(rhs.x(),min,max),Math.clamp(rhs.y(),min,max),Math.clamp(rhs.z(),min,max),Math.clamp(rhs.w(),min,max));
    }

    static vec4 smoothstep(vec4 edge0, vec4 edge1, vec4 vec4){
        return vec4(
                F32.smoothstep(edge0.x(),edge1.x(), vec4.x()),
                F32.smoothstep(edge0.y(),edge1.y(), vec4.y()),
                F32.smoothstep(edge0.z(),edge1.z(), vec4.z()),
                F32.smoothstep(edge0.w(),edge1.w(), vec4.w())
        );
    }


    static vec4 normalize(vec4 vec4){
        float lenSq = vec4.x() * vec4.x() + vec4.y() * vec4.y() + vec4.z() * vec4.z()+ vec4.w()*vec4.w();
        if (lenSq > 0.0f) {
            float invLen = 1f / F32.sqrt(lenSq);
            return vec4(vec4.x() * invLen, vec4.y() * invLen, vec4.z() * invLen, vec4.w() *invLen);
        }
        return vec4(0f, 0f, 0f,0f); // Handle zero-length case
    }

}
