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
package hat.types;

import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public interface vec3 extends  IfaceValue.vec {
    Shape shape = Shape.of( JavaType.FLOAT,3);
    float x();

    float y();

    float z();
    AtomicInteger count = new AtomicInteger(0);
    AtomicBoolean collect = new AtomicBoolean(false);
    //   if (collect.get())count.getAndIncrement();
    // A mutable variant needed for interface mapping
    interface Field extends vec3 {
        @Reflect
        default void schema(){x();y();z();}
        void x(float x);
        void y(float y);
        void z(float z);
        default vec3 of(float x, float y, float z){
            x(x);y(y);z(z);
            return this;
        }
        default vec3 of(vec3 vec3){
            of(vec3.x(),vec3.y(),vec3.z());
            return this;
        }
    }


    static vec3 vec3(float x, float y, float z) {
        record Impl(float x, float y, float z) implements vec3 {
        }
      //  if (collect.get())count.getAndIncrement();
        return new Impl(x, y,z);
    }

    static vec3 vec3(vec2 xy, float z) {
        return vec3(xy.x(), xy.y(), z);
    }

    static vec3 vec3(vec3 vec3) {return vec3(vec3.x(), vec3.y(), vec3.z());}
    static vec3 vec3(float scalar) {return vec3(scalar,scalar,scalar);}

    static vec3 add(vec3 l, vec3 r) {return vec3(l.x()+r.x(),l.y()+r.y(), l.z()+r.z());}
    static vec3 add(vec3 l, float scalar) {return vec3(l.x()+scalar,l.y()+scalar, l.z()+scalar);}
    static vec3 add(float scalar, vec3 r) {return vec3(scalar+r.x(),scalar+r.y(), scalar+r.z());}

    static vec3 sub(vec3 l, vec3 r) {return vec3(l.x()-r.x(),l.y()-r.y(), l.z()-r.z());}
    static vec3 sub(vec3 l, float scalar) {return vec3(l.x()-scalar,l.y()-scalar, l.z()-scalar);}
    static vec3 neg(vec3 vec3) {return vec3(0-vec3.x(),0-vec3.y(), 0-vec3.z());}
    static vec3 mul(vec3 l, vec3 r) {return vec3(l.x()*r.x(),l.y()*r.y(), l.z()*r.z());}
    static vec3 mul(vec3 l, float scalar ) {return vec3(l.x()*scalar,l.y()*scalar, l.z()*scalar);}
    static vec3 mul(float scalar, vec3 r) {return vec3(scalar*r.x(),scalar*r.y(), scalar*r.z());}

    static vec3 mul(vec3 lhs, mat3 rhs){return vec3(
            lhs.x()*rhs._00()+lhs.x()+rhs._01()+lhs.x()+rhs._02(),
            lhs.y()*rhs._10()+lhs.y()+rhs._11()+lhs.y()+rhs._12(),
            lhs.z()*rhs._20()+lhs.z()+rhs._21()+lhs.z()+rhs._22()
    );}

    static vec3 div(vec3 l, vec3 r) {return vec3(l.x()/r.x(),l.y()/r.y(), l.z()/r.z());}
    static vec3 div(vec3 l, float scalar) {return vec3(l.x()/scalar,l.y()/scalar, l.z()/scalar);}

    static float distance(vec3 lhs, vec3 rhs){
        return (float)Math.sqrt(lhs.x()*rhs.x()+lhs.y()*rhs.y()+lhs.z()*rhs.z());
    }
    static float length(vec3 vec3){
        return distance(vec3,vec3);
    }
    static vec3 mix(vec3 l, vec3 r, float a) {
        return vec3(
                F32.mix(l.x(),r.x(),a),
                F32.mix(l.y(),r.y(),a),
                F32.mix(l.y(),r.y(),a)
        );
    }
    static float dot(vec3 lhs, vec3 rhs) { return lhs.x()*rhs.x()+lhs.y()*rhs.y()+lhs.z()*rhs.z();}

    static vec3 reflect(vec3 I, vec3 N) {
        // I - 2.0 * dot(N, I) * N
        return vec3.sub(I, vec3.mul(vec3.mul(N, dot(N,I)),2f));
    }

    static vec3 max(vec3 lhs, vec3 rhs){
        return vec3(F32.max(lhs.x(),rhs.x()),F32.max(lhs.y(),rhs.y()),F32.max(lhs.z(),rhs.z()));
    }
    static vec3 min(vec3 lhs, vec3 rhs){
        return vec3(F32.min(lhs.x(),rhs.x()),F32.min(lhs.y(),rhs.y()),F32.min(lhs.z(),rhs.z()));
    }

    static vec3 normalize(vec3 vec3){
        float lenSq = vec3.x() * vec3.x() + vec3.y() * vec3.y() + vec3.z() * vec3.z();
        if (lenSq > 0.0f) {
            float invLen = 1.0f / F32.sqrt(lenSq);
            return vec3(vec3.x() * invLen, vec3.y() * invLen, vec3.z() * invLen);
        }
        return vec3(0.0f, 0.0f, 0.0f); // Handle zero-length case
    }


    static vec3 cross(vec3 a, vec3 b) {
        return vec3(
                a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x());

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
    static vec3 sin(float x, float y, float z){
        return vec3(F32.sin(x),F32.sin(y),F32.sin(z));
    }
    static vec3 sin(vec3 vec3){
        return sin(vec3.x(),vec3.y(),vec3.z());
    }

    static vec3 cos(vec3 vec3){
        return vec3(F32.cos(vec3.x()),F32.cos(vec3.y()),F32.cos(vec3.z()));
    }

   // static vec3 neg(vec3 vec3){
     //   return vec3(-vec3.x(),-vec3.y(), -vec3.z() );
   // }
}
