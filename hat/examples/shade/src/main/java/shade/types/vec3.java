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
package shade.types;

import jdk.incubator.code.Reflect;

public interface vec3 {
    float x();

    float y();

    float z();

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
    record Impl(float x, float y, float z) implements vec3 {
    }

    static vec3 vec3(float x, float y, float z) {
        return new Impl(x, y, z);
    }

    static vec3 vec3(vec3 vec3) {return vec3(vec3.x(), vec3.y(), vec3.z());}
    static vec3 vec3(float scalar) {return vec3(scalar,scalar,scalar);}

    static vec3 add(vec3 l, vec3 r) {return vec3(l.x()+r.x(),l.y()+r.y(), l.z()+r.z());}
    default vec3 add(vec3 rhs){return add(this,rhs);}
    default vec3 add(float scalar){return add(this,vec3(scalar));}

    static vec3 sub(vec3 l, vec3 r) {return vec3(l.x()-r.x(),l.y()-r.y(), l.z()-r.z());}
    default vec3 sub(float scalar) {return sub(this, vec3(scalar));}
    default vec3 sub(vec3 rhs){return sub(this,rhs);}

    static vec3 mul(vec3 l, vec3 r) {return vec3(l.x()*r.x(),l.y()*r.y(), l.z()*r.z());}
    default vec3 mul(float scalar) {return mul(this, vec3(scalar));}
    default vec3 mul(vec3 rhs){return mul(this,rhs);}
    default vec3 mul(mat3 rhs){return vec3(
            this.x()*rhs._00()+this.x()+rhs._01()+this.x()+rhs._02(),
            this.y()*rhs._10()+this.y()+rhs._11()+this.y()+rhs._12(),
            this.z()*rhs._20()+this.z()+rhs._21()+this.z()+rhs._22()
    );}

    static vec3 div(vec3 l, vec3 r) {return vec3(l.x()/r.x(),l.y()/r.y(), l.z()/r.z());}
    default vec3 div(float scalar) {return div(this, vec3(scalar));}
    default vec3 div(vec3 rhs){return div(this,rhs);}
    static float distance(vec3 lhs, vec3 rhs){
        return (float)Math.sqrt(lhs.x()*rhs.x()+lhs.y()*rhs.y()+lhs.z()*rhs.z());
    }
    static float length(vec3 vec3){
        return distance(vec3,vec3);
    }
    default float distance(vec3 vec3){
        return distance(this,vec3);
    }
    static vec3 mix(vec3 l, vec3 r, float a) {
        return vec3(
                F32.mix(l.x(),r.x(),a),
                F32.mix(l.y(),r.y(),a),
                F32.mix(l.y(),r.y(),a)
        );
    }
    static float dot(vec3 lhs, vec3 rhs) { return F32.dot(lhs.x(),rhs.x())+F32.dot(lhs.y(),rhs.y()+F32.dot(lhs.z(),rhs.z()));}
    default float dot(vec3 rhs) { return dot(this,rhs);}

    static vec3 reflect(vec3 I, vec3 N) {
        // I - 2.0 * dot(N, I) * N
        return I.sub(N.mul(dot(N,I)).mul(2f));
    }

    static vec3 max(vec3 lhs, vec3 rhs){
        return vec3(F32.max(lhs.x(),rhs.x()),F32.max(lhs.y(),rhs.y()),F32.max(lhs.z(),rhs.z()));
    }

    static vec3 normalize(vec3 vec3){
        float lenSq = vec3.x() * vec3.x() + vec3.y() * vec3.y() + vec3.z() * vec3.z();
        if (lenSq > 0.0f) {
            float invLen = 1.0f / F32.sqrt(lenSq);
            return vec3(vec3.x() * invLen, vec3.y() * invLen, vec3.z() * invLen);
        }
        return vec3(0.0f, 0.0f, 0.0f); // Handle zero-length case
    }
    default vec3 normalize(){
        return normalize(this);
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

    static vec3 sin(vec3 vec3){
        return vec3(F32.sin(vec3.x()),F32.sin(vec3.y()),F32.sin(vec3.z()));
    }


}
