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


public interface vec2 extends  IfaceValue.vec {
    Shape shape = Shape.of( JavaType.FLOAT,3);
    float x();

    float y();

    // A mutable form needed for interface mapping.
    interface Field extends vec2 {
        @Reflect
        default void schema(){x();y();}
        void x(float x);
        void y(float y);
        default vec2 of(float x, float y){
            x(x);y(y);
            return this;
        }
        default vec2 of(vec2 vec2){
            of(vec2.x(),vec2.y());
            return this;
        }
    }

    static vec2 vec2(float x, float y) {
        record Impl(float x, float y) implements vec2 {
        }
        return new Impl(x, y);
    }


    static vec2 vec2(vec2 vec2) {return vec2(vec2.x(), vec2.y());}
    static vec2 vec2(ivec2 ivec2) {return vec2(ivec2.x(), ivec2.y());}
    static vec2 vec2(float scalar) {return vec2(scalar,scalar);}
    static vec2 vec2() {return vec2(0f,0f);}

    static vec2 add(float lx, float ly, float rx, float ry) {return vec2(lx+rx,ly+ry);}
    static vec2 sub(float lx, float ly, float rx, float ry) {return vec2(lx-rx,ly-ry);}
    static vec2 mul(float lx, float ly, float rx, float ry) {return vec2(lx*rx,ly*ry);}
    static vec2 div(float lx, float ly, float rx, float ry) {return vec2(lx/rx,ly/ry);}

    static vec2 add(vec2 l, vec2 r) {return add(l.x(),l.y(),r.x(),r.y());}
    static vec2 add(float  l, vec2 r) {return add(l,l,r.x(),r.y());}
    static vec2 add(vec2  l, float r) {return add(l.x(),l.x(),r,r);}
    static vec2 sub(vec2 l, vec2 r) {return sub(l.x(),l.y(),r.x(),r.y());}
    static vec2 sub(vec2 l, float scalar) {return sub(l.x(),l.y(),scalar,scalar);}
    static vec2 sub(float scalar, vec2 r) {return sub(scalar,scalar, r.x(), r.y());}
    static vec2 mul(vec2 l, vec2 r) {return mul(l.x(),l.y(),r.x(),r.y());}
    static vec2 mul(vec2 l, float scalar) {return mul(l.x(),l.y(),scalar,scalar);}
    static vec2 mul( float scalar, vec2 r) {return mul(scalar,scalar,r.x(),r.y());}
    static vec2 mul(vec2 l, mat2 rhs) {return vec2(l.x()*rhs._00()+l.x()+rhs._01(),l.y()*rhs._10()+l.y()+rhs._11());}
    static vec2 div(vec2 l, vec2 r) {return div(l.x(),l.y(),r.x(),r.y());}
    static vec2 div(vec2 l, float scalar) {return div(l.x(),l.y(),scalar,scalar);}
    static vec2 div( float scalar, vec2 r) {return div(scalar,scalar,r.x(),r.y());}
    static vec2 log(vec2 vec2){return vec2(F32.log(vec2.x()),F32.log(vec2.y()));}
    static float length(vec2 vec2) {return  F32.sqrt(vec2.x() * vec2.x() + vec2.y() * vec2.y());}
    static vec2 mod(vec2 v){return vec2(F32.mod(v.x(),v.y()));}
    static float dot(vec2 lhs, vec2 rhs) { return lhs.x()*rhs.x()+lhs.y()*rhs.y();}
    static vec2 floor(vec2 vec2){return vec2(F32.floor(vec2.x()),F32.floor(vec2.y()));}
    static vec2 fract(vec2 vec2){return vec2(F32.fract(vec2.x()),F32.fract(vec2.y()));}
    static vec2 abs(vec2 vec2){return vec2(F32.abs(vec2.x()),F32.abs(vec2.y()));}
    static vec2 atan(float x, float y){return vec2(F32.atan(x), F32.atan(y));}


    static vec2 min(vec2 lhs,vec2 rhs){return vec2(F32.min(lhs.x(),rhs.x()), F32.min(lhs.y(),rhs.y()));}

    static vec2 max(vec2 lhs,vec2 rhs){return vec2(F32.max(lhs.x(),rhs.x()), F32.max(lhs.y(),rhs.y()));}

    static vec2 mix(vec2 lhs,vec2 rhs, vec2 a){return vec2(F32.mix(lhs.x(),rhs.x(),a.x()), F32.mix(lhs.y(),rhs.y(),a.y()));}

    static vec2 sin(float x, float y){
        return vec2(F32.sin(x),F32.sin(y));
    }
    static vec2 sin(vec2 vec2){
        return sin(vec2.x(),vec2.y());
    }
    static vec2 cos(float x, float y){
        return vec2(F32.cos(x),F32.cos(y));
    }
    static vec2 cos(vec2 vec2){
        return cos(vec2.x(),vec2.y());
    }

    static vec2 pow(vec2 l, vec2 r){
        return vec2(F32.pow(l.x(),r.x()),F32.pow(l.y(),r.y()));
    }

}
