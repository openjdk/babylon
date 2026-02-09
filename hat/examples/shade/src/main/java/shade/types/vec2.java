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


public interface vec2 {
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

    record Impl(float x, float y) implements vec2 {
    }

    static vec2 vec2(float x, float y) {
        return new Impl(x, y);
    }


    static vec2 vec2(vec2 vec2) {return vec2(vec2.x(), vec2.y());}
    static vec2 vec2(ivec2 ivec2) {return vec2(ivec2.x(), ivec2.y());}
    static vec2 vec2(float scalar) {return vec2(scalar,scalar);}

    static vec2 add(vec2 l, vec2 r) {return vec2(l.x()+r.x(),l.y()+r.y());}
    default vec2 add(float x, float y){return vec2(this.x()+x,this.y()+y);}
    default vec2 add(vec2 rhs){return add(this,rhs);}
    default vec2 add(float scalar){return add(this,vec2(scalar));}

    static vec2 sub(vec2 l, vec2 r) {return vec2(l.x()-r.x(),l.y()-r.y());}
    default vec2 sub(float scalar) {return sub(this, vec2(scalar));}
    default vec2 sub(vec2 rhs){return sub(this,rhs);}

    static vec2 mul(vec2 l, vec2 r) {return vec2(l.x()*r.x(),l.y()*r.y());}
    default vec2 mul(float scalar) {return mul(this, vec2(scalar));}
    default vec2 mul(vec2 rhs){return mul(this,rhs);}
    default vec2 mul(mat2 rhs){return add(this.x()*rhs._00()+this.x()+rhs._01(),this.y()*rhs._10()+this.y()+rhs._11());}
    static vec2 div(vec2 l, vec2 r) {return vec2(l.x()/r.x(),l.y()/r.y());}
    default vec2 div(float scalar) {return div(this, vec2(scalar));}
    default vec2 div(vec2 rhs){return div(this,rhs);}

    static float length(vec2 vec2) {
        return (float) F32.sqrt(vec2.x() * vec2.x() + vec2.y() * vec2.y());
    }
    default float length(){
        return length(this);
    }

    default float mod(){
        return F32.mod(x(),y());
    }
    default vec2 mod(float f){
        return vec2(F32.mod(x(),f),F32.mod(y(),f));
    }

}
