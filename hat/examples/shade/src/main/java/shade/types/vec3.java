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

    static vec3 div(vec3 l, vec3 r) {return vec3(l.x()/r.x(),l.y()/r.y(), l.z()/r.z());}
    default vec3 div(float scalar) {return div(this, vec3(scalar));}
    default vec3 div(vec3 rhs){return div(this,rhs);}
}
