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
package view.f32;

public class vec3 {
    int id;
    vec3(int id){
        this.id = id;
    }

    public vec3(float x, float y, float z) {
        this(F32Vec3.createVec3(x,y,z));
    }

    public vec3 sub(vec3 v) {
        return new vec3(F32Vec3.subVec3(id, v.id));
    }
    public vec3 add(vec3 v) {
        return new vec3(F32Vec3.addVec3(id, v.id));
    }
    public vec3 mul(vec3 v) {
        return new vec3(F32Vec3.mulVec3(id, v.id));
    }

    public float dotProd(vec3 v){
        return F32Vec3.dotProd(id, v.id);
    }
    public float sumOf(){
        return F32Vec3.sumOf(id);
    }

    public float x() {
        return F32Vec3.getX(id);
    }
    public float y() {
        return F32Vec3.getY(id);
    }
    public float z() {
        return F32Vec3.getZ(id);
    }
}
