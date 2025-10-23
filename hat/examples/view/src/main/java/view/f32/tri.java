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


import java.util.ArrayList;
import java.util.List;

public class tri {
    private int id;

    public tri( int id) {
        this.id = id;
    }

    public static List<tri> all() {
        List<tri> all = new ArrayList<>();
        for (int t = 0; t < F32Triangle3D.pool.count; t++) {
            all.add(new tri(t));
        }
        return all;
    }

    public tri mul(mat4 m) {
        return new tri(F32Triangle3D.mulMat4(id, m.id));
    }

    public tri add(vec3 v) {
        return new tri(F32Triangle3D.addVec3(id, v.id));

    }

    public vec3 normalSumOfSquares() {
        return new vec3(F32Triangle3D.normalSumOfSquares(id));
    }

    public vec3 normal() {
        return new vec3(F32Triangle3D.normal(id));
    }

    public vec3 v0() {
        return new vec3(F32Triangle3D.getV0(id));
    }

    public vec3 v1() {
        return new vec3(F32Triangle3D.getV1(id));
    }

    public vec3 v2() {
        return new vec3(F32Triangle3D.getV2(id));
    }

    public tri mul(float s) {
        return new tri(F32Triangle3D.mulScaler(id, s));
    }

    public tri add(float s) {
        return new tri(F32Triangle3D.addScaler(id, s));
    }

    public int rgb() {
        return F32Triangle3D.getRGB(id);
    }

    public vec3 center() {
        return new vec3(F32Triangle3D.getCentre(id));
    }
}
