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
package view;

import view.f32.F32Matrix4x4;
import view.f32.F32Triangle2D;
import view.f32.F32Triangle3D;
import view.f32.F32Vec2;
import view.f32.F32Vec3;

record ModelHighWaterMark(
        int markedTriangles3D,
        int markedTriangles2D,
        int markedVec2,
        int markedVec3,
        int markedMat4) {

    ModelHighWaterMark() {
        this(F32Triangle3D.pool.count, F32Triangle2D.arr.size(), F32Vec2.arr.size(), F32Vec3.pool.count, F32Matrix4x4.pool.count);
    }

    void resetAll() {
        reset3D();
        F32Triangle2D.reset(markedTriangles2D);
        F32Vec2.reset(markedVec2);
    }

    void reset3D() {
        F32Triangle3D.pool.count = markedTriangles3D;
        F32Vec3.pool.count = markedVec3;
        F32Matrix4x4.pool.count = markedMat4;
    }
}
