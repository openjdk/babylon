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

import view.ViewFrame;
import view.f32.pool.F32x2Pool;
import view.f32.pool.F32x2TrianglePool;
import view.f32.pool.F32x3Pool;
import view.f32.pool.F32x3TrianglePool;
import view.f32.pool.F32x4x4Pool;

public record ModelHighWaterMark(
        int markedTriangles3D,
        int markedTriangles2D,
        int markedVec2,
        int markedVec3,
        int markedMat4) {

    public ModelHighWaterMark() {
        this(
                ((F32x3TrianglePool)ViewFrame.f32.f32x3TriangleFactory()).count,
                ((F32x2TrianglePool)ViewFrame.f32.f32x2TriangleFactory()).count,
                ((F32x2Pool)ViewFrame.f32.f32x2Factory()).count,
                ((F32x3Pool)ViewFrame.f32.f32x3Factory()).count,
                ((F32x4x4Pool)ViewFrame.f32.f32x4x4Factory()).count);
    }

    public void resetAll() {
        reset3D();
        ((F32x2TrianglePool)ViewFrame.f32.f32x2TriangleFactory()).reset(markedTriangles2D);
        ((F32x2Pool)ViewFrame.f32.f32x2Factory()).reset(markedVec2);
    }

    public void reset3D() {
        ((F32x3TrianglePool)ViewFrame.f32.f32x3TriangleFactory()).count = markedTriangles3D;
        ((F32x3Pool)ViewFrame.f32.f32x3Factory()).count = markedVec3;
        ((F32x4x4Pool)ViewFrame.f32.f32x4x4Factory()).count = markedMat4;
    }
}
