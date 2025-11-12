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

public class ZPos implements Comparable<ZPos> {
    public enum ColourMode {NORMALIZED_COLOUR, NORMALIZED_INV_COLOUR, COLOUR, NORMALIZED_WHITE, NORMALIZED_INV_WHITE, WHITE}
    public static final ColourMode colourMode = ColourMode.COLOUR;
    F32 f32;
    final F32x3 v0;
    final F32x3 v1;
    final F32x3 v2;
    final float x0, y0, x1, y1, x2, y2;
    final float z0, z1, z2;
    final float z;
    final float howVisible;
    final int rgb;

    @Override
    public int compareTo(ZPos zPos) {
        return Float.compare(z, zPos.z);
    }

    public ZPos(F32 f32, F32x3Triangle t, float howVisible) {
        this.f32 = f32;
        this.v0 = t.v0();
        this.v1 = t.v1();
        this.v2 = t.v2();
        this.x0 = v0.x();
        this.y0 = v0.y();
        this.z0 = v0.z();
        this.x1 = v1.x();
        this.y1 =  v1.y();
        this.z1 = v1.z();
        this.x2 =  v2.x();
        this.y2 =  v2.y();
        this.z2 = v2.z();
        this.rgb = t.rgb();
        this.howVisible = howVisible;
        this.z = Math.min(z0, Math.min(z1, z2));
    }


    public F32x2Triangle create() {
        int r = ((rgb & 0xff0000) >> 16);
        int g = ((rgb & 0x00ff00) >> 8);
        int b = ((rgb & 0x0000ff) >> 0);

        if (colourMode == ColourMode.NORMALIZED_COLOUR) {
            r = r - (int) (20 * howVisible);
            g = g - (int) (20 * howVisible);
            b = b - (int) (20 * howVisible);
        } else if (colourMode == ColourMode.NORMALIZED_INV_COLOUR) {
            r = r + (int) (20 * howVisible);
            g = g + (int) (20 * howVisible);
            b = b + (int) (20 * howVisible);
        } else if (colourMode == ColourMode.NORMALIZED_WHITE) {
            r = g = b = (int) (0x7f - (20 * howVisible));
        } else if (colourMode == ColourMode.NORMALIZED_INV_WHITE) {
            r = g = b = (int) (0x7f + (20 * howVisible));
        } else if (colourMode == ColourMode.WHITE) {
            r = g = b = 0xff;
        }
        F32x2 v0 = f32.f32x2Factory().of(x0,y0);
        F32x2 v1 = f32.f32x2Factory().of(x1,y1);
        F32x2 v2 = f32.f32x2Factory().of(x2,y2);
        return f32.f32x2TriangleFactory().of(v0,v1,v2, (r & 0xff) << 16 | (g & 0xff) << 8 | (b & 0xff));
    }
}
