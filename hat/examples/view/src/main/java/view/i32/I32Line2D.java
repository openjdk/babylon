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
package view.i32;

public class I32Line2D {
    public static final int SIZE = 3;
    public static final int V0 = 0;
    public static final int V1 = 1;
    public static int MAX = 1000;
    public static int count = 0;

    public static int[] entries = new int[MAX * SIZE];
    public static int[] colors = new int[MAX];


    public static boolean online(float x, float y, float x0, float y0, float x1, float y1, float deltaSquare) {
        float dxl = x1 - x0;
        float dyl = y1 - y0;
        float cross = (x - x0) * dyl - (y - y0) * dxl;
        if ((cross * cross) < deltaSquare) {
            if (dxl * dxl >= dyl * dyl)
                return dxl > 0 ? x0 <= x && x <= x1 : x1 <= x && x <= x0;
            else
                return dyl > 0 ? y0 <= y && y <= y1 : y1 <= y && y <= y0;
        } else {
            return false;
        }
    }


public static int createLine(int x0, int y0, int x1, int y1, int col) {
        entries[count * SIZE + V0] = I32Vec2.createVec2(x0,y0);
        entries[count * SIZE + V1] = I32Vec2.createVec2(x1,y1);

        colors[count] = col;
        return count++;
    }

    static int createLine(int v0, int v1, int col) {
        entries[count * SIZE + V0] = v0;
            entries[count * SIZE + V1] = v1;

        colors[count] = col;
        return count++;
    }
}
