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

public class I32Triangle2D {
    public static final int SIZE = 3;
    public static final int V0 = 0;
    public static final int V1 = 1;
    public static final int V2 = 2;
    public static int MAX = 1000;
    public static int count = 0;

    public static int[] entries = new int[MAX * SIZE];
    public static int[] colors = new int[MAX];
    public static float side(int x, int y, int x0, int y0, int x1, int y1) {
        return (y1 - y0) * (x - x0) + (-x1 + x0) * (y - y0);
    }

    public static float side(int v, int v0, int v1) {
        v*= I32Vec2.SIZE;
        v0*= I32Vec2.SIZE;
        v1*= I32Vec2.SIZE;
        return (I32Vec2.entries[v1+ I32Vec2.Y] - I32Vec2.entries[v0+ I32Vec2.Y] * (I32Vec2.entries[v+ I32Vec2.X] - I32Vec2.entries[v0+ I32Vec2.X]) + (-I32Vec2.entries[v1+ I32Vec2.X] + I32Vec2.entries[v0+ I32Vec2.X]) * (I32Vec2.entries[v+ I32Vec2.Y] - I32Vec2.entries[v0+ I32Vec2.Y]));
    }

    public static boolean intriangle(int x, int y, int x0, int y0, int x1, int y1, int x2, int y2) {
        return side(x, y, x0, y0, x1, y1) >= 0 && side(x, y, x1, y1, x2, y2) >= 0 && side(x, y, x2, y2, x0, y0) >= 0;
    }
    public static boolean intriangle(int v, int v0, int v1, int v2){
        return side(v, v0, v1) >= 0 && side(v, v1, v2) >= 0 && side(v, v2, v0) >= 0;
    }

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
    public static boolean onedge(float x, float y, float x0, float y0, float x1, float y1, float x2, float y2, float deltaSquare) {
        return online(x, y, x0, y0, x1, y1, deltaSquare) || I32Triangle2D.online(x, y, x1, y1, x2, y2, deltaSquare) || I32Triangle2D.online(x, y, x2, y2, x0, y0, deltaSquare);
    }


public static int createTriangle(int x0, int y0, int x1, int y1, int x2, int y2, int col) {
        entries[count * SIZE + V0] = I32Vec2.createVec2(x0,y0);
        // We need the triangle to be clock wound
        if (side(x0, y0, x1, y1, x2, y2) > 0) {
            entries[count * SIZE + V1] = I32Vec2.createVec2(x1,y1);
            entries[count * SIZE + V2] = I32Vec2.createVec2(x2,y2);
        } else {
            entries[count * SIZE + V1] = I32Vec2.createVec2(x2,y2);
            entries[count * SIZE + V2] = I32Vec2.createVec2(x1,y1);
        }
        colors[count] = col;
        return count++;
    }

    static int createTriangle(int v0, int v1, int v2, int col) {
        entries[count * SIZE + V0] = v0;
        // We need the triangle to be clock wound
        if (side(v0, v1, v2) > 0) {
            entries[count * SIZE + V1] = v1;
            entries[count * SIZE + V2] = v2;
        } else {
            entries[count * SIZE + V1] = v2;
            entries[count * SIZE + V2] = v1;
        }
        colors[count] = col;
        return count++;
    }
}
