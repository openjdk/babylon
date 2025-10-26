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

/*
 *  Based on mesh descriptions found here
 *      https://6502disassembly.com/a2-elite/
 *      https://6502disassembly.com/a2-elite/meshes.html
 *
 */
package view;

import view.f32.F32Triangle2D;
import view.f32.F32Vec2;
import view.f32.F32Vec3;

import java.util.stream.IntStream;

import static view.f32.F32Triangle2D.colors;

public class Rasterizer {
    public static final float deltaSquare = 10000f;

    public enum DisplayMode {
        FILL(false,true,false),
        WIRE(true,false,false),
        WIRE_SHOW_HIDDEN(true,false,true),
        WIRE_AND_FILL(true, true, false);
        final public boolean wire;
        final public boolean filled;
        final public boolean showHidden;
        DisplayMode(boolean wire, boolean filled, boolean showHidden){
            this.wire=wire;
            this.filled=filled;
            this.showHidden =showHidden;
        }
    }

    public  final DisplayMode displayMode ;
    final View view;
    int triangle2DEntries[];
    int triangle2DEntriesCount;

    private  Rasterizer(View view, DisplayMode displayMode) {
        this.view = view;
        this.displayMode = displayMode;
    }

    static public Rasterizer of (View view, DisplayMode displayMode){
        return new Rasterizer(view, displayMode);
    }

    private void accept(int gid) {
        int x = gid % view.image.getWidth();
        int y = gid / view.image.getHeight();
        int col = 0x00000;
        for (int t = 0; t < triangle2DEntriesCount; t++) {
            int v0 = triangle2DEntries[F32Triangle2D.SIZE * t + F32Triangle2D.V0];
            int v1 = triangle2DEntries[F32Triangle2D.SIZE * t + F32Triangle2D.V1];
            int v2 = triangle2DEntries[F32Triangle2D.SIZE * t + F32Triangle2D.V2];
            float x0 = F32Vec2.pool.entries[v0 * F32Vec2.pool.stride + F32Vec2.X];
            float y0 = F32Vec2.pool.entries[v0 * F32Vec2.pool.stride + F32Vec2.Y];
            float x1 = F32Vec2.pool.entries[v1 * F32Vec2.pool.stride + F32Vec2.X];
            float y1 = F32Vec2.pool.entries[v1 * F32Vec2.pool.stride + F32Vec2.Y];
            float x2 = F32Vec2.pool.entries[v2 * F32Vec2.pool.stride + F32Vec2.X];
            float y2 = F32Vec2.pool.entries[v2 * F32Vec2.pool.stride + F32Vec2.Y];
            if (displayMode.filled && F32Triangle2D.intriangle(x, y, x0, y0, x1, y1, x2, y2)) {
                col = colors[t];
            } else if (displayMode.wire && F32Triangle2D.onedge(x, y, x0, y0, x1, y1, x2, y2, deltaSquare)) {
                col = 0xffffff;
            }
        }
        view.offscreenRgb[gid] = col;
    }

    public void execute() {
        int range = view.image.getHeight()*view.image.getWidth();
        IntStream.range(0, range).parallel().forEach(this::accept);
    }
}
