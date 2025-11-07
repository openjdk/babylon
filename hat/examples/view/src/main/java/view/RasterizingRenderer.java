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

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.stream.IntStream;

public record RasterizingRenderer(int width, int height,  DisplayMode displayMode, BufferedImage image, int[] offscreenRgb) implements Renderer {
    static private Renderer of(int width, int height, DisplayMode displayMode) {
        var image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        return new RasterizingRenderer(width, height, displayMode, image, new int[((DataBufferInt) image.getRaster().getDataBuffer()).getData().length]);
    }

    static public Renderer wireOf(int width, int height) {
        return of(width, height, DisplayMode.WIRE);
    }

    static public Renderer fillOf(int width, int height) {
        return of(width, height, DisplayMode.FILL);
    }

    private void kernel(int gid) {
        int x = gid % width;
        int y = gid / height;
        int col = 0x404040;

            for (int t = 0; t < F32Triangle2D.f32Triangle2DPool.count; t++) {
                int v0 = F32Triangle2D.f32Triangle2DPool.entries[F32Triangle2D.f32Triangle2DPool.stride * t + F32Triangle2D.F32Triangle2DPool.V0];
                int v1 = F32Triangle2D.f32Triangle2DPool.entries[F32Triangle2D.f32Triangle2DPool.stride * t + F32Triangle2D.F32Triangle2DPool.V1];
                int v2 = F32Triangle2D.f32Triangle2DPool.entries[F32Triangle2D.f32Triangle2DPool.stride * t + F32Triangle2D.F32Triangle2DPool.V2];
                float x0 = F32Vec2.f32Vec2Pool.entries[v0 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.X];
                float y0 = F32Vec2.f32Vec2Pool.entries[v0 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.Y];
                float x1 = F32Vec2.f32Vec2Pool.entries[v1 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.X];
                float y1 = F32Vec2.f32Vec2Pool.entries[v1 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.Y];
                float x2 = F32Vec2.f32Vec2Pool.entries[v2 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.X];
                float y2 = F32Vec2.f32Vec2Pool.entries[v2 * F32Vec2.f32Vec2Pool.stride + F32Vec2.F32Vec2Pool.Y];
                if (displayMode.filled && F32Triangle2D.intriangle(x, y, x0, y0, x1, y1, x2, y2)) {
                    col = F32Triangle2D.f32Triangle2DPool.entries[F32Triangle2D.f32Triangle2DPool.stride * t + F32Triangle2D.F32Triangle2DPool.RGB];
                } else if (displayMode.wire && F32Triangle2D.onedge(x, y, x0, y0, x1, y1, x2, y2)) {
                    col = F32Triangle2D.f32Triangle2DPool.entries[F32Triangle2D.f32Triangle2DPool.stride * t + F32Triangle2D.F32Triangle2DPool.RGB];
                }
            }

        offscreenRgb[gid] = col;
    }

    @Override
    public void render() {
        IntStream.range(0, width * height).parallel().forEach(this::kernel);
        System.arraycopy(offscreenRgb, 0, ((DataBufferInt) image.getRaster().getDataBuffer()).getData(), 0, offscreenRgb.length);
    }

    @Override
    public void paint(Graphics2D g) {
        g.drawImage(image, 0, 0, width, height, null);
    }


}
