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
package mandel;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.buffer.S32Array2D;

import java.awt.Color;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class MandelCompute {
    @CodeReflection
    public static void mandel(KernelContext kc, S32Array2D s32Array2D, S32Array pallette, float offsetx, float offsety, float scale) {
        if (kc.x < kc.maxX) {
            float width = s32Array2D.width();
            float height = s32Array2D.height();
            float x = ((kc.x % s32Array2D.width()) * scale - (scale / 2f * width)) / width + offsetx;
            float y = ((kc.x / s32Array2D.width()) * scale - (scale / 2f * height)) / height + offsety;
            float zx = x;
            float zy = y;
            float new_zx;
            int colorIdx = 0;
            while ((colorIdx < pallette.length()) && (((zx * zx) + (zy * zy)) < 4f)) {
                new_zx = ((zx * zx) - (zy * zy)) + x;
                zy = (2f * zx * zy) + y;
                zx = new_zx;
                colorIdx++;
            }
            int color = colorIdx < pallette.length() ? pallette.array(colorIdx) : 0;
            s32Array2D.array(kc.x, color);
        }
    }


    @CodeReflection
    static public void compute(final ComputeContext computeContext, S32Array pallete, S32Array2D s32Array2D, float x, float y, float scale) {

        computeContext.dispatchKernel(
                s32Array2D.size(), //0..S32Array2D.size()
                kc -> MandelCompute.mandel(kc, s32Array2D, pallete, x, y, scale));
    }

    public static void main(String[] args) {

        final int width = 1024;
        final int height = 1024;
        final float defaultScale = 2f;
        final float originX = -1f;
        final float originY = 0;
        final int maxIterations = 64;
        final int zoomFrames = 200;

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        S32Array2D s32Array2D = S32Array2D.create(accelerator, width, height);

        int[] palletteArray = new int[maxIterations];
        for (int i = 0; i < maxIterations; i++) {
            final float h = i / (float) maxIterations;
            final float b = 1.0f - (h * h);
            palletteArray[i] = Color.HSBtoRGB(h, 1f, b);
        }
        S32Array pallette = S32Array.create(accelerator, palletteArray);

        accelerator.compute(cc -> MandelCompute.compute(cc, pallette, s32Array2D, originX, originY, defaultScale));

        MandelViewer mandelViewer = new MandelViewer("mandel", s32Array2D);
        mandelViewer.imageViewer.syncWithRGB(s32Array2D);

        while (mandelViewer.imageViewer.getZoomPoint(defaultScale) instanceof MandelViewer.PointF32 zoomPoint) {
            float x = originX;
            float y = originY;
            float scale = defaultScale;
            final long startMillis = System.currentTimeMillis();

            for (int sign : new int[]{-1, 1}) {
                for (int i = 0; i < zoomFrames; i++) {
                    scale = scale + ((sign * defaultScale) / zoomFrames);
                    final float fscale = scale;
                    final float fx = x - sign * zoomPoint.x / zoomFrames;
                    final float fy = y - sign * zoomPoint.y / zoomFrames;
                    accelerator.compute(cc -> MandelCompute.compute(cc, pallette, s32Array2D, fx, fy, fscale));
                    mandelViewer.imageViewer.syncWithRGB(s32Array2D);
                }
            }
            System.out.println("FPS = " + ((zoomFrames * 2 * 1000) / (System.currentTimeMillis() - startMillis)));
        }
    }
}
