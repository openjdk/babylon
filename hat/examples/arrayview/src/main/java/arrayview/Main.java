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
package arrayview;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.S32Array;
import hat.buffer.S32Array2D;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;

import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;

import static hat.ifacemapper.MappableIface.RO;
import static hat.ifacemapper.MappableIface.RW;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public class Main {

    @CodeReflection
    public static int mandelCheck(@RO KernelContext kc, @RW S32Array2D s32Array2D, @RO S32Array pallette, float offsetx, float offsety, float scale) {
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
            return colorIdx < pallette.length() ? pallette.array(colorIdx) : 0;
        }
        return 0;
    }

    @CodeReflection
    public static void mandel(@RO KernelContext kc, @RW S32Array2D s32Array2D, @RO S32Array pallette, float offsetx, float offsety, float scale) {
        if (kc.x < kc.maxX) {
            int[] pal = pallette.arrayView();
            int[][] s32 = s32Array2D.arrayView();
            float width = s32Array2D.width();
            float height = s32Array2D.height();
            float x = ((kc.x % s32Array2D.width()) * scale - (scale / 2f * width)) / width + offsetx;
            float y = ((kc.x / s32Array2D.width()) * scale - (scale / 2f * height)) / height + offsety;
            float zx = x;
            float zy = y;
            float new_zx;
            int colorIdx = 0;
            while ((colorIdx < pal.length) && (((zx * zx) + (zy * zy)) < 4f)) {
                new_zx = ((zx * zx) - (zy * zy)) + x;
                zy = (2f * zx * zy) + y;
                zx = new_zx;
                colorIdx++;
            }
            int color = colorIdx < pal.length ? pal[colorIdx] : 0;
            s32[kc.x % s32Array2D.width()][kc.x / s32Array2D.width()] = color;
        }
    }


    @CodeReflection
    static public void compute(final ComputeContext computeContext, S32Array pallete, S32Array2D s32Array2D, float x, float y, float scale) {

        computeContext.dispatchKernel(
                s32Array2D.width()*s32Array2D.height(), //0..S32Array2D.size()
                kc -> mandel(kc, s32Array2D, pallete, x, y, scale));
    }

    public static void main(String[] args) {
        final int width = 1024;
        final int height = 1024;
        final float defaultScale = 3f;
        final float originX = -1f;
        final float originY = 0;
        final int maxIterations = 64;

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        S32Array2D s32Array2D = S32Array2D.create(accelerator, width, height);

        int[] palletteArray = new int[maxIterations];

        for (int i = 1; i < maxIterations; i++) {
            palletteArray[i]=(i/8+1);// 0-7?
        }
        palletteArray[0]=0;
        S32Array pallette = S32Array.createFrom(accelerator, palletteArray);

        accelerator.compute(cc -> compute(cc, pallette, s32Array2D, originX, originY, defaultScale));

        // Well take 1 in 4 samples (so 1024 -> 128 grid) of the pallette.
        int subsample = 16;
        char[] charPallette9 = new char []{' ', '.', ',',':', '-', '+','*', '#', '@', '%'};
        for (int y = 0; y<height/subsample; y++) {
            for (int x = 0; x<width/subsample; x++) {
                int palletteValue = s32Array2D.get(x*subsample,y*subsample); // so 0->8
                System.out.print(charPallette9[palletteValue]);
            }
            System.out.println();
        }
    }
}
