/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package hat.test;

import hat.Accelerator;
import hat.ComputeContext;
import hat.NDRange;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.*;
import hat.ifacemapper.MappableIface.*;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;

import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public class TestArrayView {

    /*
     * simple square kernel example using S32Array's ArrayView
     */
    @CodeReflection
    public static void squareKernel(@RO  KernelContext kc, @RW S32Array s32Array) {
        if (kc.gix < kc.gsx){
            int[] arr = s32Array.arrayView();
            arr[kc.gix] *= arr[kc.gix];
        }
    }

    @CodeReflection
    public static void square(@RO ComputeContext cc, @RW S32Array s32Array) {
        cc.dispatchKernel(NDRange.of(s32Array.length()),
                kc -> squareKernel(kc, s32Array)
        );
    }

    @HatTest
    public static void testSquare() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array.create(accelerator, 32);
        for (int i = 0; i < arr.length(); i++) {
            arr.array(i, i);
        }
        accelerator.compute(
                cc -> square(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.length(); i++) {
            HATAsserts.assertEquals(i * i, arr.array(i));
        }
    }

    /*
     * making sure arrayviews aren't reliant on varOps
     */
    @CodeReflection
    public static void squareKernelNoVarOp(@RO  KernelContext kc, @RW S32Array s32Array) {
        if (kc.gix<kc.gsx){
            s32Array.arrayView()[kc.gix] *= s32Array.arrayView()[kc.gix];
        }
    }

    @CodeReflection
    public static void squareNoVarOp(@RO ComputeContext cc, @RW S32Array s32Array) {
        NDRange ndRange = NDRange.of(NDRange.Global1D.of(s32Array.length()));
        cc.dispatchKernel(ndRange,
                kc -> squareKernelNoVarOp(kc, s32Array)
        );
    }

    @HatTest
    public static void testSquareNoVarOp() {
        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array.create(accelerator, 32);
        for (int i = 0; i < arr.length(); i++) {
            arr.array(i, i);
        }
        accelerator.compute(
                cc -> squareNoVarOp(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.length(); i++) {
            HATAsserts.assertEquals(i * i, arr.array(i));
        }
    }

    @CodeReflection
    public static void square2DKernel(@RO  KernelContext kc, @RW S32Array2D s32Array2D) {
        if (kc.gix < kc.gsx){
            int[][] arr = s32Array2D.arrayView();
            arr[kc.gix][kc.giy] *= arr[kc.gix][kc.giy];
        }
    }

    @CodeReflection
    public static void square2D(@RO ComputeContext cc, @RW S32Array2D s32Array2D) {
        cc.dispatchKernel(NDRange.of(s32Array2D.width() * s32Array2D.height()),
                kc -> square2DKernel(kc, s32Array2D)
        );
    }

    @HatTest
    public static void testSquare2D() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array2D.create(accelerator, 5, 5);
        for (int i = 0; i < arr.height(); i++) {
            for (int j = 0; j < arr.width(); j++) {
                arr.set(i, j, i * 5 + j);
            }
        }
        accelerator.compute(
                cc -> square2D(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.height(); i++) {
            for (int j = 0; j < arr.width(); j++) {
                HATAsserts.assertEquals((i * 5 + j) * (i * 5 + j), arr.get(i, j));
            }
        }
    }

    /*
     * simplified version of Game of Life using ArrayView
     */
    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public interface CellGrid extends Buffer {
        /*
         * struct CellGrid{
         *     int width;
         *     int height;
         *     byte[width*height*2] cellArray;
         *  }
         */
        int width();

        int height();

        byte array(long idx);

        void array(long idx, byte b);

        Schema<CellGrid> schema = Schema.of(CellGrid.class, lifeData -> lifeData
                .arrayLen("width", "height").stride(2).array("array")
        );

        static CellGrid create(Accelerator accelerator, int width, int height) {
            return schema.allocate(accelerator, width, height);
        }

        ValueLayout valueLayout = JAVA_BYTE;

        default byte[] arrayView() {
            int size = this.width() * this.height();
            byte[] arr = new byte[size];
            for (int i = 0; i < size; i++) {
                arr[i] = this.array(i);
            }
            return arr;
        }
    }

    public interface Control extends Buffer {
        /*
         * struct Control{
         *     int from;
         *     int to;
         *  }
         */
        int from();

        void from(int from);

        int to();

        void to(int to);

        Schema<Control> schema = Schema.of(
                Control.class, control ->
                        control.fields("from", "to"));//, "generation", "requiredFrameRate", "maxGenerations"));

        static Control create(Accelerator accelerator, CellGrid cellGrid) {
            var instance = schema.allocate(accelerator);
            instance.from(cellGrid.width() * cellGrid.height());
            instance.to(0);
            return instance;
        }
    }

    public static class Compute {
        @CodeReflection
        public static void lifePerIdx(int idx, @RO Control control, @RW CellGrid cellGrid) {
            int w = cellGrid.width();
            int h = cellGrid.height();
            int from = control.from();
            int to = control.to();
            int x = idx % w;
            int y = idx / w;

            // byte[] bytes = cellGrid.arrayView();
            // byte cell = bytes[idx + from];
            // byte[] lookup = new byte[]{};
            // if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
            //     int lookupIdx =
            //             (bytes[(y - 1) * w + x - 1 + from]&1 <<0)
            //                     |(bytes[(y + 0) * w + x - 1 + from]&1 <<1)
            //                     |(bytes[(y + 1) * w + x - 1 + from]&1 <<2)
            //                     |(bytes[(y - 1) * w + x + 0 + from]&1 <<3)
            //                     |(bytes[(y - 0) * w + x + 0 + from]&1 <<4) // current cell added
            //                     |(bytes[(y + 1) * w + x + 0 + from]&1 <<5)
            //                     |(bytes[(y + 0) * w + x + 1 + from]&1 <<6)
            //                     |(bytes[(y - 1) * w + x + 1 + from]&1 <<7)
            //                     |(bytes[(y + 1) * w + x + 1 + from]&1 <<8) ;
            //     // conditional removed!
            //     bytes[idx + to] = lookup[lookupIdx];
            // }

            byte[] bytes = cellGrid.arrayView();
            byte cell = bytes[idx];
            if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                int count =
                        (bytes[(y - 1) * w + (x - 1)] & 1)
                                + (bytes[(y + 0) * w + (x - 1)] & 1)
                                + (bytes[(y + 1) * w + (x - 1)] & 1)
                                + (bytes[(y - 1) * w + (x + 0)] & 1)
                                + (bytes[(y + 1) * w + (x + 0)] & 1)
                                + (bytes[(y - 1) * w + (x + 1)] & 1)
                                + (bytes[(y + 0) * w + (x + 1)] & 1)
                                + (bytes[(y + 1) * w + (x + 1)] & 1);
                cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
            }
            bytes[idx] = cell;
        }

        @CodeReflection
        public static void life(@RO KernelContext kc, @RO Control control, @RW CellGrid cellGrid) {
            if (kc.gix < kc.gsx) {
                Compute.lifePerIdx(kc.gix, control, cellGrid);
            }
        }

        @CodeReflection
        static public void compute(final @RO ComputeContext cc, @RO Control ctrl, @RW CellGrid grid) {
            int range = grid.width() * grid.height();
            cc.dispatchKernel(NDRange.of(range), kc -> Compute.life(kc, ctrl, grid));
        }
    }

    @HatTest
    public static void testLife() {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup());//,new OpenCLBackend("INFO,MINIMIZE_COPIES,SHOW_COMPUTE_MODEL"));

        // We oversize the grid by adding 1 to n,e,w and s
        CellGrid cellGrid = CellGrid.create(accelerator,
                17,
                17);

        byte[][] actualGrid = new byte[][]{
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
        };

        // By shifting all cells +1,+1 so we only need to scan 1..width-1, 1..height-1
        // we don't worry about possibly finding cells in 0,n width,n or n,0 height,n
        for (int i = 0; i < cellGrid.height(); i++) {
            for (int j = 0; j < cellGrid.width(); j++) {
                cellGrid.array(((long) i * cellGrid.width()) + j, actualGrid[i][j]);
            }
        }

        Control control = Control.create(accelerator, cellGrid);

        accelerator.compute(cc -> Compute.compute(cc, control, cellGrid));

        byte[][] resultGrid = new byte[][]{
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD},
                {DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  ALIVE, DEAD,  DEAD,  DEAD},
                {DEAD,  ALIVE, ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  ALIVE, ALIVE, ALIVE, DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  ALIVE, ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  ALIVE, DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
                {DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD,  DEAD},
        };

        for (int i = 0; i < cellGrid.height(); i++) {
            for (int j = 0; j < cellGrid.width(); j++) {
                HATAsserts.assertEquals(resultGrid[i][j], cellGrid.array(((long) i * cellGrid.width()) + j));
            }
        }
    }

    /*
     * simplified version of mandel using ArrayView
     */
    @CodeReflection
    public static int mandelCheck(int i, int j, float width, float height, int[] pallette, float offsetx, float offsety, float scale) {
        float x = (i * scale - (scale / 2f * width)) / width + offsetx;
        float y = (j * scale - (scale / 2f * height)) / height + offsety;
        float zx = x;
        float zy = y;
        float new_zx;
        int colorIdx = 0;
        while ((colorIdx < pallette.length) && (((zx * zx) + (zy * zy)) < 4f)) {
            new_zx = ((zx * zx) - (zy * zy)) + x;
            zy = (2f * zx * zy) + y;
            zx = new_zx;
            colorIdx++;
        }
        return colorIdx < pallette.length ? pallette[colorIdx] : 0;
    }

    @CodeReflection
    public static void mandel(@RO KernelContext kc, @RW S32Array2D s32Array2D, @RO S32Array pallette, float offsetx, float offsety, float scale) {
        if (kc.gix < kc.gsx) {
            int[] pal = pallette.arrayView();
            int[][] s32 = s32Array2D.arrayView();
            float width = s32Array2D.width();
            float height = s32Array2D.height();
            float x = ((kc.gix % s32Array2D.width()) * scale - (scale / 2f * width)) / width + offsetx;
            float y = ((kc.gix / s32Array2D.width()) * scale - (scale / 2f * height)) / height + offsety;
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
            s32[kc.gix % s32Array2D.width()][kc.gix / s32Array2D.width()] = color;
        }
    }


    @CodeReflection
    static public void compute(final ComputeContext computeContext, S32Array pallete, S32Array2D s32Array2D, float x, float y, float scale) {

        computeContext.dispatchKernel(
                NDRange.of(s32Array2D.width()*s32Array2D.height()), //0..S32Array2D.size()
                kc -> mandel(kc, s32Array2D, pallete, x, y, scale));
    }

    @HatTest
    public static void testMandel() {
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
                int paletteCheck = mandelCheck(x*subsample, y*subsample, width, height, palletteArray, originX, originY, defaultScale);
                // System.out.print(charPallette9[palletteValue]);
                HATAsserts.assertEquals(paletteCheck, palletteValue);
            }
            // System.out.println();
        }
    }

    /*
     * simplified version of BlackScholes using ArrayView
     */
    @CodeReflection
    public static float[] blackScholesCheck(float s, float x, float t, float r, float v) {
        float expNegRt = (float) Math.exp(-r * t);
        float d1 = (float) ((Math.log(s / x) + (r + v * v * .5f) * t) / (v * Math.sqrt(t)));
        float d2 = (float) (d1 - v * Math.sqrt(t));
        float cnd1 = CND(d1);
        float cnd2 = CND(d2);
        float call = s * cnd1 - expNegRt * x * cnd2;
        float put = expNegRt * x * (1 - cnd2) - s * (1 - cnd1);
        return new float[]{call, put};
    }

    @CodeReflection
    public static void blackScholesKernel(@RO KernelContext kc,
                                          @WO F32Array call,
                                          @WO F32Array put,
                                          @RO F32Array sArray,
                                          @RO F32Array xArray,
                                          @RO F32Array tArray,
                                          float r,
                                          float v) {
        if (kc.gix<kc.gsx){
            float[] callArr = call.arrayView();
            float[] putArr = put.arrayView();
            float[] sArr = sArray.arrayView();
            float[] xArr = xArray.arrayView();
            float[] tArr = tArray.arrayView();

            float expNegRt = (float) Math.exp(-r * tArr[kc.gix]);
            float d1 = (float) ((Math.log(sArr[kc.gix] / xArr[kc.gix]) + (r + v * v * .5f) * tArr[kc.gix]) / (v * Math.sqrt(tArr[kc.gix])));
            float d2 = (float) (d1 - v * Math.sqrt(tArr[kc.gix]));
            float cnd1 = CND(d1);
            float cnd2 = CND(d2);
            float value = sArr[kc.gix] * cnd1 - expNegRt * xArr[kc.gix] * cnd2;
            callArr[kc.gix] = value;
            putArr[kc.gix] = expNegRt * xArr[kc.gix] * (1 - cnd2) - sArr[kc.gix] * (1 - cnd1);
        }
    }

    @CodeReflection
    public static float CND(float input) {
        float x = input;
        if (input < 0f) { // input = Math.abs(input)?
            x = -input;
        }

        float term = 1f / (1f + (0.2316419f * x));
        float term_pow2 = term * term;
        float term_pow3 = term_pow2 * term;
        float term_pow4 = term_pow2 * term_pow2;
        float term_pow5 = term_pow2 * term_pow3;

        float part1 = (1f / (float)Math.sqrt(2f * 3.1415926535f)) * (float)Math.exp((-x * x) * 0.5f);

        float part2 = (0.31938153f * term) +
                (-0.356563782f * term_pow2) +
                (1.781477937f * term_pow3) +
                (-1.821255978f * term_pow4) +
                (1.330274429f * term_pow5);

        if (input >= 0f) {
            return 1f - part1 * part2;
        }
        return part1 * part2;

    }

    @CodeReflection
    public static void blackScholes(@RO ComputeContext cc, @WO F32Array call, @WO F32Array put, @RO F32Array S, @RO F32Array X, @RO F32Array T, float r, float v) {
        cc.dispatchKernel(NDRange.of(call.length()),
                kc -> blackScholesKernel(kc, call, put, S, X, T, r, v)
        );
    }

    static F32Array floatArray(Accelerator accelerator, int size, float low, float high, Random rand) {
        F32Array array = F32Array.create(accelerator, size);
        for (int i = 0; i <size; i++) {
            array.array(i, rand.nextFloat() * (high - low) + low);
        }
        return array;
    }

    @HatTest
    public static void testBlackScholes() {
        int size = 50;
        Random rand = new Random();
        var accelerator = new Accelerator(java.lang.invoke.MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var call = F32Array.create(accelerator, size);
        for (int i = 0; i < call.length(); i++) {
            call.array(i, i);
        }

        var put = F32Array.create(accelerator, size);
        for (int i = 0; i < put.length(); i++) {
            put.array(i, i);
        }

        var S = floatArray(accelerator, size,1f, 100f, rand);
        var X = floatArray(accelerator, size,1f, 100f, rand);
        var T = floatArray(accelerator,size, 0.25f, 10f, rand);
        float r = 0.02f;
        float v = 0.30f;

        accelerator.compute(cc -> blackScholes(cc, call, put, S, X, T, r, v));
        float[] res;
        for (int i = 0; i < call.length(); i++) {
            res = blackScholesCheck(S.array(i), X.array(i), T.array(i), r, v);
            HATAsserts.assertEquals(res[0], call.array(i), 0.0001);
            HATAsserts.assertEquals(res[1], put.array(i), 0.0001);
        }
    }

    /*
     * basic test of local and private buffer ArrayViews
     */
    private interface SharedMemory extends Buffer {
        void array(long index, int value);
        int array(long index);
        Schema<SharedMemory> schema = Schema.of(SharedMemory.class,
                arr -> arr.array("array", 1024));
        static SharedMemory create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static SharedMemory createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }

        default int[] localArrayView() {
            int[] view = new int[1024];
            for (int i = 0; i < 1024; i++) {
                view[i] = this.array(i);
            }
            return view;
        }
    }

    public interface PrivateArray extends Buffer {
        void array(long index, int value);
        int array(long index);
        Schema<PrivateArray> schema = Schema.of(PrivateArray.class,
                arr -> arr.array("array", 16));
        static PrivateArray create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }
        static PrivateArray createPrivate() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }

        default int[] privateArrayView() {
            int[] view = new int[16];
            for (int i = 0; i < 16; i++) {
                view[i] = this.array(i);
            }
            return view;
        }
    }

    @CodeReflection
    public static void squareKernelWithPrivateAndLocal(@RO  KernelContext kc, @RW S32Array s32Array) {
        SharedMemory shared = SharedMemory.createLocal();
        if (kc.gix < kc.gsx){
            int[] arr = s32Array.arrayView();
            arr[kc.gix] += arr[kc.gix];
            // int[] a = new int[4];
            // a[1] = 4;

            PrivateArray priv = PrivateArray.createPrivate();
            int[] privView = priv.privateArrayView();
            privView[0] = 1;
            arr[kc.gix] += privView[0];

            int[] sharedView = shared.localArrayView();
            sharedView[0] = 16;
            arr[kc.gix] += sharedView[0];
        }
    }

    @CodeReflection
    public static void privateAndLocal(@RO ComputeContext cc, @RW S32Array s32Array) {
        cc.dispatchKernel(NDRange.of(s32Array.length()),
                kc -> squareKernelWithPrivateAndLocal(kc, s32Array)
        );
    }

    @HatTest
    public static void testPrivateAndLocal() {

        var accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);//new JavaMultiThreadedBackend());
        var arr = S32Array.create(accelerator, 32);
        for (int i = 0; i < arr.length(); i++) {
            arr.array(i, i);
        }
        accelerator.compute(
                cc -> privateAndLocal(cc, arr)  //QuotableComputeContextConsumer
        );                                     //   extends Quotable, Consumer<ComputeContext>
        for (int i = 0; i < arr.length(); i++) {
            HATAsserts.assertEquals(2 * i + 17, arr.array(i));
        }
    }
}
