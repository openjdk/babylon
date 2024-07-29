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
 * Based on code from HealingBrush renderscript example
 *
 * https://github.com/yongjhih/HealingBrush/tree/master
 *
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package heal;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.F32Array;
import hat.buffer.S32Array2D;
import java.awt.Point;
import java.lang.runtime.CodeReflection;
import java.util.stream.IntStream;

public class Compute {
    /*
 From the original renderscript

 float3 __attribute__((kernel)) solve1(uchar in, uint32_t x, uint32_t y) {
  if (in > 0) {
     float3 k = getF32_3(dest1, x - 1, y);
     k += getF32_3(dest1, x + 1, y);
     k += getF32_3(dest1, x, y - 1);
     k += getF32_3(dest1, x, y + 1);
     k += getF32_3(laplace, x, y);
     k /= 4;
     return k;
  }
  return rsGetElementAt_float3(dest1, x, y);;
}


float3 __attribute__((kernel)) solve2(uchar in, uint32_t x, uint32_t y) {
  if (in > 0) {
    float3 k = getF32_3(dest2, x - 1, y);
    k += getF32_3(dest2, x + 1, y);
    k += getF32_3(dest2, x, y - 1);
    k += getF32_3(dest2, x, y + 1);
       k += getF32_3(laplace, x, y);
       k /= 4;
       return k;
  }
  return getF32_3(dest2, x, y);;
}
*/

    public static void heal(Accelerator accelerator, ImageData imageData, Selection selection, Point healPositionOffset) {
        long start = System.currentTimeMillis();
        Selection.Mask mask = selection.getMask();
        var src = new int[mask.maskRGBData.length];
        var dest = new int[mask.maskRGBData.length];

        for (int i = 0; i < mask.maskRGBData.length; i++) { //parallel
            int x = i % mask.width;
            int y = i / mask.width;
            src[i] = imageData.get(selection.x1() + x + healPositionOffset.x, selection.y1() + y - 1 + healPositionOffset.y);
            dest[i] = (mask.maskRGBData[i] != 0)
                    ? src[i]
                    : imageData.get(+selection.x1() + x, selection.y1() + y - 1);
        }

        System.out.println("mask " + (System.currentTimeMillis() - start) + "ms");
/*
        int[] stencil = new int[]{-1, 1, -mask.width, mask.width};

        int[] laplaced = new int[dest.length];

        boolean laplacian = true;
        if (laplacian) {
            start = System.currentTimeMillis();

            for (int p = 0; p < src.length; p++) { //parallel
                int x = p % mask.width;
                int y = p / mask.width;

                int r = 0, g = 0, b = 0;
                if (x > 0 && x < mask.width - 1 && y > 0 && y < mask.height - 1) {
                    for (int offset : stencil) {
                        var v = src[p + offset];
                        r += red(v);
                        g += green(v);
                        b += blue(v);
                    }
                }
                laplaced[p] = rgb(r, g, b);
            }
        }

        System.out.println("laplacian " + (System.currentTimeMillis() - start) + "ms");
        boolean solve = false;
        if (solve) {

            var tmp = new int[dest.length];
            start = System.currentTimeMillis();
            for (int i = 0; i < 500; i++) {
                for (int p = 0; p < mask.width * mask.height; p++) { // parallel
                    int x = p % mask.width;
                    int y = p / mask.width;
                    if (x > 0 && x < mask.width - 1 && y > 0 && y < mask.height - 1 && mask.data[p] != 0) {
                        //   var rgb = rgbList.rgb(p);

                        var r = red(laplaced[i]);//rgb.r();
                        var g = green(laplaced[i]);//rgb.g();
                        var b = blue(laplaced[i]);//rgb.b();
                        for (int offset : stencil) {
                            var v = dest[p + offset];
                            r += red(v);
                            g += green(v);
                            b += blue(v);
                        }
                        tmp[p] = rgb((r + 2) / 4, (g + 2) / 4, (b + 2) / 4);
                    }
                }
                var swap = tmp;
                tmp = dest;
                dest = swap;
            }
            System.out.println("solve " + (System.currentTimeMillis() - start) + "ms");
        }
*/
        start = System.currentTimeMillis();
        for (int i = 0; i < mask.maskRGBData.length; i++) { //parallel
            int x = i % mask.width;
            int y = i / mask.width;
            imageData.set(selection.x1() + x, selection.y1() + y - 1, dest[i]);
        }
        System.out.println("heal2 " + (System.currentTimeMillis() - start) + "ms");
    }

    @CodeReflection
    static int red(int rgb) {
        return (rgb >> 16) & 0xff;
    }

    @CodeReflection
    static int green(int rgb) {
        return (rgb >> 8) & 0xff;
    }

    @CodeReflection
    static int blue(int rgb) {
        return rgb & 0xff;
    }

    @CodeReflection
    public static void bestFitCore(int id,
                                  S32Array2D s32Array2D,
                                  Box searchBox,
                                  Box selBox,
                                  XYRGBList selectionXYRGBList,
                                  F32Array sumArray) {
        int x = searchBox.x1() + id % searchBox.width();
        int y = searchBox.y1() + id / searchBox.width();
        float sum = 0;
        // don't search inside the area we are healing :)
        if  (x > selBox.x2() || x + selBox.width() < selBox.x1() || y > selBox.y2() || y + selBox.height() < selBox.y1()){
            /*
            Renderscript version
            float __attribute__((kernel)) bordercorrelation(uint32_t x, uint32_t y) {
               float sum = 0;
               for(int i = 0 ; i < borderLength; i++) {
                  int2  coord = rsGetElementAt_int2(border_coords,i);
                  float3 orig = convert_float3(rsGetElementAt_uchar4(image, coord.x + x, coord.y + y).xyz);
                  float3 candidate = rsGetElementAt_float3(border, i).xyz;
                  sum += distance(orig, candidate);
               }
               return sum;
            }
            */
            int offset = (y - selBox.y1()) * s32Array2D.width() + (x - selBox.x1());
            for (int i = 0; i < selectionXYRGBList.length(); i++) {
                var xyrgb = selectionXYRGBList.xyrgb(i);
                int rgbInt = s32Array2D.array(offset + xyrgb.y() * s32Array2D.width() + xyrgb.x());
                int dr = red(rgbInt) - xyrgb.r();
                int dg = green(rgbInt) - xyrgb.g();
                int db = blue(rgbInt) - xyrgb.b();
                sum += dr * dr + dg * dg + db * db;
            }
        }else{
            sum = Float.MAX_VALUE;
        }
        sumArray.array(id,sum);
    }

    @CodeReflection
    public static void bestFitKernel(KernelContext kc,
                                  S32Array2D s32Array2D,
                                  Box searchBox,
                                  Box selectionBox,
                                  XYRGBList selectionXYRGBList,
                                  F32Array sumArray) {
        bestFitCore(kc.x, s32Array2D, searchBox, selectionBox, selectionXYRGBList, sumArray);
    }

    @CodeReflection
    public static void  bestFitCompute(ComputeContext cc,
             Point offset, S32Array2D s32Array2D, Box searchBox, Box selectionBox, XYRGBList selectionXYRGBList){

        F32Array sumArrayF32 = F32Array.create(cc.accelerator, searchBox.area());

        cc.dispatchKernel(searchBox.area(),
                kc -> bestFitKernel(kc,  s32Array2D, searchBox, selectionBox, selectionXYRGBList, sumArrayF32)
        );
        float[] sumArray = new float[searchBox.area()];

        sumArrayF32.copyTo(sumArray);
        float minSoFar = Float.MAX_VALUE;

        int id = sumArray.length;
        for (int i = 0; i < sumArray.length; i++) {
            float value = sumArray[i];
            if (value < minSoFar) {
                id = i;
                minSoFar = value;
            }
        }
        int x = searchBox.x1() + (id % searchBox.width());
        int y = searchBox.y1() + (id / searchBox.width());
        offset.setLocation(x - selectionBox.x1(),y - selectionBox.y1());
    }

    public static Point getOffsetOfBestMatch(Accelerator accelerator, ImageData imageData, Selection selection) {
        final Point offset  =new Point(0,0);
        if (!selection.pointList.isEmpty()) {
            long hatStart = System.currentTimeMillis();
            XYRGBList xyrgbList = XYRGBList.create(accelerator, selection.pointList.size());
            /*
             Map Point's in the selection to list of XYRGB values at those coordinates.
             //Renderscript
             float3 __attribute__((kernel))extractBorder(int2 in) {
                return convert_float3(rsGetElementAt_uchar4(image, in.x, in.y).xyz);
             }
            */

             IntStream.range(0,selection.pointList.size()).parallel().forEach(i->{
                 Point point=selection.pointList.get(i);
                 var to = xyrgbList.xyrgb(i);
                 var rgbint = imageData.array((long) point.y * imageData.width() + point.x);
                 to.x(point.x);
                 to.y(point.y);
                 to.r(red(rgbint));
                 to.g(green(rgbint));
                 to.b(blue(rgbint));
            });

            // Create a search box of pad * selection (w & h), but constrain to bounds of the image
            /*
             *            +----------------------+Image
             *            |                      |
             *       +.W..W--W--W--W--W--W--W-+  |
             *       .  ^ |                   |  |
             *       .    |    +-W-+Selection |  |
             *       .    |   H|   |          |  |
             *       .    |    +---+          |  |
             *       .    |                   |  |
             *       .    |                   |  |
             *       .  v +-------------------|--+
             *       .    <- SearchBoxWidth ->.
             *       +........................+
             */
            int pad = 8;
            int padx = selection.width() * pad;
            int pady = selection.height() * pad;

            Box searchBox = Box.create(accelerator,
                    Math.max(0, selection.x1() - padx),
                    Math.max(0, selection.y1() - pady),
                    Math.min(imageData.width(), selection.x2() + padx) - selection.width(),
                    Math.min(imageData.height(), selection.y2() + pady) - selection.height()
            );
            Box selectionBox = Box.create(accelerator, selection.x1(), selection.y1(), selection.x2(), selection.y2());

            S32Array2D s32Array2D = S32Array2D.create(accelerator,imageData.width(),imageData.height());
            s32Array2D.copyFrom(imageData.arrayOfData);

            accelerator.compute(cc->
                    Compute.bestFitCompute(cc, offset, s32Array2D, searchBox, selectionBox, xyrgbList
            ));
            System.out.println("total search " + (System.currentTimeMillis() - hatStart) + "ms");
        }
        return offset;
    }
}
