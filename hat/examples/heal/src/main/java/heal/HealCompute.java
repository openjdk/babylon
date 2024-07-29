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

import java.awt.Point;
import java.lang.runtime.CodeReflection;

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
public class HealCompute {

    static int red(int rgb) {
        return (rgb >> 16) & 0xff;
    }


    static int green(int rgb) {
        return (rgb >> 8) & 0xff;
    }


    static int blue(int rgb) {
        return rgb & 0xff;
    }

    @CodeReflection
    static int rgb(int r, int g, int b) {
        return ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
    }


    public static void heal(Accelerator accelerator, ImageData imageData, Selection selection, Point healPositionOffset) {
        long start = System.currentTimeMillis();
        Mask mask = new Mask(selection);
        var src = new int[mask.data.length];
        var dest = new int[mask.data.length];

        for (int i = 0; i < mask.data.length; i++) { //parallel
            int x = i % mask.width;
            int y = i / mask.width;
            src[i] = imageData.get(selection.x1() + x + healPositionOffset.x, selection.y1() + y - 1 + healPositionOffset.y);
            dest[i] = (mask.data[i] != 0)
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
        for (int i = 0; i < mask.data.length; i++) { //parallel
            int x = i % mask.width;
            int y = i / mask.width;
            imageData.set(selection.x1() + x, selection.y1() + y - 1, dest[i]);
        }
        System.out.println("heal2 " + (System.currentTimeMillis() - start) + "ms");
    }

}
