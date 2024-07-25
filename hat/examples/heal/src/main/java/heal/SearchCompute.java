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
import hat.buffer.S32Array;
import hat.buffer.S32Array2D;

import java.awt.Point;
import java.lang.runtime.CodeReflection;
import java.util.stream.IntStream;

/*
 From the original renderscript

float3 __attribute__((kernel))extractBorder(int2 in) {
  return convert_float3(rsGetElementAt_uchar4(image, in.x, in.y).xyz);
}

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
public class SearchCompute {
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
    public static float getSum(S32Array2D imageData, Box selectionBox, XYList selectionXYList, RGBList selectionRgbList, int x, int y) {
        int offset = (y - selectionBox.y1()) * imageData.width() + (x - selectionBox.x1());
        float sum = 0;
        for (int i = 0; i < selectionXYList.length(); i++) {
            var xy = selectionXYList.xy(i);
            var rgb = selectionRgbList.rgb(i);
            int rgbFromImage = imageData.array(offset + xy.y() * imageData.width() + xy.x());
            int dr = red(rgbFromImage) - rgb.r();
            int dg = green(rgbFromImage) - rgb.g();
            int db = blue(rgbFromImage) - rgb.b();
            sum += dr * dr + dg * dg + db * db;
        }
        return sum;
    }
    @CodeReflection
    public static boolean isInSelection(Box selectionBox, int x, int y) {
        int selectionBoxWidth = selectionBox.x2() - selectionBox.x1();
        int selectionBoxHeight = selectionBox.y2() - selectionBox.y1();
        return (!(x > selectionBox.x2() || x + selectionBoxWidth < selectionBox.x1() || y > selectionBox.y2() || y + selectionBoxHeight < selectionBox.y1()));
    }

    public static Point original(ImageData imageData, RGBList rgbList, Box searchBox, Box selectionBox, XYList selectionXYList) {
        float minSoFar = Float.MAX_VALUE;
        Point bestSoFar = new Point(0, 0);
        for (int y = searchBox.y1(); y < searchBox.y2(); y++) {
            for (int x = searchBox.x1(); x < searchBox.x2(); x++) {
                if (!isInSelection(selectionBox, x, y)) {// don't search inside the area we are healing
                    float sum = getSum(imageData, selectionBox, selectionXYList, rgbList, x, y);
                    if (sum < minSoFar) {
                        minSoFar = sum;
                        bestSoFar.setLocation(x - selectionBox.x1(), y - selectionBox.y1());
                    }
                }
            }
        }
        return bestSoFar;
    }


    public static Point sequential(ImageData imageData, RGBList rgbList, Box searchBox, Box selectionBox, XYList selectionXYList) {
        int searchBoxWidth = searchBox.x2() - searchBox.x1();
        int searchBoxHeight = searchBox.y2() - searchBox.y1();
        int range = searchBoxWidth * searchBoxHeight;
        float minSoFar = Float.MAX_VALUE;
        int bestId = range+1;
        for (int id = 0; id < range; id++) {
            int x = searchBox.x1() + id % searchBoxWidth;
            int y = searchBox.y1() + id / searchBoxWidth;
            if (!isInSelection(selectionBox, x, y)) {// don't search inside the area we are healing
                float sum = getSum(imageData, selectionBox, selectionXYList, rgbList, x, y);
                if (sum < minSoFar) {
                    minSoFar = sum;
                    bestId = id;

                }
            }
        }
        int x = searchBox.x1() + (bestId % searchBoxWidth);
        int y = searchBox.y1() + (bestId / searchBoxWidth);
        return new Point(x - selectionBox.x1(), y - selectionBox.y1());
    }


    public static Point parallel(ImageData imageData, RGBList rgbList, Box searchBox, Box selectionBox, XYList selectionXYList) {
        int searchBoxWidth = searchBox.x2() - searchBox.x1();
        int searchBoxHeight = searchBox.y2() - searchBox.y1();
        int range = searchBoxWidth * searchBoxHeight;
        float[] sumArray = new float[range];
        IntStream.range(0, range).parallel().forEach(id -> {
            int x = searchBox.x1() + id % searchBoxWidth;
            int y = searchBox.y1() + id / searchBoxWidth;
            if (isInSelection(selectionBox, x, y)) {// don't search inside the area we are healing
                sumArray[id] = Float.MAX_VALUE;
            } else {
                sumArray[id] = getSum(imageData, selectionBox, selectionXYList, rgbList, x, y);
            }
        });
        float minSoFar = Float.MAX_VALUE;
        int id = sumArray.length + 1;
        for (int i = 0; i < sumArray.length; i++) {
            float value = sumArray[i];
            if (value < minSoFar) {
                id = i;
                minSoFar = value;
            }
        }
        int x = searchBox.x1() + (id % searchBoxWidth);
        int y = searchBox.y1() + (id / searchBoxWidth);
        return new Point(x - selectionBox.x1(), y - selectionBox.y1());
    }

    /*
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
    @CodeReflection
    public static void bestKernel(KernelContext kc,
                                  S32Array2D s32Array2D,
                                  RGBList rgbList,
                                  Box searchBox,
                                  Box selectionBox,
                                  XYList selectionXYList,
                                  F32Array sumArray) {
        int id = kc.x;
        int searchBoxWidth = searchBox.x2() - searchBox.x1();
        int x = searchBox.x1() + id % searchBoxWidth;
        int y = searchBox.y1() + id / searchBoxWidth;
        if (isInSelection(selectionBox, x, y)) {// don't search inside the area we are healing
            sumArray.array(id, Float.MAX_VALUE);
        } else {
            sumArray.array(id,  getSum(s32Array2D, selectionBox, selectionXYList, rgbList, x, y));
        }
    }

    @CodeReflection
    public static int getMinIdx(float[] sumArray){
        float minSoFar = Float.MAX_VALUE;
        int id = sumArray.length;
        for (int i = 0; i < sumArray.length; i++) {
            float value = sumArray[i];
            if (value < minSoFar) {
                id = i;
                minSoFar = value;
            }
        }
        return id;
    }

    @CodeReflection
    public static void  bestCompute(ComputeContext cc,
                                  S32Array2D s32Array2D,  RGBList rgbList,
                                    Box searchBox, Box selectionBox, XYList selectionXYList, XY result){

        int searchBoxWidth = searchBox.x2() - searchBox.x1();
        int searchBoxHeight = searchBox.y2() - searchBox.y1();
        int range = searchBoxWidth * searchBoxHeight;

        F32Array sumArrayF32 = F32Array.create(cc.accelerator, range);

        cc.dispatchKernel(range,
                kc -> bestKernel(kc,  s32Array2D,rgbList, searchBox, selectionBox, selectionXYList, sumArrayF32));

        float[] sumArray = new float[range];
        sumArrayF32.copyTo(sumArray);
        int id = getMinIdx(sumArray);


        int x = searchBox.x1() + (id % searchBoxWidth);
        int y = searchBox.y1() + (id / searchBoxWidth);
        result.x(x - selectionBox.x1());
        result.y(y - selectionBox.y1());
        //return new Point(x - selectionBox.x1(), y - selectionBox.y1());
    }

    public static Point getOffsetOfBestMatch(Accelerator accelerator,ImageData imageData, Path selectionPath) {
        Point offset = null;
        if (selectionPath.xyList.length() != 0) {
            /*
            Walk the list of xy coordinates in the path and extract a list of RGB values
            for those coordinates.
             */
            RGBListImpl rgbList = new RGBListImpl();
            for (int i = 0; i < selectionPath.xyList.length(); i++) {
                XYList.XY xy = selectionPath.xyList.xy(i);
                rgbList.addRGB(imageData.array(xy.y() * imageData.width() + xy.x()));
            }

            /*
              Create a search box of pad * selection (w & h), but constrain the box to bounds of the image
             */
            int pad = 4;
            int padx = selectionPath.width() * pad;
            int pady = selectionPath.height() * pad;
            int x1 = Math.max(0, selectionPath.x1() - padx);
            int y1 = Math.max(0, selectionPath.y1() - pady);
            int x2 = Math.min(imageData.width(), selectionPath.x2() + padx) - selectionPath.width();
            int y2 = Math.min(imageData.height(), selectionPath.y2() + pady) - selectionPath.height();
            BoxImpl searchBox = new BoxImpl(x1, y1, x2, y2);
            long searchStart = System.currentTimeMillis();

            boolean useOriginal = true;
            boolean useSequential = true;
            // if (useOriginal) {
            long originalStart = System.currentTimeMillis();
            Box selectionBox = new BoxImpl(selectionPath.x1(), selectionPath.y1(), selectionPath.x2(), selectionPath.y2());
            XYList selectionXYList = selectionPath.xyList;
            offset = original(imageData, rgbList, searchBox, selectionBox, selectionXYList);
            System.out.println("original search " + (System.currentTimeMillis() - originalStart) + "ms");
            //  } else {
            //   if (useSequential) {
            long sequentialStart = System.currentTimeMillis();
            offset = sequential(imageData, rgbList, searchBox, selectionBox, selectionXYList);
            System.out.println("sequential search " + (System.currentTimeMillis() - sequentialStart) + "ms");
            //    } else {
            long parallelStart = System.currentTimeMillis();
            offset = parallel(imageData, rgbList, searchBox, selectionBox, selectionXYList);
            System.out.println("parallel search " + (System.currentTimeMillis() - parallelStart) + "ms");
            // }

            //All data passed to accelerator needs to be iface mapped segments
            RGBList mappedRGBList = RGBList.create(accelerator,rgbList.length);
            for (int i=0;i<rgbList.length; i++){
                var from = rgbList.rgb(i);
                var to = mappedRGBList.rgb(i);
                to.r(from.r());
                to.g(from.g());
                to.b(from.b());
            }

            Box  mappedSearchBox = Box.create(accelerator, searchBox.x1(),searchBox.y1(),searchBox.x2(),searchBox.y2());
            Box  mappedSelectionBox = Box.create(accelerator, selectionPath.x1(),selectionPath.y1(),selectionPath.x2(),selectionPath.y2());
            XY result = XY.create(accelerator,0,0);

            XYList  mappedSelectionXYList = XYList.create(accelerator,selectionPath.xyList.length());
            for (int i=0;i<mappedSelectionXYList.length();i++){
                var from = selectionPath.xyList.xy(i);
                var to = mappedSelectionXYList.xy(i);
                to.x(from.x());
                to.y(from.y());
            }
            S32Array2D s32Array2D = S32Array2D.create(accelerator,imageData.width(),imageData.height());
            s32Array2D.copyFrom(imageData.arrayOfData);
            long hatStart = System.currentTimeMillis();
            accelerator.compute(cc->SearchCompute.bestCompute(cc,s32Array2D,mappedRGBList,mappedSearchBox,
                    mappedSelectionBox,
                    mappedSelectionXYList,
                    result
            ));

            System.out.println("offset "+offset+ " result"+result.x()+","+result.y());
            System.out.println("hat search " + (System.currentTimeMillis() - hatStart) + "ms");


            System.out.println("total search " + (System.currentTimeMillis() - searchStart) + "ms");
        }
        return offset;
    }
}
