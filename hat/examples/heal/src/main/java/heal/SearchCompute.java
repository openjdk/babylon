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
import java.awt.geom.Point2D;
import java.lang.runtime.CodeReflection;
import java.util.stream.IntStream;

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
    public static void bestFitCore(int id,
                                  S32Array2D s32Array2D,
                                  RGBList rgbList,
                                  Box searchBox,
                                  Box selBox,
                                  XYList selectionXYList,
                                  F32Array sumArray) {
        int x = searchBox.x1() + id % (searchBox.width());
        int y = searchBox.y1() + id / (searchBox.width());
        float sum = 0;
        // lets not search inside the area we are healing :)
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
            for (int i = 0; i < selectionXYList.length(); i++) {
                var xy = selectionXYList.xy(i);
                var rgb = rgbList.rgb(i);
                int rgbInt = s32Array2D.array(offset + xy.y() * s32Array2D.width() + xy.x());
                int dr = red(rgbInt) - rgb.r();
                int dg = green(rgbInt) - rgb.g();
                int db = blue(rgbInt) - rgb.b();
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
                                  RGBList rgbList,
                                  Box searchBox,
                                  Box selectionBox,
                                  XYList selectionXYList,
                                  F32Array sumArray) {
        bestFitCore(kc.x, s32Array2D, rgbList, searchBox, selectionBox, selectionXYList, sumArray);
    }

    @CodeReflection
    public static void  bestFitCompute(ComputeContext cc,
             Point offset, S32Array2D s32Array2D,  RGBList rgbList, Box searchBox, Box selectionBox, XYList selectionXYList){

        F32Array sumArrayF32 = F32Array.create(cc.accelerator, searchBox.area());

        cc.dispatchKernel(searchBox.area(),
                kc -> bestFitKernel(kc,  s32Array2D, rgbList, searchBox, selectionBox, selectionXYList, sumArrayF32)
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
            // Create a search box of pad * selection (w & h), but constrain to bounds of the image
            int pad = 4;
            int padx = selection.width() * pad;
            int pady = selection.height() * pad;
            int x1 = Math.max(0, selection.x1() - padx);
            int y1 = Math.max(0, selection.y1() - pady);
            int x2 = Math.min(imageData.width(), selection.x2() + padx) - selection.width();
            int y2 = Math.min(imageData.height(), selection.y2() + pady) - selection.height();

            /*
             Map xy's in the path to list of RGB values at those coordinates.
             //Renderscript
             float3 __attribute__((kernel))extractBorder(int2 in) {
                return convert_float3(rsGetElementAt_uchar4(image, in.x, in.y).xyz);
             }
            */

            RGBList mappedRGBList = RGBList.create(accelerator, selection.pointList.size());
            XYList  mappedSelectionXYList = XYList.create(accelerator, selection.pointList.size());
          /*  int xys = selection.xyList.length();
            int pnts = selection.pointList.size();
            if (xys != pnts){
                for (int i = 0; i < xys; i++) {
                    XYList.XY xy = selection.xyList.xy(i);
                    int x = xy.x();//(int)point.getX();
                    int y = xy.y();//int)point.getY();
                    Point p = selection.pointList.get(i);
                    System.out.println("x: " + x + " y: " + y+ " p: "+p);
                }
            }*/
           // IntStream s1 = IntStream.range(0,selection.xyList.length()).parallel();
            IntStream s2 = IntStream.range(0,selection.pointList.size()).parallel();
            /*
            s1.forEach(i->{ //Old school! Maybe a Kernel?
                XYList.XY xy = selection.xyList.xy(i);
              //  Point2D point=selection.pointList.get(i);
                int x = xy.x();//(int)point.getX();
                int y = xy.y();//int)point.getY();
                var to = mappedSelectionXYList.xy(i);
                var rgb = mappedRGBList.rgb(i);
                var rgbint = imageData.array((long) y * imageData.width() + x);
                to.x(x);
                to.y(y);
                rgb.r(red(rgbint));
                rgb.g(green(rgbint));
                rgb.b(blue(rgbint));
            });*/
            s2.forEach(i->{ //Old school! Maybe a Kernel?
              //  XYList.XY xy = selection.xyList.xy(i);
                  Point point=selection.pointList.get(i);
                int x = point.x;
                int y = point.y;//xy.y();//int)point.getY();
                var to = mappedSelectionXYList.xy(i);
                var rgb = mappedRGBList.rgb(i);
                var rgbint = imageData.array((long) y * imageData.width() + x);
                to.x(x);
                to.y(y);
                rgb.r(red(rgbint));
                rgb.g(green(rgbint));
                rgb.b(blue(rgbint));
            });

            Box  mappedSearchBox = Box.create(accelerator, x1,y1,x2,y2);
            Box  mappedSelectionBox = Box.create(accelerator, selection.x1(), selection.y1(), selection.x2(), selection.y2());

            S32Array2D s32Array2D = S32Array2D.create(accelerator,imageData.width(),imageData.height());
            s32Array2D.copyFrom(imageData.arrayOfData);

            accelerator.compute(cc->
                    SearchCompute.bestFitCompute(cc,
                    offset,
                    s32Array2D,
                    mappedRGBList,
                    mappedSearchBox,
                    mappedSelectionBox,
                    mappedSelectionXYList
            ));
            System.out.println("total search " + (System.currentTimeMillis() - hatStart) + "ms");
        }
        return offset;
    }
}
