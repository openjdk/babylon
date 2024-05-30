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

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;


public class HealingBrush {

    public static int[] getMask(Path path, int width, int height) {
        Polygon polygon = new Polygon();
        for (int i = 0; i < path.length(); i++) {

        XYList.XY xy= (XYList.XY)path.xy(i);
            polygon.addPoint(xy.x() - path.x1 + 1, xy.y() - path.y1 + 1);
        }
        BufferedImage maskImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int[] mask = ((DataBufferInt) (maskImg.getRaster().getDataBuffer())).getData();
        Arrays.fill(mask, 0);
        Graphics2D g = maskImg.createGraphics();
        g.setColor(Color.WHITE);
        g.fillPolygon(polygon);
        return mask;
    }

    public static void heal(Selection selection,
                            int fromDeltaX,
                            int fromDeltaY) {
        int reg_width = 2 + selection.width;
        int reg_height = 2 + selection.height;
        int[] mask = getMask(selection.path, reg_width, reg_height);
        int[] dest = new int[mask.length];
        int[] src = new int[mask.length];
long start = System.currentTimeMillis();
        for (int i = 0; i < mask.length; i++) { //parallel
            int x = i % reg_width;
            int y = i / reg_width;
            src[i] = selection.imageData.data[(selection.path.y1 + y - 1 + fromDeltaY) * selection.imageData.width + selection.path.x1 + x + fromDeltaX];
            if (mask[i] != 0) {
                dest[i] = src[i];
            } else {
                dest[i] = selection.imageData.data[(selection.path.y1 + y - 1) * selection.imageData.width + selection.path.x1 + x];
            }
        }
        System.out.println("heal " +(System.currentTimeMillis()-start)+"ms");

        RGBList srclap = laplacian(src, reg_width, reg_height);
          displayLapacian(srclap, dest, mask);
        solve(dest, mask, srclap, reg_width, reg_height);

start=System.currentTimeMillis();
        for (int i = 0; i < mask.length; i++) { //parallel
            int x = i % reg_width;
            int y = i / reg_width;
            selection.imageData.data[(selection.path.y1 + y - 1) * selection.imageData.width + selection.path.x1 + x] = dest[i];
        }
        System.out.println("heal2 " +(System.currentTimeMillis()-start)+"ms");
    }

    static void solve(int[] dest, int[] mask, RGBList lap_rgb, int width, int height) {
        int r, g, b, v;
        int[] tmp = new int[dest.length];
        System.arraycopy(dest, 0, tmp, 0, tmp.length);
        long start = System.currentTimeMillis();
        for (int i = 0; i < 200; i++) {
            for (int p = 0; p < width * height; p++) { // parallel
                int x = p % width;
                int y = p / width;
                if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && mask[p] != 0) {
                    v = dest[p - 1];
                    r = ((v >> 16) & 0xff);
                    g = ((v >> 8) & 0xff);
                    b = ((v >> 0) & 0xff);
                    v = dest[p + 1];
                    r += ((v >> 16) & 0xff);
                    g += ((v >> 8) & 0xff);
                    b += ((v >> 0) & 0xff);
                    v = dest[p - width];
                    r += ((v >> 16) & 0xff);
                    g += ((v >> 8) & 0xff);
                    b += ((v >> 0) & 0xff);
                    v = dest[p + width];
                    r += ((v >> 16) & 0xff);
                    g += ((v >> 8) & 0xff);
                    b += ((v >> 0) & 0xff);
                    r += (lap_rgb.rgb[p * 3 + 0]);
                    g += (lap_rgb.rgb[p * 3 + 1]);
                    b += (lap_rgb.rgb[p * 3 + 2]);
                    r = (r + 2) / 4;
                    g = (g + 2) / 4;
                    b = (b + 2) / 4;
                    tmp[p] = (((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF));
                }
            }
            int[] swap = tmp;
            tmp = dest;
            dest = swap;
        }
        System.out.println("solve " +(System.currentTimeMillis()-start)+"ms");
    }


    static void displayLapacian(RGBList lap_rgb, int[] dst, int[] mask) {
        for (RGBList.RGB rgb : lap_rgb) {
            if (mask[rgb.idx] != 0) {
                dst[rgb.idx] = (((Math.abs(rgb.r) & 0xFF) << 16) | ((Math.abs(rgb.g) & 0xFF) << 8) | (Math.abs(rgb.b) & 0xFF));
            }
        }
    }

    static RGBList laplacian(int[] src, int width, int height) {
        RGBList rgbList = new RGBList();
        long start = System.currentTimeMillis();
        for (int p = 0; p < width * height; p++) { //parallel
            int x = p % width;
            int y = p / width;
            if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                int v = src[p];
                int r = ((v >> 16) & 0xff) << 2;
                int g = ((v >> 8) & 0xff) << 2;
                int b = ((v >> 0) & 0xff) << 2;

                v = src[p - 1];
                r -= ((v >> 16) & 0xff);
                g -= ((v >> 8) & 0xff);
                b -= ((v >> 0) & 0xff);

                v = src[p + 1];
                r -= ((v >> 16) & 0xff);
                g -= ((v >> 8) & 0xff);
                b -= ((v >> 0) & 0xff);

                v = src[p - width];
                r -= ((v >> 16) & 0xff);
                g -= ((v >> 8) & 0xff);
                b -= ((v >> 0) & 0xff);

                v = src[p + width];
                r -= ((v >> 16) & 0xff);
                g -= ((v >> 8) & 0xff);
                b -= ((v >> 0) & 0xff);
                rgbList.add(r, g, b);
            } else {
                rgbList.add(0, 0, 0);
            }
        }
        System.out.println("laplacian " +(System.currentTimeMillis()-start)+"ms");
        return rgbList;
    }


    public static Point getBestMatch(Selection selection) {
        Point offset = null;
        if (selection.path.length() != 0) {
            int xmin = Math.max(0, selection.path.x1 - selection.width * 3);
            int ymin = Math.max(0, selection.path.y1 - selection.height * 3);
            int xmax = Math.min(selection.imageData.width, selection.path.x2 + selection.width * 3);
            int ymax = Math.min(selection.imageData.height, selection.path.y2 + selection.height * 3);

            RGBList orig_rgb = extractCurve(selection);

            RGBList comp = new RGBList(orig_rgb);

            float min = Float.MAX_VALUE;
            int bestdx = -11111, bestdy = 0;
            long start = System.currentTimeMillis();

            for (int y = ymin; y < ymax - selection.height; y++) {
                for (int x = xmin; x < xmax - selection.width; x++) {
                    if (!selection.contains(x, y)) { // don't search inside the area we are healing
                        int sdx = x - selection.path.x1;
                        int sdy = y - selection.path.y1;

                        for (int i=0;i<selection.path.length();i++){
                            XYList.XY xy= (XYList.XY)selection.path.xy(i);
                            comp.setRGB(xy.idx(), selection.imageData.data[sdy * selection.imageData.width + sdx + xy.y() * selection.imageData.width + xy.x()]);
                        }

                        float sum = 0;
                        for (RGBList.RGB rgb : orig_rgb) {
                            int dx = comp.rgb[rgb.idx * 3 + 0] - rgb.r;
                            int dy = comp.rgb[rgb.idx * 3 + 1] - rgb.g;
                            int dz = comp.rgb[rgb.idx * 3 + 2] - rgb.b;
                            sum += dx * dx + dy * dy + dz * dz;
                        }

                        if (sum < min) {
                            min = sum;
                            bestdx = x - selection.path.x1;
                            bestdy = y - selection.path.y1;
                        }
                    }
                }
            }
            System.out.println("search " +(System.currentTimeMillis()-start)+"ms");
            offset = new Point(bestdx, bestdy);
        }
        return offset;

    }
    static RGBList extractCurve(Selection selection) {
        RGBList rgbList = new RGBList();
        for (int i=0;i<selection.path.length();i++){
        XYList.XY xy= (XYList.XY)selection.path.xy(i);
            rgbList.addRGB(selection.imageData.data[xy.y() * selection.imageData.width + xy.x()]);
        }
        return rgbList;
    }

    static void extractCurve(RGBList rgbList, Selection selection,
                                int dx,
                                int dy) {
        for (int i=0;i<selection.path.length();i++){
            XYList.XY xy= (XYList.XY)selection.path.xy(i);
            rgbList.setRGB(xy.idx(), selection.imageData.data[dy * selection.imageData.width + dx + xy.y() * selection.imageData.width + xy.x()]);
        }

    }


}
