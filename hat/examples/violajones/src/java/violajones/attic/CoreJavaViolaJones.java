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
package violajones.attic;

import hat.buffer.F32Array2D;
import violajones.buffers.RgbS08x3Image;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public class CoreJavaViolaJones {


    public static int b2i(int v) {
        return v < 0 ? 256 + v : v;
    }

    public static int grey(int r, int g, int b) {
        return (29 * b2i(r) + 60 * b2i(g) + 11 * b2i(b)) / 100;
    }

    public static void rgbToGreyKernel(int id, byte[] rgbBytes, int[] greyInts) {
        byte r = rgbBytes[id * 3 + 0];
        byte g = rgbBytes[id * 3 + 1];
        byte b = rgbBytes[id * 3 + 2];
        greyInts[id] = grey(r, g, b);
    }


    public static void rgbToGreyKernel(int id, RgbS08x3Image rgbImage, F32Array2D floatImage) {
        byte r = rgbImage.data(id * 3L + 0);
        byte g = rgbImage.data(id * 3L + 1);
        byte b = rgbImage.data(id * 3L + 2);
        float grey = grey(r, g, b);
        floatImage.array(id, grey);
    }

    public static void integralColKernel(int id, int[] greyInts, int width, int height, float[] integral, float[] integralSq) {
        float greyValue = greyInts[id];
        float greyValueSq = greyValue * greyValue;
        integralSq[id] = greyValueSq + integralSq[id - width];
        integral[id] = greyValue + integral[id - width];
    }

    public static void integralColKernel(int id, int width, F32Array2D greyFloats, F32Array2D integral, F32Array2D integralSq) {
        float greyValue = greyFloats.array(id);
        float greyValueSq = greyValue * greyValue;
        integralSq.array(id, greyValueSq + integralSq.array(id - width));
        integral.array(id, greyValue + integral.array(id - width));
    }


    public static void integralRowKernel(int id, float[] integral, float[] integralSq) {
        integral[id] = integral[id] + integral[id - 1];
        integralSq[id] = integralSq[id] + integralSq[id - 1];
    }

    public static void integralRowKernel(int id, F32Array2D integral, F32Array2D integralSq) {
        integral.array(id, integral.array(id) + integral.array(id - 1));
        integralSq.array(id, integralSq.array(id) + integralSq.array(id - 1));
    }

    /*
                        A +-------+ B
                          |       |       D-B-C+A
                        C +-------+ D
                  */
    static float gradient(
            float[] image, //
            int imageWidth,
            int x,
            int y,
            int w,
            int h) {
        float A = image[(y * imageWidth) + x];
        float D = image[((y + h) * imageWidth) + x + w];
        float C = image[((y + h) * imageWidth) + x];
        float B = image[((y * imageWidth) + x + w)];
        return D - B - C + A;
    }

    static float gradient(
            MemorySegment image, //
            int imageWidth,
            int x,
            int y,
            int w,
            int h) {
        float A = image.get(JAVA_FLOAT, (((long) y * imageWidth) + x) * 4);
        float D = image.get(JAVA_FLOAT, (((long) (y + h) * imageWidth) + x + w) * 4);
        float C = image.get(JAVA_FLOAT, (((long) (y + h) * imageWidth) + x) * 4);
        float B = image.get(JAVA_FLOAT, (((long) y * imageWidth) + x + w) * 4);
        return D - B - C + A;
    }

    /*
                           A +-------+ B
                             |       |       D-B-C+A
                           C +-------+ D
                     */
    static float gradient(
            F32Array2D image, //
            int x,
            int y,
            int w,
            int h) {
        float A = image.get(x, y);
        float D = image.get(x + w, y + h);
        float C = image.get(x, y + h);
        float B = image.get(x + w, y);
        return D - B - C + A;
    }

    static long rgbToGreyScale(byte[] rgb, int[] grey) {
        long start = System.currentTimeMillis();
        for (int i = 0; i < grey.length; i++) {
            rgbToGreyKernel(i, rgb, grey);
        }

        return System.currentTimeMillis() - start;
    }

    static long rgbToGreyScale(RgbS08x3Image rgb, F32Array2D grey) {
        long start = System.currentTimeMillis();
        int size = grey.width() * grey.height();

        for (int i = 0; i < size; i++) {
            rgbToGreyKernel(i, rgb, grey);
        }

        return System.currentTimeMillis() - start;
    }


    static long integralImage(int[] grey, int imageWidth, int imageHeight, float[] integral, float[] integralSq) {

        long start = System.currentTimeMillis();
        // The col pass can create the integral and intergralsq cols and populate the 'square'
        for (int x = 0; x < imageWidth; x++) {
            for (int y = 1; y < imageHeight; y++) {
                int monoOffset = (y * imageWidth) + x;
                int greyValue = grey[monoOffset];
                int greyValueSq = greyValue * greyValue;
                integralSq[monoOffset] = greyValueSq + integralSq[monoOffset - imageWidth];
                integral[monoOffset] = greyValue + integral[monoOffset - imageWidth];
            }
        }

        for (int y = 0; y < imageHeight; y++) {
            for (int x = 1; x < imageWidth; x++) {
                int monoOffset = (y * imageWidth) + x;
                integral[monoOffset] = integral[monoOffset] + integral[monoOffset - 1];
                integralSq[monoOffset] = integralSq[monoOffset] + integralSq[monoOffset - 1];
            }
        }

        return System.currentTimeMillis() - start;
    }

    public static long createIntegralImage(int[] greyInts, int width, int height, float[] integral, float[] integralSq) {
        long start = System.currentTimeMillis();

        // The col pass can create the integral and intergralsq cols and populate the 'square'
        for (int x = 0; x < width; x++) {
            for (int y = 1; y < height; y++) {
                integralColKernel((y * width) + x, greyInts, width, height, integral, integralSq);
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width; x++) {
                integralRowKernel((y * width) + x, integral, integralSq);
            }
        }
        return System.currentTimeMillis() - start;
    }

    public static long createIntegralImage(F32Array2D greyFloats, F32Array2D integral, F32Array2D integralSq) {
        long start = System.currentTimeMillis();
        int width = greyFloats.width();
        int height = greyFloats.height();

        // The col pass can create the integral and intergralsq cols and populate the 'square'
        for (int x = 0; x < width; x++) {
            for (int y = 1; y < height; y++) {
                integralColKernel((y * width) + x, width, greyFloats, integral, integralSq);
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width; x++) {
                integralRowKernel((y * width) + x, integral, integralSq);
            }
        }
        return System.currentTimeMillis() - start;
    }

}
