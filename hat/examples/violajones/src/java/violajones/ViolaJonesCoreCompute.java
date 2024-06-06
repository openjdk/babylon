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
package violajones;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.S32Array;
import hat.buffer.F32Array2D;
import org.xml.sax.SAXException;
import violajones.attic.ViolaJones;
import violajones.attic.ViolaJonesRaw;
import violajones.buffers.RgbS08x3Image;
import violajones.ifaces.Cascade;
import violajones.ifaces.ResultTable;
import violajones.ifaces.ScaleTable;

import javax.imageio.ImageIO;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

public class ViolaJonesCoreCompute {

    @CodeReflection
    public static int b2i(int v) {
        return v < 0 ? 256 + v : v;
    }

    @CodeReflection
    public static int grey(int r, int g, int b) {
        return (29 * b2i(r) + 60 * b2i(g) + 11 * b2i(b)) / 100;
    }

    @CodeReflection
    public static void rgbToGrey(int id, RgbS08x3Image rgbImage, F32Array2D greyImage) {
        byte r = rgbImage.data(id * 3 + 0);
        byte g = rgbImage.data(id * 3 + 1);
        byte b = rgbImage.data(id * 3 + 2);
        greyImage.array(id, grey(r, g, b));
    }

    /*
     * A pure java implementation so no @CodeReflection
     */
    static long javaRgbToGreyScale(RgbS08x3Image rgb, F32Array2D grey) {
        long start = System.currentTimeMillis();
        int size = grey.width() * grey.height();

        for (int i = 0; i < size; i++) {
            rgbToGrey(i, rgb, grey);
        }

        return System.currentTimeMillis() - start;
    }

    @CodeReflection
    public static void rgbToGreyKernel(KernelContext kc, RgbS08x3Image rgbImage, F32Array2D greyImage) {
        if (kc.x < kc.maxX){
           rgbToGrey(kc.x, rgbImage, greyImage);
        }
    }

    @CodeReflection
    public static void integralCol(int id, int width, F32Array2D greyImage, F32Array2D integral, F32Array2D integralSq) {
        float greyValue = greyImage.array(id);
        float greyValueSq = greyValue * greyValue;
        integralSq.array(id, greyValueSq + integralSq.array(id - width));
        integral.array(id, greyValue + integral.array(id - width));
    }

    @CodeReflection
    public static void integralColKernel(KernelContext kc, F32Array2D greyImage, F32Array2D integral, F32Array2D integralSq) {
        if (kc.x <kc.maxX){  // kc.maxX = imageWidth
           int x = kc.x;
           int width = kc.maxX;
           int height = greyImage.height();
           for (int y = 1; y < height; y++) {
               int id =(y * width) + x;
               integralCol(id, width, greyImage, integral, integralSq);
           }
        }
    }

    public static void javaIntegralCol(F32Array2D greyImage, F32Array2D integral, F32Array2D integralSq) {
        int height = greyImage.height();
        int width = greyImage.width();
        for (int x = 0; x<width; x++){
           for (int y = 1; y < height; y++) {
               int id =(y * width) + x;
               integralCol(id, width, greyImage, integral, integralSq);
           }
        }
    }

    @CodeReflection
    public static void integralRow(int id, F32Array2D integral, F32Array2D integralSq) {
        integral.array(id, integral.array(id) + integral.array(id - 1));
        integralSq.array(id, integralSq.array(id) + integralSq.array(id - 1));
    }

    @CodeReflection
    public static void integralRowKernel(KernelContext kc, F32Array2D integral, F32Array2D integralSq) {
        if (kc.x <kc.maxX){  // kc.maxX == imageHeight
           int y = kc.x;
           int width = integral.width();
           for (int x = 1; x < width; x++) {
               int id =(y * width) + x;
               integralRow(id, integral, integralSq);
           }
        }
    }
    public static void javaIntegralRow(F32Array2D integral, F32Array2D integralSq) {
        int height = integral.height();
        int width = integral.width();
        for (int y = 0; y<height; y++){
           for (int x = 1; x < width; x++) {
               int id =(y * width) + x;
               integralRow(id, integral, integralSq);
           }
        }
    }

    /*
     * A pure java implementation so no @CodeReflection
     */
    public static long javaCreateIntegralImage(F32Array2D greyFloats, F32Array2D integral, F32Array2D integralSq) {
        long start = System.currentTimeMillis();
        int width = greyFloats.width();
        int height = greyFloats.height();

        // The col pass creates both the integral and integralSq cols and populate the 'square'
        for (int x = 0; x < width; x++) {
            for (int y = 1; y < height; y++) {
                integralCol((y * width) + x, width, greyFloats, integral, integralSq);
            }
        }

        for (int y = 0; y < height; y++) {
            for (int x = 1; x < width; x++) {
                integralRow((y * width) + x, integral, integralSq);
            }
        }
        return System.currentTimeMillis() - start;
    }

    @CodeReflection
    static long xyToLong(int imageWidth, int x, int y) {
        return (long) y * imageWidth + x;
    }

    @CodeReflection
    static float gradient(F32Array2D integralOrIntegralSqImage, int x, int y, int w, int h) {
        int imageWidth = integralOrIntegralSqImage.width();
        float A = integralOrIntegralSqImage.array(xyToLong(imageWidth, x, y));
        float D = integralOrIntegralSqImage.array(xyToLong(imageWidth, x + w, y + h));   //  [A]-------[B]
        float C = integralOrIntegralSqImage.array(xyToLong(imageWidth, x, y + h));       //   |         |
        float B = integralOrIntegralSqImage.array(xyToLong(imageWidth, x + w, y));       //  [C]-------[D]
        return D - B - C + A;
    }


    @CodeReflection
    static boolean isAFaceStage(
            long gid,
            float scale,
            float invArea,
            int x,
            int y,
            float vnorm,
            F32Array2D integral,
            Cascade.Stage stage,
            Cascade cascade
	    ) {
        float sumOfThisStage = 0;
        int startTreeIdx = stage.firstTreeId();
        int endTreeIdx = startTreeIdx + stage.treeCount();
        for (int treeIdx = startTreeIdx; treeIdx < endTreeIdx; treeIdx++) {
            // Todo: Find a way to iterate which is interface mapped segment friendly.
            Cascade.Tree tree = cascade.tree(treeIdx);
            Cascade.Feature feature = cascade.feature(tree.firstFeatureId());

            while (feature != null) {
                float featureGradientSum = .0f;
                // features have 1, 2 or 3 rects to scan  we might be best to unroll
                // but we made sure that x,y,w,h and weight were all 0 for 'unused' rects.
                // so this is theoretically less wave divergent...
                for (int r = 0; r < 3; r++) {
                    Cascade.Feature.Rect rect = feature.rect(r);
                    if (rect != null) {
                        featureGradientSum += gradient(integral,
                                x + (int) (rect.x() * scale), //x
                                y + (int) (rect.y() * scale),   //y
                                (int) (rect.width() * scale),   //w
                                (int) (rect.height() * scale)   //h
                        ) * rect.weight();
                    }// weight is 0 for unused so should not impact featureGradientSum
                }

                // Now either navigate the tree (left or right) or update the sumOfThisStage
                // with left or right value based on comparison with features Threshold
                float featureThreshold = feature.threshold();
                boolean isLeft = ((featureGradientSum * invArea) < (featureThreshold * vnorm));
                Cascade.Feature.LinkOrValue leftOrRight = isLeft ? feature.left() : feature.right();
                Cascade.Feature.LinkOrValue.Anon anon = leftOrRight.anon();
                if (leftOrRight.hasValue()) {
                    sumOfThisStage += anon.value(); // leftOrRight.anon().value() breaks C99 codegen
                    feature = null; // loop ends
                } else {
                    feature = cascade.feature(tree.firstFeatureId() + anon.featureId());
                }
            }
        }
        return sumOfThisStage > stage.threshold(); // true if this looks like a face
    }

    @CodeReflection
    public static void findFeaturesKernel(KernelContext kc,
                                          Cascade cascade,
                                          F32Array2D integral,
                                          F32Array2D integralSq,
                                          ScaleTable scaleTable,
                                          ResultTable resultTable

    ) {

        if (kc.x < kc.maxX){//;scaleTable.multiScaleAccumulativeRange()) {
            // We need to determine the scale information for a given gid.
            // we check each scale in the scale table and check if our gid is
            // covered by the scale.
            int scalc = 0;
            ScaleTable.Scale scale = scaleTable.scale(scalc);
	    scalc++;
            while (kc.x >= scale.accumGridSizeMax() && scalc<scaleTable.length()) {
                scale = scaleTable.scale(scalc);
                scalc++;
            }

            // Now we need to convert our scale relative git to an x,y,w,h
            int scaleGid = kc.x - scale.accumGridSizeMin();

            int x = (int) ((scaleGid % scale.gridWidth()) * scale.scaledXInc());
            int y = (int) ((scaleGid / scale.gridWidth()) * scale.scaledYInc());
            int w = scale.scaledFeatureWidth();
            int h = scale.scaledFeatureHeight();

            // We now have  a unique x,y,w,h and scale value which we use to walk the cascade


            float integralGradient = gradient(integral, x, y, w, h) * scale.invArea();
            float integralSqGradient = gradient(integralSq, x, y, w, h) * scale.invArea();
            float vnorm = integralSqGradient - integralGradient * integralGradient;
            vnorm = (vnorm > 1) ? (float) Math.sqrt(vnorm) : 1;

            // If we had converted greyScale of original image into an edge highlighted image
            // (via soebel filter) it is possible to determine that the area x,y,x+w,x+h is
            // uninteresting based on the gradient values and vnorm. This is called 'Canny' pruning.

            boolean stillLooksLikeAFace = true;

            // Walk the stage list whilst each stage still resembles a face.

            int stageCount = cascade.stageCount();
            for (int stagec = 0; stagec < stageCount && stillLooksLikeAFace; stagec++) {
                Cascade.Stage stage = cascade.stage(stagec);
                stillLooksLikeAFace = isAFaceStage(kc.x, scale.scaleValue(), scale.invArea(), x, y, vnorm, integral, stage, cascade);
            }

            if (stillLooksLikeAFace) {
               int index = resultTable.atomicResultTableCountInc();
               if (index < resultTable.length()) {
                   ResultTable.Result result = resultTable.result(index);
                   result.x(x);
                   result.y(y);
                   result.width(w);
                   result.height(h);
               }
           }
        }
    }

    @CodeReflection
    static public void compute(final ComputeContext cc, Cascade cascade, BufferedImage bufferedImage, RgbS08x3Image rgbS08x3Image, ResultTable resultTable) {
        long start = System.currentTimeMillis();
        int width = rgbS08x3Image.width();




        int height = rgbS08x3Image.height();
        Accelerator accelerator = cc.accelerator;
        F32Array2D greyImage = F32Array2D.create(accelerator, width, height);
        //javaRgbToGreyScale(rgbS08x3Image, greyImage);

        cc.dispatchKernel(width * height, kc -> rgbToGreyKernel(kc, rgbS08x3Image, greyImage));
        F32Array2D integralImage = F32Array2D.create(accelerator, width, height);
        F32Array2D integralSqImage = F32Array2D.create(accelerator, width, height);

        //javaCreateIntegralImage(greyImage, integralImage, integralSqImage);

        //javaIntegralCol(greyImage, integralImage, integralSqImage);
        //javaIntegralRow(integralImage, integralSqImage);
        cc.dispatchKernel(width, kc -> integralColKernel(kc, greyImage, integralImage, integralSqImage));
        cc.dispatchKernel(height, kc -> integralRowKernel(kc, integralImage, integralSqImage));
        // harViz.showIntegrals();
        ScaleTable scaleTable = ScaleTable.create(accelerator, cascade, width, height);
        System.out.print("range requested=");
        System.out.print(scaleTable.multiScaleAccumulativeRange());
        System.out.println();

        cc.dispatchKernel(scaleTable.multiScaleAccumulativeRange(), kc ->
                findFeaturesKernel(kc, cascade, integralImage, integralSqImage, scaleTable, resultTable));
        long end = System.currentTimeMillis();
        System.out.print(end - start);
        System.out.println("ms");
        // harViz.showResults(resultTable, null, null);
    }


}
