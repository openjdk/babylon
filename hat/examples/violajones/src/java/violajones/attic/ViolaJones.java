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


import hat.Accelerator;
import hat.backend.JavaMultiThreadedBackend;
import hat.backend.WorkStealer;
import hat.buffer.F32Array2D;
import org.xml.sax.SAXException;
import violajones.HaarViewer;
import violajones.XMLHaarCascadeModel;
import violajones.buffers.RgbS08x3Image;
import violajones.ifaces.Cascade;
import violajones.ifaces.ResultTable;
import violajones.ifaces.ScaleTable;

import javax.imageio.ImageIO;
import javax.xml.parsers.ParserConfigurationException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.Objects;

public class ViolaJones {

    public static void main(String[] _args) throws IOException, ParserConfigurationException, SAXException {
        //  Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend::isJava);
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), (backend -> backend instanceof JavaMultiThreadedBackend));


        BufferedImage nasa = ImageIO.read(Objects.requireNonNull(ViolaJones.class.getResourceAsStream("/images/Nasa1996.jpg")));
        XMLHaarCascadeModel xmlHaarCascade = XMLHaarCascadeModel.load(ViolaJonesRaw.class.getResourceAsStream("/cascades/haarcascade_frontalface_default.xml"));
        Cascade cascade = Cascade.create(accelerator, xmlHaarCascade);

        var rgbImage = RgbS08x3Image.create(accelerator, nasa);

        var width = nasa.getWidth();
        var height = nasa.getHeight();
        var scaleTable = ScaleTable.create(accelerator, cascade, width, height);// multiScaleTable.multiScaleCount);


        var greyImageF32 = F32Array2D.create(accelerator, width, height);
        var integralImageF32 = F32Array2D.create(accelerator, width, height);
        var integralSqImageF32 = F32Array2D.create(accelerator, width, height);
        var resultTable = ResultTable.create(accelerator, 1000);
        resultTable.atomicResultTableCount(0);

        CoreJavaViolaJones.rgbToGreyScale(rgbImage, greyImageF32);
        CoreJavaViolaJones.createIntegralImage(greyImageF32, integralImageF32, integralSqImageF32);

        HaarViewer harViz = new HaarViewer(accelerator, nasa, rgbImage, cascade, integralImageF32, integralSqImageF32);

        harViz.showIntegrals();


        //   long floatToShortKernel = accelerator.bridge.getKernel(progHandle, "floatToShortKernel");
        //   long integralColKernel = accelerator.bridge.getKernel(progHandle, "integralColKernel");
        //   long integralRowKernel = accelerator.bridge.getKernel(progHandle, "integralRowKernel");
        //   long singlePassCascadeKernel = accelerator.bridge.getKernel(progHandle, "singlePassCascadeKernel");


        // openCLBridge.dump(cascadeMemorySegment, ((OpenCLStructLayout.Tools)cascadeInterface).getLayout());
        // openCLBridge.dump(scaleTable,scaleTableLayout.layout);


        //   openCLBridge.dump(treeTable, treeTableLayout.layout);
        // openCLBridge.dump(stageTable, stageTableLayout.layout);
        //   FloatBuffer integralImage = FloatBuffer.create(accelerator, rgbImageLayout.getElementCount());
        //   FloatBuffer integralSqImage = FloatBuffer.create(accelerator, rgbImageLayout.getElementCount());

        // MemorySegment integralImageMemorySegment = arena.allocateArray(JAVA_FLOAT, rgbImageLayout.getElementCount());
        // MemorySegment integralSqImageMemorySegment = arena.allocateArray(JAVA_FLOAT, rgbImageLayout.getElementCount());

        // openCLBridge.dump(cascadeMemorySegment, cascadeLayout.layout);


        int groupSize = 256;
        int rangeModGroupSize = scaleTable.rangeModGroupSize(groupSize);
        //(scaleTable.multiScaleAccumulativeRange() / groupSize) + ((scaleTable.multiScaleAccumulativeRange() % groupSize) == 0 ? 0 : 1)) * groupSize;

/*
        OpenCLCodeBuilder c99 = (OpenCLCodeBuilder) accelerator.getCodeBuilder();
        c99
                .typedef(FeatureTable.Feature.RectTable.Rect.class)
                .typedef(FeatureTable.Feature.LinkOrValue.class)
                .typedef(FeatureTable.Feature.class)
                .typedef(ScaleTable.Scale.class)
                .typedef(StageTable.Stage.class)
                .typedef(TreeTable.Tree.class)
                .typedef(ResultTable.Result.class)
                .typedef(Cascade.class)
                .append("""

                #define SCOPE_START  ndrange_t ndrange;ndrange.id.x=get_global_id(0);ndrange.id.maxX=get_global_size(0);
                #ifdef NDRANGE_CUDA
                #define atomicInc(p) atomicAdd(p, 1)
                #else
                #define atomicInc(p) atom_add(p, 1)
                #endif

                inline int b2i(i4 v){
                   return v < 0 ? 256 + v : v;
                }
                inline int rgbToGrey(i4 r, i4 g, i4 b){
                   return (29 * b2i(r) + 60 * b2i(g) + 11 * b2i(b)) / 100;
                }
                inline void integralColById(i4 id, __global cascade_t *cascadeContext, __global b1 *rgb, __global f4 *integral, __global f4 *integralSq){
                   integralSq[id] = integral[id] = 0.0f;
                   for (s32_t y = 1; y < cascadeContext->imageHeight; y++) {
                       s32_t monoOffset = (y * cascadeContext->imageWidth) + id;
                       f32_t lastSq = integralSq[monoOffset - cascadeContext->imageWidth];
                       f32_t last = integral[monoOffset - cascadeContext->imageWidth];
                       char r = rgb[monoOffset * 3 + 0];
                       char g = rgb[monoOffset * 3 + 1];
                       char b = rgb[monoOffset * 3 + 2];
                       f32_t greyValue = rgbToGrey(r, g, b);
                       f32_t greyValueSq = greyValue * greyValue;
                       integralSq[monoOffset] = greyValueSq + lastSq;
                       integral[monoOffset] = greyValue + last;
                   }
                }
                __kernel void integralColKernel(__global cascade_t *cascadeContext, __global b1 *rgb, __global f4 *integral, __global f4 *integralSq){
                     SCOPE_START
                     integralColById(ndrange.id.x, cascadeContext, rgb,  integral, integralSq);
                }
                inline void integralRowById(i4 id, __global cascade_t *cascadeContext, __global f4 *integral, __global f4 *integralSq){
                     for (s32_t x = 1; x < cascadeContext->imageWidth; x++) {
                        s32_t monoOffset = (id * cascadeContext->imageWidth) + x;
                        integral[monoOffset] = integral[monoOffset] + integral[monoOffset - 1];
                     }
                     for (s32_t x = 1; x < cascadeContext->imageWidth; x++) {
                        s32_t monoOffset = (id * cascadeContext->imageWidth) + x;
                        integralSq[monoOffset] = integralSq[monoOffset] + integralSq[monoOffset - 1];
                     }
                }
                __kernel void integralRowKernel(__global cascade_t *cascadeContext, __global f4 *integral, __global f4 *integralSq){
                   SCOPE_START
                   integralRowById(ndrange.id.x,  cascadeContext, integral, integralSq);
                }
                __kernel void floatToShortKernel(__global cascade_t *cascadeContext, __global f4 *fromIntegral, __global s2 *toIntegral, __global f4 *fromIntegralSq, __global s2 *toIntegralSq){
                   SCOPE_START
                   toIntegral[ndrange.id.x] = (s16_t)(fromIntegral[ndrange.id.x]*(65536/fromIntegral[ndrange.id.maxX-1]));
                   toIntegralSq[ndrange.id.x] = (s16_t)(fromIntegralSq[ndrange.id.x]*(65536/fromIntegralSq[ndrange.id.maxX-1]));
                }


                /
                      A +-------+ B
                        |       |       D-B-C+A
                      C +-------+ D
                /
                inline float gradient(__global f4 *image, i4 imageWidth, i4 x, i4 y, i4 width, i4 height){
                   f32_t A = image[(y * imageWidth) + x];
                   f32_t D = image[((y + height) * imageWidth) + x + width];
                   f32_t C = image[((y + height) * imageWidth) + x];
                   f32_t B = image[(y * imageWidth) + x + width];
                   return D-B-C+A;
                }
                inline boolean isAFaceStage(__global cascade_t *cascadeContext, __global scale_t *scale, i4 x, i4 y, f4 vnorm, __global f4 *integral, __global stage_t *stagePtr, __global tree_t *treeTable, __global feature_t *featureTable){
                   f32_t sumOfThisStage = 0;
                   for (s32_t treeId = stagePtr->firstTreeId; treeId < (stagePtr->firstTreeId+stagePtr->treeCount); treeId++) {
                       // featureId from 0 to how many roots there are.... we use -1 for none! hence s32_t
                       const __global tree_t *treePtr = &treeTable[treeId];
                       s32_t featureId = treePtr->firstFeatureId;
                       while (featureId >= 0) {
                           const __global feature_t *featurePtr = &featureTable[featureId];
                           f32_t featureGradientSum = .0f;
                           for (s32_t i = 0; i < 3; i++) {
                               const __global rect_t  *rect = &featurePtr->rects[i];
                               featureGradientSum +=   featurePtr->rects[i].weight *
                                   gradient(integral,  cascadeContext->imageWidth,
                                       x + (int) (rect->x * scale->scaleValue),
                                       y + (int) (rect->y * scale->scaleValue),
                                       (int) (rect->width * scale->scaleValue),
                                       (int) (rect->height * scale->scaleValue)
                                   ) ;
                           }
                           if ((featureGradientSum * scale->invArea) < (featurePtr->threshold * vnorm)) {//left
                              if (featurePtr->left.hasValue) {
                                  sumOfThisStage += featurePtr->left.anon.value;
                                  featureId = -1;
                              } else {
                                  featureId = treePtr->firstFeatureId+featurePtr->left.anon.featureId;
                              }
                           }else{ // right
                              if (featurePtr->right.hasValue) {
                                  sumOfThisStage += featurePtr->right.anon.value;
                                  featureId = -1;
                              } else {
                                  featureId = treePtr->firstFeatureId+featurePtr->right.anon.featureId;
                              }
                           }
                       }
                   }
                   return sumOfThisStage > stagePtr->threshold;
                }
                __kernel void singlePassCascadeKernel(__global cascade_t *cascadeContext, __global f4 *integral, __global f4 *integralSq, __global scale_t *scaleTable, __global result_t *resultTable, __global stage_t *stageTable, __global tree_t *treeTable, __global feature_t *featureTable){
                   SCOPE_START

                   size_t gid = ndrange.id.x;
                   if (gid < cascadeContext->multiScaleAccumulativeRange){
                      s32_t i;
                      // This is where we select the scale to use.
                      for (i=0; gid >=scaleTable[i].accumGridSizeMax; i++)
                         ;

                      __global scale_t *scale = &scaleTable[i];

                      s16_t x = (s16_t)(((gid-scale->accumGridSizeMin) % scale->gridWidth) * scale->scaledXInc);
                      s16_t y = (s16_t)(((gid-scale->accumGridSizeMin) / scale->gridWidth) * scale->scaledYInc);

                      f32_t integralGradient = gradient(integral, cascadeContext->imageWidth, x, y, scale->scaledFeatureWidth, scale->scaledFeatureHeight) * scale->invArea;
                      f32_t integralSqGradient = gradient(integralSq, cascadeContext->imageWidth, x, y, scale->scaledFeatureWidth, scale->scaledFeatureHeight) * scale->invArea;

                      f32_t vnorm = integralSqGradient - integralGradient * integralGradient;
                      vnorm =  (vnorm > 1) ? sqrt(vnorm) : 1;

                      bool stillLooksLikeAFace = true;

                      for (s32_t stageId = 0; stillLooksLikeAFace && (stageId < cascadeContext->stageCount); stageId++) {
                         __global stage_t *stagePtr = &stageTable[stageId];
                         stillLooksLikeAFace =isAFaceStage(cascadeContext, scale,  x, y,  vnorm,  integral,  stagePtr, treeTable, featureTable);
                      }
                      if (stillLooksLikeAFace) {
                         s32_t index = atomicInc(&cascadeContext->atomicResultTableCount);
                         if (index<cascadeContext->maxResults){
                            resultTable[index].x = x;
                            resultTable[index].y = y;
                            resultTable[index].width = scale->scaledFeatureWidth;
                            resultTable[index].height = scale->scaledFeatureHeight;
                         }
                       }
                   }
                }
                """
             );



        long progHandle = accelerator.bridge.compileProgram(c99.toString());
        if (accelerator.bridge.programOK(progHandle)) {
            long floatToShortKernel = accelerator.bridge.getKernel(progHandle, "floatToShortKernel");
            long integralColKernel = accelerator.bridge.getKernel(progHandle, "integralColKernel");
            long integralRowKernel = accelerator.bridge.getKernel(progHandle, "integralRowKernel");
            long singlePassCascadeKernel = accelerator.bridge.getKernel(progHandle, "singlePassCascadeKernel");


            // openCLBridge.dump(cascadeMemorySegment, ((OpenCLStructLayout.Tools)cascadeInterface).getLayout());
            // openCLBridge.dump(scaleTable,scaleTableLayout.layout);


            //   openCLBridge.dump(treeTable, treeTableLayout.layout);
            // openCLBridge.dump(stageTable, stageTableLayout.layout);
            MemorySegment integralImageMemorySegment = arena.allocateArray(JAVA_FLOAT, rgbImageLayout.getElementCount());
            MemorySegment integralSqImageMemorySegment = arena.allocateArray(JAVA_FLOAT, rgbImageLayout.getElementCount());

            // openCLBridge.dump(cascadeMemorySegment, cascadeLayout.layout);

            int groupSize = 256;
            int range = ((multiScaleTable.multiScaleAccumulativeRange / groupSize) + ((multiScaleTable.multiScaleAccumulativeRange % groupSize) == 0 ? 0 : 1)) * groupSize;

            ImageLayout integralImage = new ImageLayout(new BufferedImage(rgbImageLayout.getWidth(), rgbImageLayout.getHeight(), BufferedImage.TYPE_USHORT_GRAY));
            ImageLayout integralSqImage = new ImageLayout(new BufferedImage(rgbImageLayout.getWidth(), rgbImageLayout.getHeight(), BufferedImage.TYPE_USHORT_GRAY));
            ImageLayout.Instance integralImageInstance = integralImage.instance(arena);
            ImageLayout.Instance integralSqImageInstance = integralSqImage.instance(arena);

            ImageLayout.Instance rgbImageLayoutInstance = rgbImageLayout.instance(arena);

            HaarVisualizer harViz = new HaarVisualizer(rgbImageLayoutInstance, haarCascade, integralImageInstance, integralSqImageInstance);

            accelerator.bridge.ndrange(integralColKernel, rgbImageLayout.getWidth(),
                    DeviceArgs.of()
                            .s08_1dRO(cascade.segment())
                            .s08_1dRO(rgbImageLayoutInstance.memorySegment)
                            .f32_1dWO(integralImageMemorySegment)
                            .f32_1dWO(integralSqImageMemorySegment)
            );

            accelerator.bridge.ndrange(integralRowKernel, rgbImageLayout.getHeight(),
                    DeviceArgs.of()
                            .s08_1dRO(cascade.segment())
                            .f32_1dRW(integralImageMemorySegment)
                            .f32_1dRW(integralSqImageMemorySegment)
            );

            // This allows us to visualize the integral or integralSq image.
            // We map the integral + integralSq floats to a grey image
            accelerator.bridge.ndrange(floatToShortKernel,
                    rgbImageLayout.getElementCount(),
                    DeviceArgs.of()
                            .s08_1dRO(cascade.segment())
                            .f32_1dRO(integralImageMemorySegment)
                            .u16_1dWO(integralImageInstance.memorySegment)
                            .f32_1dRO(integralSqImageMemorySegment)
                            .u16_1dWO(integralSqImageInstance.memorySegment)
            );
            harViz.showIntegrals();


            String mode = System.getProperty("mode", "bridge");
            System.out.println("Mode =" + mode);

            long start = System.currentTimeMillis();
*/

        if (true) {
            long start = System.currentTimeMillis();
            WorkStealer.usingAllProcessors(accelerator)
                    .forEachInRange(accelerator.range(scaleTable.multiScaleAccumulativeRange()), r -> {
                        ReferenceJavaViolaJones.findFeatures(
                                r.kid.x,
                                xmlHaarCascade,//cascade,//haarCascade, //or cascade
                                integralImageF32,
                                integralSqImageF32,
                                scaleTable,
                                resultTable);
                    });
            System.out.println("done " + (System.currentTimeMillis() - start) + "ms");
            harViz.showResults(resultTable, null, null);
        }
        //   } else if (mode.equals("javaSegments")) {

      /*  WorkStealer.of(1)
                .forEachInRange(multiScaleTable.multiScaleAccumulativeRange, gid -> {
                    ReferenceJavaViolaJones.findFeatures(
                            gid,
                            cascade,
                            harViz,
                            null,
                            integralImageInstance.memorySegment,
                            integralSqImageInstance.memorySegment,
                            scaleTable,
                            resultTable,
                            stageTable,
                            treeTable,
                            featureTable);
                }); */


                /*
            } else {
                accelerator.bridge.ndrange(singlePassCascadeKernel, range,
                        DeviceArgs.of()
                                .s08_1dRW(cascade.segment()) // RW only  for atomicResult counter
                                .f32_1dRO(integralImageMemorySegment)
                                .f32_1dRO(integralSqImageMemorySegment)
                                .s08_1dRO(scaleTable.segment())
                                .s08_1dRW(resultTable.segment())
                                .s08_1dRO(stageTable.segment())
                                .s08_1dRO(treeTable.segment())
                                .s08_1dRO(featureTable.segment())

                );
            }
            System.out.println("ms = " + (System.currentTimeMillis() - start));
            //  openCLBridge.dump(cascadeMemorySegment, cascadeLayout.layout);
            //  harViz.showResults(cascadeInstance.getAtomicResultTableCount(),cascadeInstance.getMaxResults(), resultTable, resultTableLayout);
            harViz.showResults(cascade.getAtomicResultTableCount(), cascade.getMaxResults(), resultTable);

            accelerator.bridge.releaseKernel(integralColKernel);
            accelerator.bridge.releaseKernel(integralRowKernel);
            accelerator.bridge.releaseKernel(floatToShortKernel);
            accelerator.bridge.releaseKernel(singlePassCascadeKernel);

            accelerator.bridge.releaseProgram(progHandle);
        }
        accelerator.bridge.release();
*/

    }
}
