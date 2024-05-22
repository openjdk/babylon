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
import hat.backend.Backend;
import org.xml.sax.SAXException;
import violajones.XMLHaarCascadeModel;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;

public class ViolaJonesRaw {

    public static void main(String[] _args) throws IOException, ParserConfigurationException, SAXException {

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.JAVA_MULTITHREADED);
        Arena arena = Arena.global();

        XMLHaarCascadeModel haarCascade = XMLHaarCascadeModel.load(ViolaJonesRaw.class.getResourceAsStream("/cascades/haarcascade_frontalface_default.xml"));

/*
        RGBImage rgbImage = RGBImage.create(accelerator,ImageIO.read(Objects.requireNonNull(ViolaJones.class.getResourceAsStream("/images/Nasa1996.jpg"))));


       // MultiScaleTable multiScaleTable = new MultiScaleTable(rgbImage.bufferedImage, haarCascade);

/*
        Cascade cascade = Cascade.create(accelerator);//accelerator.bindAndAllocate( Cascade.class);
        cascade.setImageWidth(rgbImageLayout.getWidth());
        cascade.setImageHeight(rgbImageLayout.getHeight());
        FeatureTable featureTable = LayoutSegmentProxy.bind(haarCascade.features.size(), arena, FeatureTable.class);
        MemorySegment memorySegment = featureTable.segment();
        memorySegment.fill((byte) 0);
        FeatureTable.Feature feature = featureTable.feature(0);
        for (int idx = 0; idx < featureTable.length(); idx++) {
            feature.idx(idx);
            var haarfeature = haarCascade.features.get(idx);
            feature.setId(haarfeature.getId());
            feature.setThreshold(haarfeature.getThreshold());
            FeatureTable.Feature.LinkOrValue left = feature.left();
            left.setHasValue(haarfeature.left.getHasValue());
            if (haarfeature.left.getHasValue()) {
                left.anon().setValue(haarfeature.left.getValue());
            } else {
                left.anon().setFeatureId(haarfeature.left.getFeatureId());
            }
            FeatureTable.Feature.LinkOrValue right = feature.right();
            right.setHasValue(haarfeature.right.getHasValue());
            if (haarfeature.right.getHasValue()) {
                right.anon().setValue(haarfeature.right.getValue());
            } else {
                right.anon().setFeatureId(haarfeature.right.getFeatureId());
            }
            FeatureTable.Feature.RectTable rectTable = feature.rects();
            for (int r = 0; r < 3; r++) {
                FeatureTable.Feature.RectTable.Rect rect = rectTable.rect(r);
                if (r < haarfeature.rects.size()) {
                    var haarrect = haarfeature.rects.get(r);
                    rect.setX(haarrect.getX());
                    rect.setY(haarrect.getY());
                    rect.setWidth(haarrect.getWidth());
                    rect.setHeight(haarrect.getHeight());
                    rect.setWeight(haarrect.getWeight());
                }
            }
        }

        StageTable stageTable = LayoutSegmentProxy.bind(haarCascade.stages.size(), arena, StageTable.class);
        for (HaarCascade.Stage haarstage : haarCascade.stages) {
            StageTable.Stage stage = stageTable.stage(haarstage.id);
            stage.setId(haarstage.id);
            stage.setThreshold(haarstage.threshold);
            stage.setFirstTreeId(haarstage.firstTreeId);
            stage.setTreeCount(haarstage.treeCount);
        }


        TreeTable treeTable = LayoutSegmentProxy.bind(haarCascade.trees.size(), arena, TreeTable.class);
        for (HaarCascade.Stage.Tree haarTree : haarCascade.trees) {
            TreeTable.Tree tree = treeTable.tree(haarTree.getId());
            tree.setId(haarTree.getId());
            tree.setFirstFeatureId(haarTree.firstFeatureId);
            tree.setFeatureCount(haarTree.featureCount);
        }
        final int maxResults = 1000;
        ResultTable resultTable = LayoutSegmentProxy.bind(maxResults, arena, ResultTable.class);
        resultTable.segment().fill((byte) 0);

        ScaleTable scaleTable = LayoutSegmentProxy.bind(multiScaleTable.multiScaleCount, arena, ScaleTable.class);


        C99CodeBuilder c99 = (C99CodeBuilder) accelerator.getCodeBuilder();
        c99
                .typedef(FeatureTable.Feature.RectTable.Rect.class)

                .typedef(FeatureTable.Feature.LinkOrValue.class)
                .typedef(FeatureTable.Feature.class)
                .typedef(ScaleTable.Scale.class)
                .typedef(StageTable.Stage.class)
                .typedef(TreeTable.Tree.class)
                .typedef(ResultTable.Result.class)
                .typedef(Cascade.class)

                .function(int.class, "b2i")
                .scalar("v", JAVA_INT)
                .body("""
                   return v < 0 ? 256 + v : v;
                   """)

                .function(int.class, "rgbToGrey")
                .scalar("r", JAVA_INT)
                .scalar("g", JAVA_INT)
                .scalar("b", JAVA_INT)
                .body("""
                   return (29 * b2i(r) + 60 * b2i(g) + 11 * b2i(b)) / 100;
                   """)

                .function(void.class, "integralColById")
                .scalar("id", JAVA_INT)
                .ptr("cascadeContext", Cascade.layout)
                .ptr("rgb", JAVA_BYTE)
                .ptr("integral", JAVA_FLOAT)
                .ptr("integralSq", JAVA_FLOAT)
                .body("""
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
                                                    
                                }""")

                .kernel("integralColKernel")
                .ptr("cascadeContext", Cascade.layout)
                .ptr("rgb", JAVA_BYTE)
                .ptr("integral", JAVA_FLOAT)
                .ptr("integralSq", JAVA_FLOAT).body("""              
                                    integralColById(ndrange.id.x, cascadeContext, rgb,  integral, integralSq);
                                """)

                .function(void.class, "integralRowById")
                .scalar("id", JAVA_INT)
                .ptr("cascadeContext", Cascade.layout)
                .ptr("integral", JAVA_FLOAT)
                .ptr("integralSq", JAVA_FLOAT).body("""
                    for (s32_t x = 1; x < cascadeContext->imageWidth; x++) {
                        s32_t monoOffset = (id * cascadeContext->imageWidth) + x;
                        integral[monoOffset] = integral[monoOffset] + integral[monoOffset - 1];
                    }
                    for (s32_t x = 1; x < cascadeContext->imageWidth; x++) {
                        s32_t monoOffset = (id * cascadeContext->imageWidth) + x;
                        integralSq[monoOffset] = integralSq[monoOffset] + integralSq[monoOffset - 1];
                    }
                """)

                .kernel("integralRowKernel")
                .ptr("cascadeContext", Cascade.layout)
                .ptr("integral", JAVA_FLOAT)
                .ptr("integralSq", JAVA_FLOAT).body("""
                    integralRowById(ndrange.id.x,  cascadeContext, integral, integralSq);
                """)

                .kernel("floatToShortKernel").ptr("cascadeContext", Cascade.layout)
                .ptr("fromIntegral", JAVA_FLOAT)
                .ptr("toIntegral", JAVA_SHORT)
                .ptr("fromIntegralSq", JAVA_FLOAT)
                .ptr("toIntegralSq", JAVA_SHORT).body("""
                                
                     toIntegral[ndrange.id.x] = (s16_t)(fromIntegral[ndrange.id.x]*(65536/fromIntegral[ndrange.id.maxX-1]));
                     toIntegralSq[ndrange.id.x] = (s16_t)(fromIntegralSq[ndrange.id.x]*(65536/fromIntegralSq[ndrange.id.maxX-1]));
                                
                """)

                .function(float.class, "gradient")
                .ptr("image", JAVA_FLOAT)
                .scalar("imageWidth", JAVA_INT)
                .scalar("x", JAVA_INT)
                .scalar("y", JAVA_INT)
                .scalar("width", JAVA_INT)
                .scalar("height", JAVA_INT)
                .body("""
                               
                /
                      A +-------+ B
                        |       |       D-B-C+A
                      C +-------+ D
                /
                    f32_t A = image[(y * imageWidth) + x];
                    f32_t D = image[((y + height) * imageWidth) + x + width];
                    f32_t C = image[((y + height) * imageWidth) + x];
                    f32_t B = image[(y * imageWidth) + x + width];
                    return D-B-C+A;
                """)

                .function(boolean.class, "isAFaceStage")
                .ptr("cascadeContext", Cascade.layout)
                .ptr("scale", ScaleTable.Scale.layout)
                .scalar("x", JAVA_INT)
                .scalar("y", JAVA_INT)
                .scalar("vnorm", JAVA_FLOAT)
                .ptr("integral", JAVA_FLOAT)
                .ptr("stagePtr", StageTable.Stage.layout)
                .ptr("treeTable", TreeTable.Tree.layout)
                .ptr("featureTable", FeatureTable.Feature.layout)
                .body("""
                                
                            
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
                                featureGradientSum +=
                                    gradient(integral,  cascadeContext->imageWidth,
                                        x + (int) (rect->x * scale->scaleValue),
                                        y + (int) (rect->y * scale->scaleValue),
                                        (int) (rect->width * scale->scaleValue),
                                        (int) (rect->height * scale->scaleValue)
                                    )
                                    * featurePtr->rects[i].weight;
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
                """)

                .kernel("singlePassCascadeKernel")
                .ptr("cascadeContext", Cascade.layout)
                .ptr("integral", JAVA_FLOAT)
                .ptr("integralSq", JAVA_FLOAT)
                .ptr("scaleTable", ScaleTable.Scale.layout)
                .ptr("resultTable", ResultTable.Result.layout)
                .ptr("stageTable", StageTable.Stage.layout)
                .ptr("treeTable", TreeTable.Tree.layout)
                .ptr("featureTable", FeatureTable.Feature.layout).body("""  
                   size_t gid = ndrange.id.x;
                   if (gid < cascadeContext->multiScaleAccumulativeRange){
                     s32_t i;
                     // This is where we select the scale to use. 
                     for ( i=0;gid >=scaleTable[i].accumGridSizeMax; i++);
                     
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
                        s32_t index =
                        #ifdef NDRANGE_CUDA
                         atomicAdd
                        #else
                         atom_add
                        #endif
                        (&cascadeContext->atomicResultTableCount, 1);
                        if (index<cascadeContext->maxResults){
                           resultTable[index].x = x;
                           resultTable[index].y = y;
                           resultTable[index].width = scale->scaledFeatureWidth;
                           resultTable[index].height = scale->scaledFeatureHeight;
                        }
                      }
                   }
                                
                                
                """);



        long progHandle = accelerator.bridge.compileProgram(c99.toString());
        if (accelerator.bridge.programOK(progHandle)) {
            long floatToShortKernel = accelerator.bridge.getKernel(progHandle, "floatToShortKernel");
            long integralColKernel = accelerator.bridge.getKernel(progHandle, "integralColKernel");
            long integralRowKernel = accelerator.bridge.getKernel(progHandle, "integralRowKernel");
            long singlePassCascadeKernel = accelerator.bridge.getKernel(progHandle, "singlePassCascadeKernel");
            ScaleTable.Scale scale = scaleTable.scale(0);
            int scalec = 0;
            for (MultiScaleTable.Scale s : multiScaleTable.table) {
                scale.idx(scalec++);
                scale.setScaleValue(s.getScaleValue());
                scale.setAccumGridSizeMax(s.getAccumGridSizeMax());
                scale.setAccumGridSizeMin(s.getAccumGridSizeMin());
                scale.setGridSize(s.getGridSize());
                scale.setGridWidth(s.getGridWidth());
                scale.setGridHeight(s.getGridHeight());
                scale.setInvArea(s.getInvArea());
                scale.setScaledFeatureWidth(s.getScaledFeatureWidth());
                scale.setScaledFeatureHeight(s.getScaledFeatureHeight());
                scale.setScaledXInc(s.getScaledXInc());
                scale.setScaledYInc(s.getScaledYInc());
            }


            cascade.setMaxGid(multiScaleTable.multiScaleAccumulativeRange);
            cascade.setAtomicResultTableCount(0);
            cascade.setMaxResults(maxResults);
            cascade.setImageSize(rgbImageLayout.getWidth(), rgbImageLayout.getHeight());
            cascade.setMultiScaleCount(multiScaleTable.multiScaleCount);
            cascade.setMultiScaleAccumulativeRange(multiScaleTable.multiScaleAccumulativeRange);
            cascade.setStageCount(haarCascade.stages.size());

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


            if (mode.equals("javaCascade")) {
                WorkStealer.of(1)
                        .forEachInRange(multiScaleTable.multiScaleAccumulativeRange, gid -> {
                            ReferenceJavaViolaJones.findFeatures(
                                    gid,
                                    cascade,
                                    harViz,
                                    haarCascade,
                                    integralImageMemorySegment,
                                    integralSqImageMemorySegment,
                                    scaleTable,
                                    resultTable,
                                    stageTable,
                                    treeTable,
                                    featureTable);
                        });
            } else if (mode.equals("javaSegments")) {

                WorkStealer.of(1)
                        .forEachInRange(multiScaleTable.multiScaleAccumulativeRange, gid -> {
                            ReferenceJavaViolaJones.findFeatures(
                                    gid,
                                    cascade,
                                    harViz,
                                    null,
                                    integralImageMemorySegment,
                                    integralSqImageMemorySegment,
                                    scaleTable,
                                    resultTable,
                                    stageTable,
                                    treeTable,
                                    featureTable);
                        });
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
