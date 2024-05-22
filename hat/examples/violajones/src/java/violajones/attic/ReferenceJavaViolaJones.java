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
import violajones.XMLHaarCascadeModel;
import violajones.attic.CoreJavaViolaJones;
import violajones.ifaces.Cascade;
import violajones.ifaces.ResultTable;
import violajones.ifaces.ScaleTable;


public class ReferenceJavaViolaJones extends CoreJavaViolaJones {
    static boolean isAFaceStage(
            float scale, float invArea,
            int x,
            int y,
            float vnorm,
            F32Array2D integral,
            Cascade.Stage stage,
           Cascade cascade) {
        float sumOfThisStage = 0;
        int startTreeIdx = stage.firstTreeId();
        int endTreeIdx = startTreeIdx + stage.treeCount();
        for (int treeIdx = startTreeIdx; treeIdx < endTreeIdx; treeIdx++) {
            Cascade.Tree tree = cascade.tree(treeIdx);
            Cascade.Feature feature = cascade.feature(tree.firstFeatureId());
            while (feature != null) {
                float featureGradientSum = .0f;

                for (int r = 0; r < 3; r++) {
                    Cascade.Feature.Rect rect = feature.rect(r);

                    float g = gradient(integral,
                            x + (int) (rect.x() * scale), y + (int) (rect.y() * scale),
                            (int) (rect.width() * scale), (int) (rect.height() * scale)) *
                            rect.weight();  // we assume weight is 0 for unused ;)
                    featureGradientSum += g;
                }
                if ((featureGradientSum * invArea) < (feature.threshold() * vnorm)) {//left
                    var left = feature.left();
                    if (left.hasValue()) {
                        sumOfThisStage += left.anon().value();
                        feature = null;
                    } else {
                        feature =cascade.feature(tree.firstFeatureId() + left.anon().featureId());

                    }
                } else {//right
                    var right = feature.right();
                    if (right.hasValue()) {
                        sumOfThisStage += right.anon().value();
                        feature = null;
                    } else {
                        feature =cascade.feature(tree.firstFeatureId() + right.anon().featureId());
                    }
                }
            }
        }
        return sumOfThisStage > stage.threshold();
    }

    static boolean isAFaceStageHaar(

            float scale, float invArea,
            int x,
            int y,
            float vnorm,
            F32Array2D integral,
            XMLHaarCascadeModel.Stage stage,
            XMLHaarCascadeModel haarCascade
    ) {
        float sumOfThisStage = 0;

            int startTreeIdx = stage.firstTreeId();
            int endTreeIdx = startTreeIdx + stage.treeCount();
            for (int treeIdx = startTreeIdx; treeIdx < endTreeIdx; treeIdx++) {
                XMLHaarCascadeModel.Tree tree= haarCascade.trees.get(treeIdx);
              //  Cascade.Tree tree = cascade.tree(treeIdx);


            XMLHaarCascadeModel.Feature feature = haarCascade.features.get(tree.firstFeatureId);
            while (feature != null) {
                float featureGradientSum = .0f;
                for (XMLHaarCascadeModel.Feature.Rect r : feature.rects) {
                    if (r != null) {
                        float g = gradient(integral,
                                x + (int) (r.x() * scale), y + (int) (r.y() * scale),
                                (int) (r.width() * scale), (int) (r.height() * scale)) *
                                r.weight();
                        featureGradientSum += g;
                    }
                }
                if ((featureGradientSum * invArea) < (feature.threshold() * vnorm)) {//left
                    if (feature.left.hasValue()) {
                        sumOfThisStage += feature.left.value();
                        feature = null;
                    } else {
                        feature = haarCascade.features.get(tree.firstFeatureId()+feature.left.featureId());
                    }
                } else {//right
                    if (feature.right.hasValue()) {
                        sumOfThisStage += feature.right.value();
                        feature = null;
                    } else {
                        feature = haarCascade.features.get(tree.firstFeatureId()+feature.right.featureId());
                      //  feature = tree.features.get(feature.right.featureId());
                    }
                }
            }

        }
        return sumOfThisStage > stage.threshold;
    }

    static void findFeatures(int gid,
                             Cascade cascade,
                             F32Array2D integral,
                             F32Array2D integralSq,
                             ScaleTable scaleTable,
                             ResultTable resultTable

    ) {
        if (gid < scaleTable.multiScaleAccumulativeRange()) {

            int scalc = 0;
            ScaleTable.Scale scale = scaleTable.scale(scalc++);
            for (;
                 gid >= scale.accumGridSizeMax();
                 scale=scaleTable.scale(scalc++))
                ;
            int scaleGid = gid - scale.accumGridSizeMin();
            int x = (int) ((scaleGid % scale.gridWidth()) * scale.scaledXInc());
            int y = (int) ((scaleGid / scale.gridWidth()) * scale.scaledYInc());
            float integralGradient = gradient(integral,  x, y, scale.scaledFeatureWidth(), scale.scaledFeatureHeight()) * scale.invArea();
            float integralSqGradient = gradient(integralSq,  x, y, scale.scaledFeatureWidth(), scale.scaledFeatureHeight()) * scale.invArea();

            float vnorm = integralSqGradient - integralGradient * integralGradient;
            vnorm = (vnorm > 1) ? (float) Math.sqrt(vnorm) : 1;
            boolean stillLooksLikeAFace = true;
            if (cascade instanceof XMLHaarCascadeModel haarCascade) {
                for (XMLHaarCascadeModel.Stage stage : haarCascade.stages) {
                    if (!(stillLooksLikeAFace = isAFaceStageHaar(
                            scale.scaleValue(), scale.invArea(), x, y, vnorm, integral, stage, haarCascade))) {
                        break;
                    }
                }
            } else {
                int stageCount =  cascade.stageCount();
                for (int stagec = 0; stagec < stageCount; stagec++) {
                    Cascade.Stage stage = cascade.stage(stagec);
                    if (!(stillLooksLikeAFace = isAFaceStage(
                            scale.scaleValue(), scale.invArea(), x, y, vnorm, integral, stage, cascade))){

                        break;
                    }
                }
            }
            if (stillLooksLikeAFace) {
                int index = resultTable.atomicResultTableCount();
                index++;
                resultTable.atomicResultTableCount(index);
                System.out.println("face at " + x + "," + y);
                if (index < resultTable.length()) {
                    ResultTable.Result result = resultTable.result(index);
                    result.x(x);
                    result.y(y);
                    result.width(scale.scaledFeatureWidth());
                    result.height(scale.scaledFeatureWidth());
                }
            }
        }
    }

}
