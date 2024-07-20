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
package violajones.ifaces;


import hat.Accelerator;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public interface ScaleTable extends Buffer {
    interface Scale extends Struct {

        float scaleValue();

        int accumGridSizeMax();

        float invArea();

        int scaledFeatureWidth();

        int scaledFeatureHeight();

        int accumGridSizeMin();

        int gridWidth();

        int gridHeight();

        float scaledXInc();

        float scaledYInc();

        int gridSize();

        void scaleValue(float scaleValue);

        void scaledFeatureWidth(int scaledFeatureWidth);

        void scaledFeatureHeight(int scaledFeatureHeight);

        void scaledXInc(float scaledXInc);

        void scaledYInc(float scaledYInc);

        void gridWidth(int gridWidth);

        void gridHeight(int gridHeight);

        void gridSize(int gridSize);

        void invArea(float invArea);

        void accumGridSizeMin(int accumGridSizeMin);

        void accumGridSizeMax(int accumGridSizeMax);
    }

     class Constraints{
        final float startScale;
        final float scaleMultiplier;
        final float increment;
        final int cascadeWidth; int cascadeHeight;  int imageWidth; int imageHeight;
        final float maxScale;
        public final int scales;
        Constraints(float startScale, float scaleMultiplier, float increment, int cascadeWidth, int cascadeHeight, int imageWidth, int imageHeight){
            this.startScale = startScale;
            this.scaleMultiplier = scaleMultiplier;
            this.increment = increment;
            this.cascadeWidth = cascadeWidth;
            this.cascadeHeight = cascadeHeight;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.maxScale  = (Math.min(
                    (float) imageWidth / cascadeWidth,
                    (float) imageHeight / cascadeHeight));
            int nonFinalScales = 0;
            for (float scale = this.startScale; scale < this.maxScale; scale *= scaleMultiplier) {
                nonFinalScales++;
            }
            this.scales = nonFinalScales;
        }
        Constraints( int cascadeWidth, int cascadeHeight, int imageWidth, int imageHeight){
           this(1f,2f,0.06f,cascadeWidth,cascadeHeight,imageWidth,imageHeight);
        }
        public Constraints(Cascade cascade, int imageWidth, int imageHeight){
            this(1f,2f,0.06f,cascade.width(),cascade.height(),imageWidth,imageHeight);
        }
    }

      default ScaleTable applyConstraints ( Constraints constraints) {
        int multiScaleAccumulativeRangeVar = 0;
        long idx = 0;
        for (float scaleValue = constraints.startScale; scaleValue < constraints.maxScale; scaleValue *= constraints.scaleMultiplier) {
            ScaleTable.Scale scale = scale(idx++);
            scale.accumGridSizeMin(multiScaleAccumulativeRangeVar);
            scale.scaleValue(scaleValue);
            final int scaledFeatureWidth = (int) (constraints.cascadeWidth * scaleValue);
            final int scaledFeatureHeight = (int) (constraints.cascadeHeight * scaleValue);
            scale.scaledFeatureWidth(scaledFeatureWidth);
            scale.scaledFeatureHeight(scaledFeatureHeight);

            final float scaledXInc = scaledFeatureWidth * constraints.increment;
            final float scaledYInc = scaledFeatureHeight * constraints.increment;
            scale.scaledXInc(scaledXInc);
            scale.scaledYInc(scaledYInc);

            int gridWidth = (int) ((constraints.imageWidth - scaledFeatureWidth) / scaledXInc);
            int gridHeight = (int) ((constraints.imageHeight - scaledFeatureHeight) / scaledYInc);
            scale.gridWidth(gridWidth);
            scale.gridHeight(gridHeight);

            int gridSize = gridWidth * gridHeight;
            scale.accumGridSizeMax(multiScaleAccumulativeRangeVar + gridSize);
            float invArea = (float) (1.0 / (scaledFeatureWidth * scaledFeatureHeight));
            scale.invArea(invArea);
            multiScaleAccumulativeRangeVar += gridSize;
        }
        multiScaleAccumulativeRange(multiScaleAccumulativeRangeVar);

        System.out.println("Scaled overlapping rectangles to search " + multiScaleAccumulativeRangeVar);
        return this;
       }

    int length();
    void length(int length);

    Scale scale(long idx);


    void multiScaleAccumulativeRange(int multiScaleAccumulativeRange);


    int multiScaleAccumulativeRange();

    default int rangeModGroupSize(int groupSize) {
        return ((multiScaleAccumulativeRange() / groupSize) + ((multiScaleAccumulativeRange() % groupSize) == 0 ? 0 : 1)) * groupSize;
    }
    Schema<ScaleTable> schema = Schema.of(ScaleTable.class, scaleTable->scaleTable
            .field("multiScaleAccumulativeRange")
            .arrayLen("length").array("scale", array->array
                    .fields(
                    "scaleValue",
                            "scaledXInc", "scaledYInc",
                            "invArea",
                            "scaledFeatureWidth","scaledFeatureHeight",
                            "gridWidth", "gridHeight",
                            "gridSize",
                            "accumGridSizeMin", "accumGridSizeMax"
                    )
            )
    );

    static ScaleTable create(Accelerator accelerator, int length){
        var instance = schema.allocate(accelerator,length);
        instance.length(length);
        return instance;
    }

    static ScaleTable createFrom(Accelerator accelerator, Constraints constraints){
        return create(accelerator,constraints.scales).applyConstraints(constraints);
    }

}
