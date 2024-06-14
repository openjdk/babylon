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
import hat.buffer.Table;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SequenceLayout;
import java.lang.foreign.StructLayout;
import java.lang.invoke.MethodHandles;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface ScaleTable extends Table<ScaleTable.Scale> {


    interface Scale extends Buffer {
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_FLOAT.withName("scaleValue"),
                JAVA_FLOAT.withName("scaledXInc"),
                JAVA_FLOAT.withName("scaledYInc"),
                JAVA_FLOAT.withName("invArea"),
                JAVA_INT.withName("scaledFeatureWidth"),
                JAVA_INT.withName("scaledFeatureHeight"),
                JAVA_INT.withName("gridWidth"),
                JAVA_INT.withName("gridHeight"),
                JAVA_INT.withName("gridSize"),
                JAVA_INT.withName("accumGridSizeMin"),
                JAVA_INT.withName("accumGridSizeMax")
        ).withName("Scale");

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

        default void copyFrom(Scale s) {
            scaleValue(s.scaleValue());
            accumGridSizeMax(s.accumGridSizeMax());
            accumGridSizeMin(s.accumGridSizeMin());
            gridSize(s.gridSize());
            gridWidth(s.gridWidth());
            gridHeight(s.gridHeight());
            invArea(s.invArea());
            scaledFeatureWidth(s.scaledFeatureWidth());
            scaledFeatureHeight(s.scaledFeatureHeight());
            scaledXInc(s.scaledXInc());
            scaledYInc(s.scaledYInc());
        }
    }
    StructLayout layout =  MemoryLayout.structLayout(
            JAVA_INT.withName("length"),
            JAVA_INT.withName("multiScaleAccumulativeRange"),
            MemoryLayout.sequenceLayout(0, ScaleTable.Scale.layout).withName("scale")
    ).withName(ScaleTable.class.getSimpleName());
    private static ScaleTable create(BufferAllocator bufferAllocator, int length) {
        return Buffer.setLength(
                bufferAllocator.allocate(SegmentMapper.ofIncomplete(MethodHandles.lookup(),ScaleTable.class,layout,length)),length);
    }

    static ScaleTable create(BufferAllocator bufferAllocator, Cascade cascade, int imageWidth, int imageHeight) {

        final float startScale = 1f;
        final float scaleMultiplier = 2f;
        final float increment = 0.06f;

        // We need to capture multi scale data
        // this is unique per image as it is
        // based on size, how many scales we want and the overlap desired

        var maxScale = (Math.min(
                (float) imageWidth / cascade.width(),
                (float) imageHeight / cascade.height()));

        //System.out.println("Image " + imageWidth + "x" + imageHeight);
        // Alas we need to do this twice. We need a count to allocate the segment size
        int multiScaleCountVar = 0;
        for (float scale = startScale; scale < maxScale; scale *= scaleMultiplier) {
            multiScaleCountVar++;
        }

        ScaleTable scaleTable = ScaleTable.create(bufferAllocator, multiScaleCountVar);

        // now we know the size

        int multiScaleAccumulativeRangeVar = 0;
        long idx = 0;
        for (float scaleValue = startScale; scaleValue < maxScale; scaleValue *= scaleMultiplier) {
            ScaleTable.Scale scale = scaleTable.scale(idx++);
            scale.accumGridSizeMin(multiScaleAccumulativeRangeVar);
            scale.scaleValue(scaleValue);
            final int scaledFeatureWidth = (int) (cascade.width() * scaleValue);
            final int scaledFeatureHeight = (int) (cascade.height() * scaleValue);
            scale.scaledFeatureWidth(scaledFeatureWidth);
            scale.scaledFeatureHeight(scaledFeatureHeight);

            final float scaledXInc = scaledFeatureWidth * increment;
            final float scaledYInc = scaledFeatureHeight * increment;
            scale.scaledXInc(scaledXInc);
            scale.scaledYInc(scaledYInc);

            int gridWidth = (int) ((imageWidth - scaledFeatureWidth) / scaledXInc);
            int gridHeight = (int) ((imageHeight - scaledFeatureHeight) / scaledYInc);
            scale.gridWidth(gridWidth);
            scale.gridHeight(gridHeight);

            int gridSize = gridWidth * gridHeight;
            scale.accumGridSizeMax(multiScaleAccumulativeRangeVar + gridSize);
            float invArea = (float) (1.0 / (scaledFeatureWidth * scaledFeatureHeight));
            scale.invArea(invArea);
            multiScaleAccumulativeRangeVar += gridSize;
        }
        scaleTable.multiScaleAccumulativeRange(multiScaleAccumulativeRangeVar);
        //System.out.println("Scales " + scaleTable.length());
        System.out.println("Scaled overlapping rectangles to search " + multiScaleAccumulativeRangeVar);
        return scaleTable;
    }

    Scale scale(long idx);

    default Scale get(int i) {
        return scale(i);
    }

    void multiScaleAccumulativeRange(int multiScaleAccumulativeRange);


    int multiScaleAccumulativeRange();

    default int rangeModGroupSize(int groupSize) {
        return ((multiScaleAccumulativeRange() / groupSize) + ((multiScaleAccumulativeRange() % groupSize) == 0 ? 0 : 1)) * groupSize;
    }

}
