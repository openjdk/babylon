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
import hat.buffer.CompleteBuffer;
import hat.ifacemapper.SegmentMapper;
import violajones.XMLHaarCascadeModel;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;

import static java.lang.foreign.MemoryLayout.sequenceLayout;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public interface Cascade extends CompleteBuffer {
    interface Feature extends CompleteBuffer{

        interface Rect extends CompleteBuffer{
            StructLayout layout = MemoryLayout.structLayout(
                    JAVA_BYTE.withName("x"),
                    JAVA_BYTE.withName("y"),
                    JAVA_BYTE.withName("width"),
                    JAVA_BYTE.withName("height"),
                    JAVA_FLOAT.withName("weight")
            ).withName("Rect");

            byte x();

            byte y();

            byte width();

            byte height();

            float weight();

            void x(byte x);

            void y(byte y);

            void width(byte width);

            void height(byte height);

            void weight(float height);
        }


        interface LinkOrValue extends CompleteBuffer {
            interface Anon {
                MemoryLayout layout = MemoryLayout.unionLayout(
                        JAVA_INT.withName("featureId"),
                        JAVA_FLOAT.withName("value")
                ).withName("Anon");

                int featureId();

                float value();

                void featureId(int featureId);

                void value(float value);
            }

            StructLayout layout = MemoryLayout.structLayout(
                    JAVA_BOOLEAN.withName("hasValue"),
                    MemoryLayout.paddingLayout(3),
                    Feature.LinkOrValue.Anon.layout.withName("anon")
            ).withName("LinkOrValue");

            boolean hasValue();

            void hasValue(boolean hasValue);

            Feature.LinkOrValue.Anon anon();
        }

        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("id"),
                JAVA_FLOAT.withName("threshold"),
                Feature.LinkOrValue.layout.withName("left"),
                Feature.LinkOrValue.layout.withName("right"),
                MemoryLayout.sequenceLayout(3, Feature.Rect.layout).withName("rect")
        ).withName(Feature.class.getSimpleName());

        int id();


        float threshold();


        void id(int id);


        void threshold(float threshold);

        Feature.LinkOrValue left();

        Feature.LinkOrValue right();

        Feature.Rect rect(long idx);
    }

    interface Stage extends CompleteBuffer{
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("id"),
                JAVA_FLOAT.withName("threshold"),
                JAVA_SHORT.withName("firstTreeId"),
                JAVA_SHORT.withName("treeCount")
        ).withName(Stage.class.getSimpleName());

        float threshold();

        short firstTreeId();

        short treeCount();

        int id();

        void id(int id);

        void threshold(float threshold);

        void firstTreeId(short firstTreeId);

        void treeCount(short treeCount);
    }

    interface Tree extends CompleteBuffer{
        StructLayout layout = MemoryLayout.structLayout(
                JAVA_INT.withName("id"),
                JAVA_SHORT.withName("firstFeatureId"),
                JAVA_SHORT.withName("featureCount")
        ).withName(Tree.class.getSimpleName());

        void id(int id);

        void firstFeatureId(short firstFeatureId);


        void featureCount(short featureCount);

        int id();


        short firstFeatureId();

        short featureCount();
    }

    static Cascade create(Accelerator accelerator, XMLHaarCascadeModel haarCascade) {

        Cascade cascade = SegmentMapper.of(accelerator.lookup, Cascade.class,
                JAVA_INT.withName("width"),
                JAVA_INT.withName("height"),
                JAVA_INT.withName("featureCount"),
                sequenceLayout(haarCascade.features.size(), Feature.layout.withName(Feature.class.getSimpleName())).withName("feature"),
                JAVA_INT.withName("stageCount"),
                sequenceLayout(haarCascade.stages.size(), Stage.layout.withName(Stage.class.getSimpleName())).withName("stage"),
                JAVA_INT.withName("treeCount"),
                sequenceLayout(haarCascade.trees.size(), Tree.layout.withName(Tree.class.getSimpleName())).withName("tree")
        ).allocate(accelerator.backend.arena());
        cascade.width(haarCascade.width());
        cascade.height(haarCascade.height());
        cascade.featureCount(haarCascade.features.size());
        cascade.stageCount(haarCascade.stages.size());
        cascade.treeCount(haarCascade.trees.size());
        for (int idx = 0; idx < haarCascade.features.size(); idx++) {
            Cascade.Feature cascadeFeature = cascade.feature(idx);
            var haarfeature = haarCascade.features.get(idx);
            cascadeFeature.id(haarfeature.id());
            cascadeFeature.threshold(haarfeature.threshold());
            Cascade.Feature.LinkOrValue cascadeLeft = cascadeFeature.left();
            cascadeLeft.hasValue(haarfeature.left.hasValue());
            if (haarfeature.left.hasValue()) {
                cascadeLeft.anon().value(haarfeature.left.value());
            } else {
                cascadeLeft.anon().value(haarfeature.left.featureId());
            }
            Cascade.Feature.LinkOrValue cascadeRight = cascadeFeature.right();
            cascadeRight.hasValue(haarfeature.right.hasValue());
            if (haarfeature.right.hasValue()) {
                cascadeRight.anon().value(haarfeature.right.value());
            } else {
                cascadeRight.anon().featureId(haarfeature.right.featureId());
            }
            for (int r = 0; r < 3; r++) {
                var haarrect = haarfeature.rects[r];
                if (haarrect != null) {
                    Cascade.Feature.Rect cascadeRect = cascadeFeature.rect(r);
                    cascadeRect.x(haarrect.x());
                    cascadeRect.y(haarrect.y());
                    cascadeRect.width(haarrect.width());
                    cascadeRect.height(haarrect.height());
                    cascadeRect.weight(haarrect.weight());
                }
            }
        }


        for (XMLHaarCascadeModel.Stage haarstage : haarCascade.stages) {
            Cascade.Stage cascadeStage = cascade.stage(haarstage.id);
            cascadeStage.id(haarstage.id());
            cascadeStage.threshold(haarstage.threshold());
            cascadeStage.firstTreeId(haarstage.firstTreeId());
            cascadeStage.treeCount(haarstage.treeCount());
        }

        for (XMLHaarCascadeModel.Tree haarTree : haarCascade.trees) {
            Cascade.Tree cascadeTree = cascade.tree(haarTree.id());
            cascadeTree.id(haarTree.id());
            cascadeTree.firstFeatureId(haarTree.firstFeatureId());
            cascadeTree.featureCount(haarTree.featureCount());
        }
        return cascade;
    }

    Feature feature(long idx);

    int featureCount();

    void featureCount(int featureCount);

    Stage stage(long idx);

    int stageCount();

    void stageCount(int stageCount);

    Tree tree(long idx);

    int treeCount();

    void treeCount(int treeCount);

    int width();

    void width(int width);

    int height();

    void height(int height);


}
