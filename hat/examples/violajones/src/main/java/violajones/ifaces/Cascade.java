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
import hat.buffer.BufferAllocator;
import hat.ifacemapper.Schema;
import hat.buffer.Buffer;

import java.lang.invoke.MethodHandles;

public interface Cascade extends Buffer {
    interface Feature extends Struct {

        interface Rect extends Struct {
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


        interface LinkOrValue extends Struct {
            interface Anon  extends Union {

                int featureId();

                float value();

                void featureId(int featureId);

                void value(float value);
            }
            boolean hasValue();

            void hasValue(boolean hasValue);

            Feature.LinkOrValue.Anon anon();
        }
        int id();


        float threshold();


        void id(int id);


        void threshold(float threshold);

        Feature.LinkOrValue left();

        Feature.LinkOrValue right();

        Feature.Rect rect(long idx);
    }

    interface Stage extends Struct {
        float threshold();

        short firstTreeId();

        short treeCount();

        int id();

        void id(int id);

        void threshold(float threshold);

        void firstTreeId(short firstTreeId);

        void treeCount(short treeCount);
    }

    interface Tree extends Struct {
        void id(int id);

        void firstFeatureId(short firstFeatureId);


        void featureCount(short featureCount);

        int id();


        short firstFeatureId();

        short featureCount();
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
    Schema<Cascade> schema = Schema.of(Cascade.class, c -> c
            .fields("width","height")
            .arrayLen("featureCount").array("feature", feature -> feature
                    .fields("id","threshold")
                    .fields("left","right",linkOrValue->linkOrValue
                            .field("hasValue")
                            .pad(3)
                            .field("anon", anon->anon.fields("featureId","value"))
                    )
                    .array("rect", 3 , rect->rect.fields("x","y","width","height","weight"))
            )
            .arrayLen("stageCount").array("stage", stage->stage.fields("id","threshold","firstTreeId","treeCount"))
            .arrayLen("treeCount").array("tree",tree->tree.fields("id","firstFeatureId","featureCount"))
    );

    static Cascade create(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator, int width, int height,
    int features,int stages,int trees){
        var instance  = schema.allocate(lookup,
                bufferAllocator,
                features,
                stages,
                trees
        );
        instance.width(width);
        instance.height(height);
        instance.featureCount(features);
        instance.stageCount(stages);
        instance.treeCount(trees);
        return instance;
    }

    static Cascade create(Accelerator accelerator, int width, int height,
                          int features, int stages, int trees){
       return create(accelerator.lookup,accelerator,width,height,features,stages,trees);
    }

    static Cascade createFrom(Accelerator accelerator, Cascade cascade){
        return create(accelerator.lookup,accelerator,cascade.width(),cascade.height(),cascade.featureCount(),cascade.stageCount(),cascade.treeCount()).copyFrom(cascade);
    }

    default Cascade copyFrom(Cascade fromCascade){
        Cascade toCascade= this;
        toCascade.width(fromCascade.width());
        toCascade.height(fromCascade.height());
        toCascade.featureCount(fromCascade.featureCount());
        toCascade.stageCount(fromCascade.stageCount());
        toCascade.treeCount(fromCascade.treeCount());
        for (int idx = 0; idx < fromCascade.featureCount(); idx++) {
            Cascade.Feature toFeature =  toCascade.feature(idx);
            Cascade.Feature fromFeature = fromCascade.feature(idx);
            toFeature.id(fromFeature.id());
            toFeature.threshold(fromFeature.threshold());
            Cascade.Feature.LinkOrValue toLeftLinkOrValue = toFeature.left();
            toLeftLinkOrValue.hasValue(fromFeature.left().hasValue());
            if (fromFeature.left().hasValue()) {
                toLeftLinkOrValue.anon().value(fromFeature.left().anon().value());
            } else {
                toLeftLinkOrValue.anon().value(fromFeature.left().anon().featureId());
            }
            Cascade.Feature.LinkOrValue toRightLinkOrValue = toFeature.right();
            toRightLinkOrValue.hasValue(fromFeature.right().hasValue());
            if (fromFeature.right().hasValue()) {
                toRightLinkOrValue.anon().value(fromFeature.right().anon().value());
            } else {
                toRightLinkOrValue.anon().featureId(fromFeature.right().anon().featureId());
            }
            for (int r = 0; r < 3; r++) {
                var fromRect = fromFeature.rect(r);
                if (fromRect != null) {
                    var toRect = toFeature.rect(r);
                    toRect.x(fromRect.x());
                    toRect.y(fromRect.y());
                    toRect.width(fromRect.width());
                    toRect.height(fromRect.height());
                    toRect.weight(fromRect.weight());
                }
            }
        }

        for (int stageIdx = 0; stageIdx<fromCascade.stageCount(); stageIdx++) {
            Cascade.Stage fromStage =  fromCascade.stage(stageIdx);// stage(haarstage.id);
            Cascade.Stage toStage =  toCascade.stage(stageIdx);
            toStage.id(fromStage.id());
            toStage.threshold(fromStage.threshold());
            toStage.firstTreeId(fromStage.firstTreeId());
            toStage.treeCount(fromStage.treeCount());
        }
        for (int treeIdx=0; treeIdx <fromCascade.treeCount(); treeIdx++) {
            Cascade.Tree toTree =  toCascade.tree(treeIdx);
            Cascade.Tree fromTree =  fromCascade.tree(treeIdx);
            toTree.id(fromTree.id());
            toTree.firstFeatureId(fromTree.firstFeatureId());
            toTree.featureCount(fromTree.featureCount());
        }
        return toCascade;
    }
}
