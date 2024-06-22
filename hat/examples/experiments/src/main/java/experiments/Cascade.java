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
package experiments;

import hat.buffer.Buffer;
import hat.buffer.CompleteBuffer;

public interface Cascade extends CompleteBuffer {
    int width();
    void width(int width);
    int height();
    void height(int height);
    interface Feature extends Buffer.StructChild {
        int id();
        float threshold();
        void id(int id);
        void threshold(float threshold);
        interface LinkOrValue extends Buffer.StructChild {
            interface Anon extends Buffer.UnionChild {
                int featureId();
                void featureId(int featureId);
                float value();
                void value(float value);
            }
            boolean hasValue();
            void hasValue(boolean hasValue);
            Anon anon();
        }
        LinkOrValue left();
        LinkOrValue right();
        interface Rect extends Buffer.StructChild {
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
        Rect rect(long idx);
    }
    int featureCount();
    void featureCount(int featureCount);
    Feature feature(long idx);
    interface Stage extends Buffer.StructChild {
        float threshold();
        short firstTreeId();
        short treeCount();
        int id();
        void id(int id);
        void threshold(float threshold);
        void firstTreeId(short firstTreeId);
        void treeCount(short treeCount);
    }
    int stageCount();
    void stageCount(int stageCount);
    Stage stage(long idx);
    interface Tree extends Buffer.StructChild {
        void id(int id);
        void firstFeatureId(short firstFeatureId);
        void featureCount(short featureCount);
        int id();
        short firstFeatureId();
        short featureCount();
    }
    int treeCount();
    void treeCount(int treeCount);
    Tree tree(long idx);


    Schema<Cascade> schema = Schema.of(Cascade.class, cascade -> cascade
            .fields("width","height")
            .arrayLen("featureCount").array("feature", feature -> feature
                    .fields("id","threshold")
                    .fields("left","right",linkOrValue->linkOrValue
                            .field("hasValue")
                            .field("anon", anon->anon.fields("featureId","value"))
                    )
                    .array("rect", 3 , rect->rect.fields("x","y","width","height","weight"))
            )
            .arrayLen("treeCount").array("tree",tree->tree.fields("id","featureCount","firstFeatureId"))
            .arrayLen("stageCount").array("stage", stage->stage.fields("id","threshold","treeCount","firstTreeId"))
    );


}
