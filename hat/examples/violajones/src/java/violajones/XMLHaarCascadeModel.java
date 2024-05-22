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


import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import violajones.ifaces.Cascade;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Scanner;
import java.util.function.Consumer;
import java.util.function.Predicate;

public class XMLHaarCascadeModel implements Cascade {

    private final Element cascadeElement;

    static Optional<Element> selectChild(Element element, Predicate<Element> predicate) {
        NodeList nodes = element.getChildNodes();
        for (int i = 0; i < nodes.getLength(); i++) {
            if (nodes.item(i) instanceof Element childElement && predicate.test(childElement)) {
                return Optional.of(childElement);
            }
        }
        return Optional.empty();
    }

    static Optional<Element> selectChild(Element element, final String name) {
        return selectChild(element, (e) -> e.getNodeName().equals(name));
    }

    static void forEachElement(Element element, Predicate<Element> predicate, Consumer<Element> consumer) {
        NodeList nodes = element.getChildNodes();
        for (int i = 0; i < nodes.getLength(); i++) {
            if (nodes.item(i) instanceof Element xmle && predicate.test(xmle)) {
                consumer.accept(xmle);
            }
        }
    }
    static float getFloat(Element element, final String name) {
        return selectChild(element, name).map(value -> Float.parseFloat(value.getTextContent())).orElse(0f);
    }
    static short getShort(Element element, final String name) {
        return selectChild(element, name).map(value -> Short.parseShort(value.getTextContent())).orElse((short)0);
    }
    final  public List<Feature> features = new ArrayList<>();
    final  public List<Tree> trees = new ArrayList<>();
    final public List<Stage> stages = new ArrayList<>();

    @Override
    public Cascade.Feature feature(long idx) {
        return features.get((int)idx);
    }

    @Override
    public int featureCount() {
        return features.size();
    }

    @Override
    public void featureCount(int featureCount) {
       throw new IllegalStateException("featureCount(int featureCount) unimplemented ");
    }

    @Override
    public Cascade.Stage stage(long idx) {
        return stages.get((int)idx);
    }

    @Override
    public int stageCount() {
        return stages.size();
    }

    @Override
    public void stageCount(int stageCount) {
        throw new IllegalStateException("stageCount(int stageCount) unimplemented ");
    }

    @Override
    public Cascade.Tree tree(long idx) {
        return trees.get((int)idx);
    }

    @Override
    public int treeCount() {
        return trees.size();
    }

    @Override
    public void treeCount(int treeCount) {
        throw new IllegalStateException("void treeCount(int treeCount) unimplemented ");
    }

    @Override
    public int width() {
        return width;
    }

    @Override
    public void width(int width) {
        throw new IllegalStateException("void width(int width) unimplemented ");
    }

    @Override
    public int height() {
        return height;
    }

    @Override
    public void height(int height) {
        throw new IllegalStateException("void height(int height) unimplemented ");
    }

    static public class Feature implements Cascade.Feature {
        private final Element featureElement;
        private final Tree tree;

        @Override
        public int id() {
            return id;
        }

        @Override
        public float threshold() {
            return threshold;
        }

        @Override
        public void id(int id) {
            throw new IllegalStateException("void id(int id) unimplemented ");
        }

        @Override
        public void threshold(float threshold) {
            throw new IllegalStateException("void threshold(float threshold) unimplemented ");
        }

        @Override
        public Cascade.Feature.LinkOrValue left() {
            return left;
           // throw new IllegalStateException("Cascade.Feature.LinkOrValue left() unimplemented ");
        }

        @Override
        public Cascade.Feature.LinkOrValue right() {
            return right;
            //throw new IllegalStateException("Cascade.Feature.LinkOrValue right() unimplemented ");
        }

        @Override
        public Cascade.Feature.Rect rect(long idx) {
            if (rects.length>idx){
                return rects[(int)idx];
            }else{
                return null;
            }
        }


        static public class LinkOrValue implements Cascade.Feature.LinkOrValue, Cascade.Feature.LinkOrValue.Anon {
            private final boolean hasValue;


            private short featureId;
            private float value;

            public LinkOrValue(short featureId) {
                this.featureId = featureId;
                hasValue = false;
            }

            public LinkOrValue(float value) {
                this.value = value;
                hasValue = true;
            }

            @Override
            public boolean hasValue() {
                return hasValue;
            }

            @Override
            public void hasValue(boolean hasValue) {
                throw new IllegalStateException("void LinkOrValue(boolean ) unimplemented ");
            }

            @Override
            public Anon anon() {
                return this;
               // throw new IllegalStateException("Anon anon() unimplemented ");
            }

            @Override
            public int featureId() {
                return featureId;
            }

            @Override
            public float value() {
                return value;
            }

            @Override
            public void featureId(int featureId) {
                throw new IllegalStateException("void featureId(int featureId) unimplemented ");
            }

            @Override
            public void value(float value) {
                throw new IllegalStateException("void value(float value) unimplemented ");
            }
        }

        static public class Rect implements Cascade.Feature.Rect {
            private final Feature feature;
            private final Element rectElement;
            private final  byte x, y, width, height;
            private final float weight;

            public Rect(Feature feature, Element rectElement) {
                this.feature  = feature;
                this.rectElement = rectElement;

                Scanner rectScanner = new Scanner(this.rectElement.getTextContent());
                this.x =rectScanner.nextByte();
                this.y =rectScanner.nextByte();
                this.width=rectScanner.nextByte();
                this.height=rectScanner.nextByte();
                this.weight=rectScanner.nextFloat();
            }

            @Override
            public byte x() {
                return x;
            }

            @Override
            public byte y() {
                return y;
            }

            @Override
            public byte width() {
                return width;
            }

            @Override
            public byte height() {
                return height;
            }

            @Override
            public float weight() {
                return weight;
            }

            @Override
            public void x(byte x) {
                throw new IllegalStateException("void x(byte x) unimplemented ");
            }

            @Override
            public void y(byte y) {
                throw new IllegalStateException("void y(byte y) unimplemented ");
            }

            @Override
            public void width(byte width) {
                throw new IllegalStateException("void width(byte width) unimplemented ");
            }

            @Override
            public void height(byte height) {
                throw new IllegalStateException("void height(byte height) unimplemented ");
            }

            @Override
            public void weight(float height) {
                throw new IllegalStateException("void weight(float weight) unimplemented ");
            }
        }

        private final short id;
        int rectCount;
        public final Rect[] rects;
        private final float threshold;

        final  public LinkOrValue left;
        final  public LinkOrValue right;


        public Feature(Tree tree, Element featureElement, int id) {
            this.tree =tree;
            this.featureElement = featureElement;
            this.id = (short) id;
            this.rectCount = 0;
            this.rects =  new Rect[3];
            this.threshold = getFloat(this.featureElement, "threshold");

            left = (selectChild(this.featureElement,"left_val")).isPresent()
                    ? new Feature.LinkOrValue(getFloat(this.featureElement,"left_val"))
                    : new Feature.LinkOrValue(getShort(this.featureElement,"left_node"));
            right = (selectChild(this.featureElement, "right_val")).isPresent()
                    ? new Feature.LinkOrValue(getFloat(this.featureElement, "right_val"))
                    : new Feature.LinkOrValue(getShort(this.featureElement,"right_node"));

            selectChild(this.featureElement, "feature").flatMap(featureXML -> selectChild(featureXML, "rects")).ifPresent(rectsXML -> {
                forEachElement(rectsXML, e -> e.getNodeName().equals("_"),
                        (rectXMLElement) -> rects[this.rectCount++] = new Rect(this, rectXMLElement)
                );
            });
        }
    }
    static public class Tree implements Cascade.Tree {
        private final Stage stage;
        final  Element treeElement;
        final  int id;

        @Override
        public void id(int id) {
            throw new IllegalStateException("void id(int  id) unimplemented ");
        }

        @Override
        public void firstFeatureId(short firstFeatureId) {
            throw new IllegalStateException("void firstFeatureId(short  firstFeatureId) unimplemented ");
        }

        @Override
        public void featureCount(short featureCount) {
            throw new IllegalStateException("void featureCount(short  featureCount) unimplemented ");
        }

        @Override
        public int id() {
            return id;
        }

        @Override
        public short firstFeatureId() {
            return firstFeatureId;
        }

        @Override
        public short featureCount() {
            return featureCount;
        }

        public short featureCount;
        public short firstFeatureId = -1;

        public Tree(Stage stage, Element treeElement, int id) {
            this.stage = stage;
            this.treeElement = treeElement;
            this.id = id;
            forEachElement(treeElement,e->e.getNodeName().equals("_"),
                    featureXMLElement -> {
                        Feature feature= new Feature(this, featureXMLElement, stage.haarCascade.features.size());
                        stage.haarCascade.features.add(feature);
                        if (firstFeatureId == -1) {
                            firstFeatureId = feature.id;
                        }
                        featureCount = (short) (feature.id - firstFeatureId + 1);
                    });
        }
    }
    public static class Stage implements Cascade.Stage {

        private final XMLHaarCascadeModel haarCascade;
        private final Element stageElement;
        @Override
        public float threshold() {
            return threshold;
        }

        @Override
        public short firstTreeId() {
            return firstTreeId;
        }

        @Override
        public short treeCount() {
            return treeCount;
        }


        @Override
        public int id() {
           return id;
        }

        @Override
        public void id(int id) {
            throw new IllegalStateException("void id(int  id) unimplemented ");
        }

        @Override
        public void threshold(float threshold) {
            throw new IllegalStateException("void threshold(float  threshold) unimplemented ");
        }

        @Override
        public void firstTreeId(short firstTreeId) {
            throw new IllegalStateException("void firstTreeId(short  firstTreeId) unimplemented ");
        }

        @Override
        public void treeCount(short treeCount) {
            throw new IllegalStateException("void treeCount(short  treeCount) unimplemented ");
        }

        final  public int id;

        final  public float threshold;
        public short firstTreeId = -1;
        public short treeCount;

        public Stage(XMLHaarCascadeModel haarCascade, Element stageElement, int id) {
            this.haarCascade = haarCascade;
            this.stageElement = stageElement;
            this.id = id;
            this.threshold = getFloat(this.stageElement,"stage_threshold");
            selectChild(this.stageElement, "trees").ifPresent(treeXML-> {
                forEachElement(treeXML, e->e.getNodeName().equals("_"), treeXMLElement -> {
                            Tree tree = new Tree(this, treeXMLElement, haarCascade.trees.size());
                            haarCascade.trees.add(tree);
                            if (firstTreeId == -1) {
                                firstTreeId = (short) tree.id;
                            }
                            treeCount = (short) (tree.id - firstTreeId + 1);
                        }
                );
            });

        }
    }

    final public int width;
    final  public int height;
    public static XMLHaarCascadeModel load(InputStream is) throws IOException, SAXException, ParserConfigurationException {
        if (is == null) {
            throw new IllegalArgumentException("input == null!");
        }
        org.w3c.dom.Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(is);
        doc.getDocumentElement().normalize();
        Element root  = doc.getDocumentElement();
        Element cascadeElement = selectChild(root, (e) -> e.hasAttribute("type_id")).get();

            return new XMLHaarCascadeModel(cascadeElement);
    }

    XMLHaarCascadeModel(Element cascadeElement){
        this.cascadeElement = cascadeElement;
        Scanner sizeScanner = new Scanner(selectChild(cascadeElement,"size").get().getTextContent());
        this.width = sizeScanner.nextInt();
        this.height = sizeScanner.nextInt();
        selectChild(cascadeElement, "stages").ifPresent(stagesXML->
                forEachElement(stagesXML,e->e.getNodeName().equals("_"),
                        (stageXMLElement) ->
                            stages.add(new Stage(this, stageXMLElement, stages.size()))
                        )
        );


    }
}
