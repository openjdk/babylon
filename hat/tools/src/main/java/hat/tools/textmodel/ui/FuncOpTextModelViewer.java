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
package hat.tools.textmodel.ui;

import hat.tools.textmodel.BabylonTextModel;
import hat.tools.textmodel.TextModel;

import javax.swing.JTextPane;
import javax.swing.JViewport;
import javax.swing.text.BadLocationException;
import javax.swing.text.Element;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Shape;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FuncOpTextModelViewer extends AbstractTextModelViewer {
    JavaTextModelViewer javaTextModelViewer;
    Map<ElementSpan, List<ElementSpan>> ssaIdFromToMap = new HashMap<>();
    Map<ElementSpan, List<ElementSpan>> ssaIdToFromMap = new HashMap<>();
    Map<Integer, ElementSpan> ssaIdToElement = new HashMap<>();
    Map<ElementSpan, List<ElementSpan>> opToJava = new HashMap<>();

    static class FuncOpTextPane extends JTextPane {
       private  FuncOpTextModelViewer viewer;
        List<Shape> shapes = new ArrayList<>();
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
                Graphics2D g2d = (Graphics2D) g;
                g2d.setColor(Color.BLACK);
                shapes.forEach(g2d::fill);
        }

        FuncOpTextPane(Font font) {
            super.setFont(font);
            this.viewer = null;
        }

        void setViewer(FuncOpTextModelViewer viewer) {
            this.viewer = viewer;
            viewer.ssaIdFromToMap.forEach((from, toList) -> {
                var fromElement = from.element();
                try {
                    var fromPoint = from.textViewer().jtextPane.modelToView2D(fromElement.getStartOffset());

                    if (fromPoint != null) {
                        Line2D line = new Line2D.Float(
                                fromPoint.getBounds().x, fromPoint.getBounds().y + 100, 0, 0);

                        shapes.add(line);
                        toList.forEach(to -> {
                            var toElement = to.element();
                            try {
                                var toPoint = to.textViewer().jtextPane.modelToView2D(toElement.getStartOffset());
                                Line2D line2D = new Line2D.Float(
                                        fromPoint.getBounds().x, fromPoint.getBounds().y + 100, toPoint.getBounds().x, toPoint.getBounds().y);
                                shapes.add(line2D);
                            } catch (BadLocationException e) {
                                throw new RuntimeException(e);
                            }
                        });
                    }
                    } catch(BadLocationException e){
                        throw new RuntimeException(e);
                    }


            });

        }
    }
    FuncOpTextModelViewer(TextModel textModel, Font font, boolean dark) {
        super(textModel, new FuncOpTextPane(font), font, dark);


        jtextPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                var clicked = getElementFromMouseEvent(e);
                removeHighlights();
                javaTextModelViewer.removeHighlights();
                if (clicked != null) {
                    if (opToJava.keySet().stream()
                            .anyMatch(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))) {
                        opToJava.keySet().stream().
                                filter(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset()))
                                .forEach(fromElementSpan -> {
                                    fromElementSpan.textViewer().highLight(fromElementSpan.element());
                                    opToJava.get(fromElementSpan).forEach(targetElementSpan -> {
                                        Element targetElement = targetElementSpan.element();
                                        targetElementSpan.textViewer().highLight(targetElement);
                                        targetElementSpan.textViewer().scrollTo(targetElement);
                                    });
                                });
                    } else {
                        System.out.println("not a locationmapping  from op");
                    }
                } else {
                    System.out.println("nothing from op");
                }
            }
        });
        textModel.find(true, t -> t instanceof BabylonTextModel.BabylonSSARef, t -> {
            var ssaRef = (BabylonTextModel.BabylonSSARef) t;
            ElementSpan babylonSSARefElement = new ElementSpan.Impl(ssaRef, this, this.getElement(ssaRef.startOffset()));
            this.ssaIdToElement.put(ssaRef.id, babylonSSARefElement);
            this.ssaIdToFromMap.computeIfAbsent(babylonSSARefElement, _ -> new ArrayList<>());
            this.ssaIdFromToMap.computeIfAbsent(babylonSSARefElement, _ -> new ArrayList<>());
        });

        ((BabylonTextModel) textModel).ssaEdgeList.stream().forEach(edge -> {
            var ssaRef = edge.ssaRef();
            var ssaDef = edge.ssaDef();
            var ssaDefElement = this.getElement(ssaDef.startOffset());
            var ssaRefElement = this.getElement(ssaRef.endOffset());

        });
        ((FuncOpTextPane) this.jtextPane).setViewer(this);

    }
}
