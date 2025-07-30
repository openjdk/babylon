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
import javax.swing.text.BadLocationException;
import javax.swing.text.Element;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class FuncOpTextModelViewer extends AbstractTextModelViewer {

    JavaTextModelViewer javaTextModelViewer;
    Map<ElementSpan, List<ElementSpan>> opToJava = new HashMap<>();
    int lineNumber =0;
    static class FuncOpTextPane extends JTextPane {
        private FuncOpTextModelViewer viewer;

        static private final Polygon arrowHead = new Polygon();

        static {
            arrowHead.addPoint(3, 0);
            arrowHead.addPoint(-3, -3);
            arrowHead.addPoint(-3, 3);
        }

        void arrow(Graphics2D g2d, Element from, Element to) {
            try {
                var fromPoint1 = viewer.jtextPane.modelToView2D(from.getStartOffset());
                var fromPoint2 = viewer.jtextPane.modelToView2D(from.getEndOffset());
                var fromRect = new Rectangle2D.Double(fromPoint1.getBounds().getMinX(), fromPoint1.getMinY()
                        , fromPoint2.getBounds().getWidth(), fromPoint2.getBounds().getHeight());
                var toPoint1 = viewer.jtextPane.modelToView2D(to.getStartOffset() + 3);
                var toPoint2 = viewer.jtextPane.modelToView2D(to.getEndOffset());
                var toRect = new Rectangle2D.Double(toPoint1.getBounds().getMinX(), toPoint1.getMinY()
                        , toPoint2.getBounds().getWidth(), toPoint2.getBounds().getHeight());
                g2d.setColor(Color.GRAY);
                var x0 = fromRect.getBounds().getMinX();
                var y0 = fromRect.getBounds().getCenterY();
                var x1 = toRect.getBounds().getMinX();
                var y1 = toRect.getBounds().getCenterY();
                final AffineTransform tx = AffineTransform.getTranslateInstance(x1, y1);
                tx.rotate(Math.atan2(y1 - y0, x1 - x0));
                g2d.fill(tx.createTransformedShape(arrowHead));
                var line = new Line2D.Double(x0, y0, x1, y1);
                g2d.setStroke(new BasicStroke(2));
                g2d.draw(line);
            } catch (BadLocationException e) {
                throw new RuntimeException(e);
            }
        }


        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            ((BabylonTextModel) viewer.textModel).ssaEdgeList.stream().forEach(edge -> {
                var ssaRef = edge.ssaRef();
                var ssaDef = edge.ssaDef();
                if (ssaRef.pos().line() == viewer.lineNumber || ssaDef.pos().line() == viewer.lineNumber) {
                    var ssaDefElement = viewer.getElement(ssaDef.startOffset());
                    var ssaRefElement = viewer.getElement(ssaRef.startOffset());
                    arrow(g2d, ssaRefElement, ssaDefElement);
                }
            });
        }

        FuncOpTextPane(Font font) {
            super.setFont(font);
            this.viewer = null;
        }

        void setViewer(FuncOpTextModelViewer viewer) {
            this.viewer = viewer;
        }
    }

    FuncOpTextModelViewer(TextModel textModel, Font font, boolean dark) {
        super(textModel, new FuncOpTextPane(font), font, dark);
        ((FuncOpTextPane) this.jtextPane).setViewer(this);

        jtextPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                var clicked = getElementFromMouseEvent(e);
                removeHighlights();
                javaTextModelViewer.removeHighlights();
                if (clicked != null) {
                    var optionalElementSpan = opToJava.keySet().stream()
                            .filter(fromElementSpan -> fromElementSpan.includes(clicked.getStartOffset())).findFirst();
                    if (optionalElementSpan.isPresent()) {
                        ElementSpan elementSpan = optionalElementSpan.get();
                        lineNumber = getLine(elementSpan.element().getStartOffset())+1;
                    }
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

    }
}
