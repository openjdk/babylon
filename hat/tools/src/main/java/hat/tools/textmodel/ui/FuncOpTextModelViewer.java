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
import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.TextModel;

import javax.swing.JTextPane;
import javax.swing.text.Element;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Stroke;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.Rectangle2D;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FuncOpTextModelViewer extends BabylonTextModelViewer {

    JavaTextModelViewer javaTextModelViewer;
    Map<ElementSpan, List<ElementSpan>> opToJava = new HashMap<>();
    int lineNumber = 0;

    static class FuncOpTextPane extends TextViewerPane<FuncOpTextModelViewer> {

        static private final Polygon arrowHead = new Polygon();

        static {
            arrowHead.addPoint(3, 0);
            arrowHead.addPoint(-3, -3);
            arrowHead.addPoint(-3, 3);
        }

        void arrow(Graphics2D g2d, Element from, Element to) {
            if (viewer.getRect(from) instanceof Rectangle2D.Double frect
                    && viewer.getRect(to) instanceof Rectangle2D.Double trect) {
                g2d.setColor(Color.BLACK);
                Line2D.Double arrow = new Line2D.Double(
                        frect.getBounds().getMinX(), frect.getBounds().getCenterY(),
                        trect.getBounds().getMinX(), trect.getBounds().getCenterY());
                final AffineTransform tx = AffineTransform.getTranslateInstance(arrow.x2, arrow.y2);
                tx.rotate(Math.atan2(arrow.y2 - arrow.y1, arrow.x2 - arrow.x1));// point the arrow to the target
                g2d.fill(tx.createTransformedShape(arrowHead));
                // Create a copy of the Graphics instance
                Graphics2D g2d2 = (Graphics2D) g2d.create();
                // Set the stroke of the copy, not the original
                Stroke dashed = new BasicStroke(2, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL,
                        0, new float[]{4f, 2f, 5f, 1f}, 0);
                g2d2.setStroke(dashed);

                //    g2d2.setStroke(new BasicStroke(1));
                g2d2.draw(arrow);
            }
        }


        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            var tm = viewer.textModel;
            ((BabylonTextModel) tm).ssaEdgeList.stream().forEach(edge -> {
                var ssaRef = edge.ssaRef();
                var ssaDef = edge.ssaDef();
                if (ssaRef.pos().line() == viewer.lineNumber || ssaDef.pos().line() == viewer.lineNumber) {
                    var ssaDefElement = viewer.getElement(ssaDef.startOffset());
                    var ssaRefElement = viewer.getElement(ssaRef.startOffset());
                    arrow(g2d, ssaRefElement, ssaDefElement);
                }
            });
        }

        FuncOpTextPane(Font font, boolean editable) {
            super(font,editable);
        }
    }
    @Override
    public  TextModel createTextModel(String text) {
        return BabylonTextModel.of(jTextPane.getText());
    }
    FuncOpTextModelViewer(TextModel textModel, FuncOpTextPane jTextPane, boolean dark) {
        super(textModel, jTextPane, dark);
        javaTextModelViewer = null;
        final var thisTextViewer = this;
        jTextPane.setViewer(this);

        jTextPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                var clickedElement = getElementFromMouseEvent(e);
                removeHighlights();
                if (javaTextModelViewer != null) {
                    javaTextModelViewer.removeHighlights();
                    if (clickedElement != null) {
                        var elementsReferencedByClickedElement = opToJava.keySet().stream()
                                .filter(fromElementSpan ->
                                        fromElementSpan.includes(clickedElement.getStartOffset())
                                ).toList();
                        if (!elementsReferencedByClickedElement.isEmpty()) {
                            lineNumber = getLine(elementsReferencedByClickedElement.getFirst().element()) + 1;
                            elementsReferencedByClickedElement.forEach(fromElementSpan -> {
                                thisTextViewer.highlight(fromElementSpan, opToJava.get(fromElementSpan));
                            });
                        }
                    }
                }
            }
        });

    }
}
