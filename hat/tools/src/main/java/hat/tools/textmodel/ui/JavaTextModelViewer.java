
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

import hat.tools.textmodel.TextModel;

import javax.swing.JTextPane;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JavaTextModelViewer extends AbstractTextModelViewer {
    FuncOpTextModelViewer funcOpTextModelViewer;
    Map<ElementSpan, List<ElementSpan>> javaToOp = new HashMap<>();

    static class JavaTextPane extends JTextPane {
        private JavaTextModelViewer viewer;
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
        }

        JavaTextPane(Font font) {
            super.setFont(font);
        }
        void setViewer(JavaTextModelViewer viewer) {
            this.viewer = viewer;
        }
    }

    JavaTextModelViewer(TextModel textModel, Font font, boolean dark) {
        super(textModel, new JavaTextPane(font), font, dark);
        final var thisTextViewer = this;
        ((JavaTextPane) this.jtextPane).setViewer(this);
        jtextPane.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                var clickedElement = getElementFromMouseEvent(e);
                funcOpTextModelViewer.removeHighlights();
                removeHighlights();
                if (clickedElement != null) {
                    var elementsReferencedByClickedElement = javaToOp.keySet().stream()
                            .filter(fromElementSpan ->
                                    fromElementSpan.includes(clickedElement.getStartOffset())
                            ).toList();
                    if (!elementsReferencedByClickedElement.isEmpty()) {
                        elementsReferencedByClickedElement.forEach(fromElementSpan -> {
                            thisTextViewer.highlight(fromElementSpan, javaToOp.get(fromElementSpan));
                        });
                    }
                }
            }
        });
    }

}
