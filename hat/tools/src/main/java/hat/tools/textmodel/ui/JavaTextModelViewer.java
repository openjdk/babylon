
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

import hat.tools.textmodel.JavaTextModel;
import hat.tools.textmodel.TextModel;

import java.awt.Font;
import java.awt.Graphics;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class JavaTextModelViewer extends TextModelViewer {
    FuncOpTextModelViewer funcOpTextModelViewer;
    Map<ElementSpan, List<ElementSpan>> javaToOp = new HashMap<>();

    static class JavaTextPane extends TextModelViewer.TextViewerPane<JavaTextModelViewer> {
        public void paintComponent(Graphics g) {
            super.paintComponent(g);
        }
        JavaTextPane(Font font, boolean editable) {
            super(font, editable);
        }
    }

    @Override
    public  TextModel createTextModel(String text) {
        return JavaTextModel.of(styleMapper.jTextPane.getText());
    }


    JavaTextModelViewer(TextModel textModel,StyleMapper styleMapper) {
        super(textModel, styleMapper);
        final var thisTextViewer = this;
        ((JavaTextPane) this.styleMapper.jTextPane).setViewer(this);
        styleMapper.jTextPane.addMouseListener(new MouseAdapter() {
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
