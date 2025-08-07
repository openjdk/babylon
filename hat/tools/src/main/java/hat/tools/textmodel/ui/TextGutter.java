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

import javax.swing.JComponent;
import javax.swing.border.Border;
import javax.swing.border.CompoundBorder;
import javax.swing.border.EmptyBorder;
import javax.swing.border.MatteBorder;
import javax.swing.text.Element;
import javax.swing.text.Utilities;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Insets;
import java.awt.Point;
import java.awt.Rectangle;

public class TextGutter extends JComponent {
    private final static Border OUTER = new MatteBorder(0, 0, 0, 2, Color.GRAY);
    private final TextModelViewer lhs;
    private final TextModelViewer rhs;
    private final Color currentLineForeground;
    private final FontMetrics fontMetrics;
    private final int colWidth;
    private final Dimension size;

    @Override
    public Dimension getMinimumSize() {
        return size;
    }

    @Override
    public Dimension getPreferredSize() {
        return size;
    }

    public TextGutter(TextModelViewer lhs, TextModelViewer rhs) {
        this.lhs = lhs;
        this.rhs = rhs;
        setFont(this.lhs.styleMapper.jTextPane.getFont());
        setBorder(new CompoundBorder(OUTER, new EmptyBorder(0, 5, 0, 5)));
        this.fontMetrics = getFontMetrics(getFont());
        Insets insets = getInsets();
        colWidth = fontMetrics.charWidth('0') * 4; // for digits either side
        size = new Dimension(insets.left + insets.right + colWidth * 2, 1000 * fontMetrics.getHeight());
        this.currentLineForeground = Color.RED;
        this.lhs.styleMapper.jTextPane.addCaretListener(_ -> repaint());
        this.lhs.scrollPane.getViewport().addChangeListener(_->repaint());
        this.rhs.scrollPane.getViewport().addChangeListener(_->repaint());

    }

    void paintNumbers(Graphics g, TextModelViewer tv, int col) {
        var viewPort = tv.scrollPane.getViewport();
        var viewportPosition = viewPort.getViewPosition();
        int rowStartOffset = tv.styleMapper.jTextPane.viewToModel(new Point(0, viewportPosition.y));
        int endOffset = tv.styleMapper.jTextPane.viewToModel(new Point(0, viewportPosition.y + viewPort.getHeight()));
        Element root = tv.styleMapper.jTextPane.getDocument().getDefaultRootElement();
        while (rowStartOffset <= endOffset) {
            try {
                int caretPosition = tv.styleMapper.jTextPane.getCaretPosition();
                g.setColor(root.getElementIndex(rowStartOffset) == root.getElementIndex(caretPosition)
                        ? currentLineForeground
                        : getForeground());
                int index = root.getElementIndex(rowStartOffset);
                String lineNumber = (root.getElement(index).getStartOffset() == rowStartOffset) ? Integer.toString(index + 1) : "";
                int width = fontMetrics.stringWidth(lineNumber);
                Rectangle r = tv.styleMapper.jTextPane.modelToView(rowStartOffset);
                int x = this.getInsets().left + ((col == 0) ? 0 : colWidth * 2 - width);
                int y = r.y + r.height - fontMetrics.getDescent() - viewportPosition.y;
                g.drawString(lineNumber, x, y);
                rowStartOffset = Utilities.getRowEnd(tv.styleMapper.jTextPane, rowStartOffset) + 1;
            } catch (Exception e) {
                break;
            }
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        paintNumbers(g, lhs, 0);
        paintNumbers(g, rhs, 1);
    }

}
